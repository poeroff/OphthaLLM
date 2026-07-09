"""
BLIP-2 스타일 CT 판독문 생성 모델

Stage 1 — Frozen Image Encoder + Q-Former 동시 학습
    ITC : Q-Former query 출력 ↔ frozen 텍스트 CLS  (contrastive)
    ITM : Q-Former + 텍스트 토큰 → 매칭 이진 분류
    ITG : Q-Former 출력을 prefix로 → BioGPT causal LM

Stage 2 — Frozen Image Encoder + Frozen Q-Former + BioGPT LoRA
    Q-Former 출력 → linear projection → BioGPT prefix
    BioGPT LoRA 파라미터만 학습
"""

import copy
import math
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

# Momentum encoder (ALBEF): EMA decay
_MOMENTUM = 0.995

# Negative cross-attention (MINDiff): dominant patch 억제 하이퍼파라미터
_NEG_LAMBDA     = 0.1   # random negative 전환 후 signal 보존 위해 0.5→0.1
_NEG_TOPK_RATIO = 0.2   # top-k 비율 (이미지 패치 수 × 0.2 = top-20%)



# ── 유틸 ──────────────────────────────────────────────────────────────────────

def _unwrap(m):
    return m.module if isinstance(m, nn.DataParallel) else m


# ── ITC 모듈 ──────────────────────────────────────────────────────────────────

class CrossAttentionFusion(nn.Module):
    """이미지 시퀀스 ↔ 텍스트 시퀀스 간 cross-attention 후 CLS 반환."""
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.img_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.txt_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.img_norm = nn.LayerNorm(embed_dim)
        self.txt_norm = nn.LayerNorm(embed_dim)

    def forward(self, img_seq, txt_seq):
        img_out, _ = self.img_attn(img_seq, txt_seq, txt_seq)
        txt_out, _ = self.txt_attn(txt_seq, img_seq, img_seq)
        img_out = self.img_norm(img_seq + img_out)
        txt_out = self.txt_norm(txt_seq + txt_out)
        return img_out[:, 0], txt_out[:, 0]   # (B, D), (B, D)


class ProjectionHead(nn.Module):
    """3-layer MLP Projection Head."""
    def __init__(self, in_dim, out_dim=512, hidden_dim=2048):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.layer2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.layer3 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        return self.layer3(self.layer2(self.layer1(x)))


def info_nce_loss(img_emb, txt_emb, log_temp):
    """InfoNCE (symmetric cross-entropy). Inputs are unnormalized."""
    img = F.normalize(img_emb, dim=-1)
    txt = F.normalize(txt_emb, dim=-1)
    temp   = log_temp.exp().clamp(max=100)
    logits = img @ txt.T * temp          # (B, B)
    labels = torch.arange(logits.size(0), device=logits.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


def soft_info_nce_loss(img_emb, txt_emb, log_temp, soft_i2t, soft_t2i):
    """Soft InfoNCE with momentum soft targets (ALBEF 방식).
    hard one-hot 대신 momentum 모델의 출력 분포를 teacher signal로 사용.
    → Q-Former가 특정 학습 쌍을 암기하는 대신 분포를 학습하도록 강제.
    """
    img    = F.normalize(img_emb.float(), dim=-1)
    txt    = F.normalize(txt_emb.float(), dim=-1)
    temp   = log_temp.exp().clamp(max=100)
    logits = img @ txt.T * temp                              # (B, B)
    loss   = (
        -(soft_i2t * F.log_softmax(logits,   dim=-1)).sum(-1).mean()
        -(soft_t2i * F.log_softmax(logits.T, dim=-1)).sum(-1).mean()
    ) / 2
    return loss


# ── Negative Cross-Attention (MINDiff Q-Former 적용) ─────────────────────────

class NegativeCrossAttention(nn.Module):
    """MINDiff-style negative attention for Q-Former cross-attention.

    일반 cross-attention: out = Attn(Q, K, V)
    Negative attention:  out = Attn(Q, K, V) − λ · NegAttn(Q, K_topk, V_topk)

    각 query가 지나치게 집중하는 top-k 이미지 패치의 기여를 λ 만큼 억제.
    → Q-Former가 특정 CT 패치 패턴(환자별 특이점)을 암기하는 것을 방지.
    → 학습 중에만 활성화 (inference에서는 standard attention 동일).
    """

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float,
                 neg_lambda: float = _NEG_LAMBDA,
                 neg_topk_ratio: float = _NEG_TOPK_RATIO):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.H              = num_heads
        self.d              = hidden_dim // num_heads
        self.D              = hidden_dim
        self.scale          = self.d ** -0.5
        self.neg_lambda     = neg_lambda
        self.neg_topk_ratio = neg_topk_ratio

        self.q_proj    = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj    = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj    = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj  = nn.Linear(hidden_dim, hidden_dim)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        """
        query     : (B, Nq, D)  Q-Former 쿼리 토큰
        key_value : (B, L,  D)  frozen 이미지 패치 토큰
        return    : (B, Nq, D)
        """
        B, Nq, D = query.shape
        L        = key_value.size(1)
        H, d     = self.H, self.d

        Q = self.q_proj(query    ).reshape(B, Nq, H, d).permute(0, 2, 1, 3)  # (B,H,Nq,d)
        K = self.k_proj(key_value).reshape(B, L,  H, d).permute(0, 2, 1, 3)  # (B,H,L, d)
        V = self.v_proj(key_value).reshape(B, L,  H, d).permute(0, 2, 1, 3)  # (B,H,L, d)

        # ── Full cross-attention ───────────────────────────────────────────
        scores = (Q @ K.transpose(-2, -1)) * self.scale   # (B, H, Nq, L)
        attn   = scores.softmax(dim=-1)
        attn   = self.attn_drop(attn)
        out    = attn @ V                                   # (B, H, Nq, d)

        # ── Negative attention (학습 중에만) ───────────────────────────────
        # head-평균 attention으로 dominant 패치 식별 → neg_lambda 만큼 억제
        if self.training and self.neg_lambda > 0.0:
            top_k     = max(1, int(L * self.neg_topk_ratio))
            mean_attn = attn.mean(dim=1)                         # (B, Nq, L)
            topk_idx  = mean_attn.topk(top_k, dim=-1).indices   # (B, Nq, top_k)

            # top-k 위치만 통과, 나머지 -inf 마스킹
            neg_mask = torch.full((B, Nq, L), float('-inf'),
                                  device=query.device, dtype=scores.dtype)
            neg_mask.scatter_(-1, topk_idx, 0.0)

            # head-평균 score + 마스크 → negative attention weight
            neg_scores = scores.mean(dim=1) + neg_mask            # (B, Nq, L)
            neg_attn   = neg_scores.softmax(dim=-1)               # (B, Nq, L)
            V_mean     = V.mean(dim=1)                            # (B, L,  d)
            neg_out    = (neg_attn @ V_mean).unsqueeze(1)         # (B, 1, Nq, d)
            out        = out - self.neg_lambda * neg_out          # broadcast → (B,H,Nq,d)

        out = out.permute(0, 2, 1, 3).reshape(B, Nq, D)
        return self.out_proj(out)


# ── Q-Former 단일 레이어 ──────────────────────────────────────────────────────

class QFormerLayer(nn.Module):
    """
    BLIP-2 Q-Former 레이어:
      1. Query self-attention  (쿼리끼리 + 선택적으로 텍스트 토큰 포함)
      2. Cross-attention       (쿼리 → 이미지 토큰)
      3. FFN
    """

    def __init__(self, hidden_dim: int, num_heads: int, ffn_dim: int, dropout: float):
        super().__init__()

        # 1) Self-attention (query ↔ query, 또는 query ↔ query+text)
        self.self_attn  = nn.MultiheadAttention(hidden_dim, num_heads,
                                                 dropout=dropout, batch_first=True)
        self.norm1      = nn.LayerNorm(hidden_dim)
        self.drop1      = nn.Dropout(dropout)

        # 2) Cross-attention (query → frozen image tokens)
        #    NegativeCrossAttention: MINDiff-style dominant patch 억제
        self.cross_attn = NegativeCrossAttention(hidden_dim, num_heads, dropout)
        self.norm2      = nn.LayerNorm(hidden_dim)
        self.drop2      = nn.Dropout(dropout)

        # 3) FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
        )
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.drop3 = nn.Dropout(dropout)

    def forward(
        self,
        queries,          # (B, Nq, D)
        image_tokens,     # (B, L_img, D)   frozen CLIP 출력
        text_tokens=None, # (B, L_txt, D)   ITM/ITG 시 텍스트 토큰 포함
        self_attn_mask=None,  # causal mask for ITG
    ):
        # 1) self-attention: query만, 또는 query+text 합쳐서
        if text_tokens is not None:
            seq    = torch.cat([queries, text_tokens], dim=1)   # (B, Nq+Lt, D)
            Nq     = queries.size(1)
        else:
            seq    = queries
            Nq     = queries.size(1)

        sa_out, _ = self.self_attn(seq, seq, seq, attn_mask=self_attn_mask)
        seq       = self.norm1(seq + self.drop1(sa_out))
        queries   = seq[:, :Nq]   # query 부분만 추출

        # 2) cross-attention: queries → image tokens (NegativeCrossAttention)
        ca_out  = self.cross_attn(queries, image_tokens)
        queries = self.norm2(queries + self.drop2(ca_out))

        # 3) FFN
        queries   = self.norm3(queries + self.drop3(self.ffn(queries)))

        if text_tokens is not None:
            # text 부분도 self-attn 결과 반환 (ITG 시 LM 손실에 사용)
            text_out = seq[:, Nq:]
            return queries, text_out

        return queries, None


# ── Q-Former ──────────────────────────────────────────────────────────────────

class QFormer(nn.Module):
    """
    Args:
        num_queries  : 학습 가능한 쿼리 토큰 수 (BLIP-2 논문 = 32)
        hidden_dim   : M3D-CLIP embed_dim 과 일치 (768)
        num_heads    : attention head 수
        num_layers   : transformer 레이어 수
        ffn_dim      : FFN 내부 차원
        dropout      : dropout 확률
    """

    def __init__(
        self,
        num_queries:   int   = 32,
        hidden_dim:    int   = 768,
        num_heads:     int   = 12,
        num_layers:    int   = 6,
        ffn_dim:       int   = 3072,
        dropout:       float = 0.1,
    ):
        super().__init__()

        self.num_queries = num_queries
        self.hidden_dim  = hidden_dim

        # 학습 가능한 쿼리 벡터
        self.query_tokens = nn.Parameter(
            torch.zeros(1, num_queries, hidden_dim)
        )
        nn.init.trunc_normal_(self.query_tokens, std=0.02)

        self.layers = nn.ModuleList([
            QFormerLayer(hidden_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])

    # ── forward (이미지 토큰 → query 출력) ─────────────────────────────────
    def forward(
        self,
        image_tokens,     # (B, L_img, D)
        text_tokens=None, # (B, L_txt, D)  ITM/ITG 시 입력
        self_attn_mask=None,
    ):
        B = image_tokens.size(0)
        queries = self.query_tokens.expand(B, -1, -1)  # (B, Nq, D)

        for layer in self.layers:
            queries, text_tokens = layer(
                queries, image_tokens,
                text_tokens=text_tokens,
                self_attn_mask=self_attn_mask,
            )

        return queries, text_tokens   # (B, Nq, D), (B, Lt, D) or None

    # ── LM prefix: query 출력 그대로 (Stage 2에서 BioGPT prefix로 사용) ──
    def get_lm_prefix(self, queries):
        return queries                                # (B, Nq, D)


# ── BLIP 이미지 인코더 (LO-VLM fine-tuned, frozen) ───────────────────────────

_BLIP_VIS_ID = "QIAIUNCC/LO-VLM"
_BERT_ID     = "emilyalsentzer/Bio_ClinicalBERT"


class BLIPImageEncoder(nn.Module):
    """LO-VLM fine-tuned BLIP vision encoder (OCT에 특화, frozen).
    Input : pixel_values (B, 3, 384, 384)
    Output: last_hidden_state (B, 577, 768)
    """

    def __init__(self):
        super().__init__()
        from transformers import BlipForConditionalGeneration
        print(f"  [BLIPImageEncoder] 로드: {_BLIP_VIS_ID}")
        full = BlipForConditionalGeneration.from_pretrained(
            _BLIP_VIS_ID, torch_dtype=torch.float32
        )
        self.model = full.vision_model  # BlipVisionModel
        del full                        # text_decoder 메모리 해제
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

    @torch.no_grad()
    def forward(self, pixel_values):
        """pixel_values: (B, 3, 384, 384) → (B, 577, 768)"""
        with torch.backends.cudnn.flags(enabled=False):
            return self.model(pixel_values).last_hidden_state.float()


class BERTTextEncoder(nn.Module):
    """Bio_ClinicalBERT frozen text encoder.
    BLIPImageEncoder와 함께 사용하는 텍스트 인코더.
    """

    def __init__(self):
        super().__init__()
        from transformers import BertModel, BertTokenizer
        print(f"  [BERTTextEncoder] 로드: {_BERT_ID}")
        self.model = BertModel.from_pretrained(_BERT_ID)
        self.tokenizer = BertTokenizer.from_pretrained(_BERT_ID, model_max_length=512)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

    @torch.no_grad()
    def encode_cls(self, input_ids, attention_mask):
        """ITC용 CLS 토큰: (B, 768)"""
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state[:, 0].float()

    def encode_cls_grad(self, input_ids, attention_mask):
        """gradient 허용 버전 (unfrozen layer 학습 시 사용)"""
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state[:, 0].float()

    @torch.no_grad()
    def get_token_embeddings(self, input_ids):
        """word embedding lookup (non-contextual): (B, L, 768)"""
        return self.model.embeddings.word_embeddings(input_ids).float()

    @torch.no_grad()
    def get_contextual_token_embeddings(self, input_ids, attention_mask):
        """BERT full forward contextual embeddings: (B, L, 768)"""
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state.float()




# ── ITM Head ──────────────────────────────────────────────────────────────────

class ITMHead(nn.Module):
    """text-conditioned Q-Former 출력 → match/no-match 이진 분류."""
    def __init__(self, hidden_dim: int = 768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.4),   # 0.1→0.2→0.3→0.4: ITMHead 자체 암기 억제
            nn.Linear(hidden_dim // 2, 2),
        )

    def forward(self, x):
        return self.net(x)


# ── BLIP2CTModel: Stage 1 학습을 위한 통합 모델 ───────────────────────────────

class BLIP2CTModel(nn.Module):
    """
    Stage 1 통합 모델.

    forward() 는 loss dict 반환:
        'itc'      : InfoNCE loss
        'q_itc'    : Q-Former 보조 ITC loss
        'q_tc_itc' : text-conditioned Q-Former ITC loss
        'itm'      : cross-entropy (match/no-match)
    """

    _ALPHA_SOFT = 0.4   # ALBEF soft/hard blending 비율

    def __init__(
        self,
        image_encoder:  BLIPImageEncoder,
        text_encoder:   BERTTextEncoder,
        qformer:        QFormer,
        temp_init:      float = 0.07,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder  = text_encoder
        self.qformer       = qformer

        # ITC: CrossAttentionFusion + ProjectionHead (best.pt로 초기화)
        self.itc_fusion = CrossAttentionFusion(768)
        self.itc_proj   = ProjectionHead(768, out_dim=512)

        # Q-Former 보조 ITC projection — query-only / text-conditioned 분리
        # (ICLR 2024: 서로 다른 feature geometry를 같은 head로 투영하면 conflicting gradient)
        self.q_itc_proj    = nn.Linear(768, 512)   # query-only path
        self.q_tc_itc_proj = nn.Linear(768, 512)   # text-conditioned path

        # ITM head: text-conditioned query → match/no-match
        self.itm_head = ITMHead(768)

        # learnable temperature
        self.log_temp = nn.Parameter(torch.tensor(math.log(1.0 / temp_init)))

        # ── Momentum Q-Former (ALBEF EMA 방식) ─────────────────────────────
        # requires_grad=False → DDP gradient sync 없음, EMA 업데이트만
        self.qformer_m     = copy.deepcopy(qformer)
        self.q_itc_proj_m  = copy.deepcopy(self.q_itc_proj)
        for p in self.qformer_m.parameters():
            p.requires_grad = False
        for p in self.q_itc_proj_m.parameters():
            p.requires_grad = False

    # ── Momentum EMA 업데이트 ───────────────────────────────────────────────
    @torch.no_grad()
    def _momentum_update(self):
        """θ_m ← m·θ_m + (1-m)·θ  (학습 step마다 호출)"""
        for p, pm in zip(self.qformer.parameters(),
                         self.qformer_m.parameters()):
            pm.data.mul_(_MOMENTUM).add_(p.data, alpha=1.0 - _MOMENTUM)
        for p, pm in zip(self.q_itc_proj.parameters(),
                         self.q_itc_proj_m.parameters()):
            pm.data.mul_(_MOMENTUM).add_(p.data, alpha=1.0 - _MOMENTUM)

    # ── forward ────────────────────────────────────────────────────────────
    def forward(
        self,
        image_feats,           # (B, 2049, 768)  사전 추출된 M3D-CLIP 패치 시퀀스
        text_input_ids,        # (B, L)
        text_attn_mask,        # (B, L)
        itm_image_feats=None,  # (B, 2049, 768)  ITM 전용 (mixup 미적용 원본)
        synth_neg_ids=None,    # (B, L)  합성 오염 텍스트 token IDs (ITM neg)
        synth_neg_mask=None,   # (B, L)  합성 오염 텍스트 attention mask
        compute_itm=True,      # False 시 ITM 계산 스킵 (Stage 1 ITC-only 학습용)
    ):
        # 1) 사전 추출 feature 그대로 사용 (image_encoder forward 불필요)
        image_tokens = image_feats.to(next(self.qformer.parameters()).device)

        # 2) frozen text CLS (ITC용)
        txt_cls = self.text_encoder.encode_cls(
            text_input_ids, text_attn_mask)                # (B, 768)

        # 3) ITC: CrossAttentionFusion → 512d projection
        txt_seq = txt_cls.unsqueeze(1)                     # (B, 1, 768)
        img_cls_f, txt_cls_f = self.itc_fusion(image_tokens, txt_seq)
        img_emb = self.itc_proj(img_cls_f)                 # (B, 512)
        txt_emb = self.itc_proj(txt_cls_f)                 # (B, 512)
        loss_itc = info_nce_loss(img_emb, txt_emb, self.log_temp)

        # 4) Q-Former text-conditioning용 토큰 임베딩
        txt_tok_embs = self.text_encoder.get_token_embeddings(
            text_input_ids)                                # (B, L, 768)

        # 5) Momentum soft targets (ALBEF) + Q-Former query-only ITC
        with torch.no_grad():
            if self.training:
                self._momentum_update()
            queries_m, _ = self.qformer_m(image_tokens)
            q_emb_m   = F.normalize(
                self.q_itc_proj_m(queries_m.mean(1)).float(), dim=-1)   # (B, 512)
            txt_n_s   = F.normalize(txt_emb.float().detach(), dim=-1)
            temp_val  = self.log_temp.exp().clamp(max=100)
            # momentum 모델 유사도 분포 (soft)
            soft_q2t  = (q_emb_m @ txt_n_s.T * temp_val).softmax(dim=-1)  # (B, B)
            soft_t2q  = (txt_n_s @ q_emb_m.T * temp_val).softmax(dim=-1)  # (B, B)
            # hard one-hot label
            B_local   = image_tokens.size(0)
            one_hot   = torch.zeros(B_local, B_local,
                                    device=image_tokens.device, dtype=soft_q2t.dtype)
            one_hot.fill_diagonal_(1.0)
            target_q2t = self._ALPHA_SOFT * soft_q2t + (1.0 - self._ALPHA_SOFT) * one_hot
            target_t2q = self._ALPHA_SOFT * soft_t2q + (1.0 - self._ALPHA_SOFT) * one_hot

        queries_qf, _ = self.qformer(image_tokens)               # (B, Nq, 768)
        q_itc_emb     = self.q_itc_proj(queries_qf.mean(dim=1))  # (B, 512)
        loss_q_itc    = soft_info_nce_loss(
            q_itc_emb, txt_emb.detach(), self.log_temp, target_q2t, target_t2q)

        # 6) Text-conditioned Q-ITC: 전용 projection head 사용 (query-only head와 분리)
        queries_tc, _ = self.qformer(image_tokens, text_tokens=txt_tok_embs)  # (B, Nq, 768)
        q_tc_emb      = self.q_tc_itc_proj(queries_tc.mean(dim=1))            # (B, 512)
        loss_q_tc_itc = info_nce_loss(q_tc_emb, txt_emb.detach(), self.log_temp)

        # 7) ITM: Random negative sampling (compute_itm=True 일 때만 실행)
        if compute_itm:
            #    의료 CT 리포트는 내용이 유사 → hard negative mining은 암기 유발
            #    → random negative로 "이미지-텍스트가 대략 맞는가"를 학습
            #    → 일반화 가능 → val_itm 0.4~0.5 달성 가능
            #    itm_image_feats: mixup 미적용 원본 사용 (mixup 이미지로 pos label 주면 ambiguous)
            itm_img_tokens = (itm_image_feats.to(image_tokens.device)
                              if itm_image_feats is not None else image_tokens)
            B = itm_img_tokens.size(0)
            with torch.no_grad():
                if dist.is_available() and dist.is_initialized():
                    ws   = dist.get_world_size()
                    rank = dist.get_rank()
                    all_ids_l   = [torch.zeros_like(text_input_ids) for _ in range(ws)]
                    all_masks_l = [torch.zeros_like(text_attn_mask) for _ in range(ws)]
                    dist.all_gather(all_ids_l,   text_input_ids.contiguous())
                    dist.all_gather(all_masks_l, text_attn_mask.contiguous())
                    global_ids   = torch.cat(all_ids_l,   dim=0)   # (B*W, L)
                    global_masks = torch.cat(all_masks_l, dim=0)   # (B*W, L)
                else:
                    global_ids   = text_input_ids
                    global_masks = text_attn_mask
                    rank = 0

                G = global_ids.size(0)
                dev = image_tokens.device

                # 랜덤 negative text index (자기 자신 제외)
                neg_txt_global_idx = torch.randint(0, G, (B,), device=dev)
                self_idx = torch.arange(B, device=dev) + rank * B
                collision = (neg_txt_global_idx == self_idx)
                neg_txt_global_idx[collision] = (neg_txt_global_idx[collision] + 1) % G

                # ITC 유사도 기반 hard negative image
                # 가장 비슷하게 생긴 다른 환자 이미지 → Q-Former가 구별하기 어려운 hard case
                img_emb_n = F.normalize(img_emb.float().detach(), dim=-1)      # (B, 512) float32
                sim_i2i = (img_emb_n @ img_emb_n.T).float()                  # (B, B)  float32 강제
                sim_i2i.fill_diagonal_(-1e9)
                hard_neg_img_idx = sim_i2i.argmax(dim=1)                     # (B,)

            # Q-Former에 ITM gradient 허용 → matching 학습 가능
            # ITM path 전용 token position dropout: pair 암기 차단
            # q_pos는 queries_tc와 별도로 계산 (dropout 적용, q_tc_itc와 분리)
            txt_tok_embs_dropped = self._token_dropout(txt_tok_embs, text_attn_mask, drop_prob=0.20)

            if synth_neg_ids is not None:
                # 합성 오염 텍스트: 소견 존재/부재를 뒤집은 명백히 잘못된 텍스트
                # _corrupt_report()가 매 호출마다 다른 텍스트 생성 (stochastic) →
                # Q-Former가 특정 corrupted text identity를 암기할 수 없음
                neg_txt_tok  = self.text_encoder.get_token_embeddings(synth_neg_ids)
                neg_txt_mask = synth_neg_mask
            else:
                # Fallback: random other-patient text
                neg_txt_tok  = self.text_encoder.get_token_embeddings(global_ids[neg_txt_global_idx])
                neg_txt_mask = global_masks[neg_txt_global_idx]
            neg_txt_tok_dropped = self._token_dropout(neg_txt_tok, neg_txt_mask, drop_prob=0.20)

            # 이미지 토큰도 dropout: 고정 피처 암기 방지 (텍스트와 동일 원리)
            itm_img_do      = self._image_token_dropout(itm_img_tokens,                       drop_prob=0.30)
            itm_neg_img_do  = self._image_token_dropout(itm_img_tokens[hard_neg_img_idx],    drop_prob=0.30)

            q_pos,     _ = self.qformer(itm_img_do,     text_tokens=txt_tok_embs_dropped)
            q_neg_txt, _ = self.qformer(itm_img_do,     text_tokens=neg_txt_tok_dropped)
            q_neg_img, _ = self.qformer(itm_neg_img_do, text_tokens=txt_tok_embs_dropped)

            all_q      = torch.cat([q_pos, q_neg_txt, q_neg_img], dim=0)  # (3B, Nq, 768)
            logits_itm = self.itm_head(all_q).mean(dim=1)                  # (3B, 2)
            labels_itm = torch.cat([
                torch.ones (B, dtype=torch.long, device=image_tokens.device),
                torch.zeros(B, dtype=torch.long, device=image_tokens.device),
                torch.zeros(B, dtype=torch.long, device=image_tokens.device),
            ])
            loss_itm = F.cross_entropy(logits_itm, labels_itm, label_smoothing=0.10)
        else:
            loss_itm = torch.tensor(0.0, device=image_tokens.device)

        return {
            "itc":      loss_itc,
            "q_itc":    loss_q_itc,
            "q_tc_itc": loss_q_tc_itc,
            "itm":      loss_itm,
        }

    # ── ITM 전용 token position dropout ─────────────────────────────────
    @staticmethod
    def _token_dropout(token_emb, attn_mask, drop_prob=0.45):
        """ITM path 전용 text token dropout (padding 보존).
        학습/검증 모두 적용하여 train/val 분포 일관성 유지."""
        B, L, D = token_emb.shape
        drop = torch.rand(B, L, device=token_emb.device) < drop_prob
        drop = drop & attn_mask.bool()          # padding은 dropout 안 함
        dropped = token_emb.clone()
        dropped[drop] = 0.0
        return dropped

    @staticmethod
    def _image_token_dropout(img_tokens, drop_prob=0.30):
        """ITM path 전용 image patch token dropout.
        사전 추출된 고정 피처라 매 epoch 동일 → 암기 방지를 위해 무작위 패치 드롭.
        학습/검증 모두 적용하여 train/val 분포 일관성 유지."""
        B, L, D = img_tokens.shape
        drop = torch.rand(B, L, device=img_tokens.device) < drop_prob
        dropped = img_tokens.clone()
        dropped[drop] = 0.0
        return dropped

    # ── Stage 2 LM prefix 추출 (inference / Stage 2 학습) ─────────────────
    @torch.no_grad()
    def get_lm_prefix(self, images):
        """이미지 → Q-Former query 출력 (B, Nq, 768) — BioGPT prefix로 사용."""
        image_tokens = self.image_encoder(images)
        queries, _   = self.qformer(image_tokens)
        return self.qformer.get_lm_prefix(queries)


# ── 모델 빌더 ────────────────────────────────────────────────────────────────

def build_model(
    ckpt_path:   str   = None,
    num_queries: int   = 32,
    num_layers:  int   = 6,
    dropout:     float = 0.1,
) -> tuple:
    """
    Returns:
        model     : BLIP2CTModel
        tokenizer : Bio_ClinicalBERT tokenizer (ITC 텍스트 토크나이징용)
    """
    print("[build_model] BLIP 이미지 인코더 + BERT 텍스트 인코더 로드")
    img_enc = BLIPImageEncoder()
    txt_enc = BERTTextEncoder()
    tokenizer = txt_enc.tokenizer

    qformer     = QFormer(
        num_queries=num_queries,
        hidden_dim=768,
        num_heads=12,
        num_layers=num_layers,
        ffn_dim=3072,
        dropout=dropout,
    )

    model = BLIP2CTModel(img_enc, txt_enc, qformer)

    # best.pt에서 fusion+proj 가중치 로드 (이미 학습된 ITC 가중치)
    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if "fusion" in ckpt:
            model.itc_fusion.load_state_dict(ckpt["fusion"])
            print("  ✓ itc_fusion best.pt 로드")
        if "proj" in ckpt:
            model.itc_proj.load_state_dict(ckpt["proj"])
            print("  ✓ itc_proj best.pt 로드")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  학습 가능 파라미터: {trainable/1e6:.1f}M / 전체 {total/1e6:.1f}M")

    return model, tokenizer


# ── 동작 확인 ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from pathlib import Path as _Path
    _ROOT  = _Path(__file__).parent.parent
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, tokenizer = build_model(
        ckpt_path=str(_ROOT / "output/m3d_finetuned_cls/best.pt"),
    )
    model = model.to(device)

    # 더미 입력
    B = 2
    images = torch.zeros(B, 1, 32, 256, 256, device=device)
    enc    = tokenizer(
        ["Findings: No acute findings."] * B,
        max_length=64, truncation=True, padding="max_length",
        return_tensors="pt",
    )
    ids    = enc["input_ids"].to(device)
    masks  = enc["attention_mask"].to(device)

    model.train()
    out = model(images, ids, masks)
    print(f"loss_itc : {out['itc'].item():.4f}")
    print(f"loss_itm : {out['itm'].item():.4f}")
