import math
import re
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── 공유 채팅 템플릿 ───────────────────────────────────────────────────────────
_SYSTEM_PROMPT = (
    "You are an expert OCT image analysis assistant. "
    "Describe the retinal scan findings for each layer concisely."
)
_USER_PROMPT = "Analyze the OCT scan and provide layer-by-layer findings."


def _format_chat(text: str, lm_tok) -> str:
    messages = [
        {"role": "system",    "content": _SYSTEM_PROMPT},
        {"role": "user",      "content": _USER_PROMPT},
        {"role": "assistant", "content": text},
    ]
    return lm_tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


# ── desc 재정렬: "diagnosed disease: X"를 맨 뒤로 이동 ───────────────────────
# teacher forcing 시 disease label이 causal context에 먼저 등장하면 LM이 비전을
# 무시하고 텍스트 패턴으로 레이어 설명을 예측하는 shortcut을 차단.
_DX_PREFIX_RE = re.compile(r'^(diagnosed\s+disease\s*[:\-]\s*[^,;\n]+[,;\n]?\s*)', re.I)

def _reorder_desc(text: str) -> str:
    text = text.strip()
    m = _DX_PREFIX_RE.match(text)
    if not m:
        return text
    dx   = m.group(1).strip().rstrip(',;')
    rest = text[m.end():].strip()
    return f"{rest}\n{dx}" if rest else dx


# ── LM 입력 span dropout ──────────────────────────────────────────────────────
def lm_span_dropout(emb, p: float, span: int = 3, seed: int = None):
    """LM 입력 임베딩 (B,T,D)에서 길이 span의 연속 구간 여러 개를 0으로 마스킹.
    단일 토큰 드롭과 달리 findings 구절 전체를 가려, 비전 없이는 못 채우게 강제
    (균일 드롭은 남은 인접 토큰으로 LLM이 보간 가능).

    seed != None → 결정론적 마스크. 검증 시 같은 위치를 가려 val/lm을 epoch 간 비교
    가능한 '비전 의존' 지표로 만든다 (텍스트 일부가 가려져야 비전 기여가 loss에 반영됨)."""
    if p <= 0:
        return emb
    B, T = emb.shape[:2]
    n_spans = max(1, round(p * T / span))
    g = torch.Generator(device=emb.device).manual_seed(seed) if seed is not None else None
    keep = torch.ones(B, T, device=emb.device, dtype=emb.dtype)
    starts = torch.randint(0, T, (B, n_spans), device=emb.device, generator=g)
    for k in range(span):
        keep.scatter_(1, (starts + k).clamp(max=T - 1), 0.0)
    return emb * keep.unsqueeze(-1)


def _chat_prefix_len(lm_tok) -> int:
    prefix = lm_tok.apply_chat_template(
        [{"role": "system", "content": _SYSTEM_PROMPT},
         {"role": "user",   "content": _USER_PROMPT}],
        tokenize=False, add_generation_prompt=True,
    )
    return len(lm_tok(prefix, add_special_tokens=False).input_ids)


# ── SCF Block ─────────────────────────────────────────────────────────────────
class _SCFBlock(nn.Module):
    def __init__(self, embed_dim, num_heads=8, ffn_ratio=4, dropout=0.1):
        super().__init__()
        self.txt_self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.txt_self_norm = nn.LayerNorm(embed_dim)
        self.cross_attn    = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.txt_norm1     = nn.LayerNorm(embed_dim)
        self.img_norm      = nn.LayerNorm(embed_dim)
        ffn_dim = embed_dim * ffn_ratio
        self.txt_ffn  = nn.Sequential(nn.Linear(embed_dim, ffn_dim), nn.GELU(), nn.Dropout(dropout),
                                      nn.Linear(ffn_dim, embed_dim), nn.Dropout(dropout))
        self.txt_norm2 = nn.LayerNorm(embed_dim)

    def forward(self, img_seq, txt_seq):
        # txt SA
        txt_s, _ = self.txt_self_attn(self.txt_self_norm(txt_seq), self.txt_self_norm(txt_seq), self.txt_self_norm(txt_seq))
        txt_seq = txt_seq + txt_s
        # txt CA ← img (image는 K,V로만 사용)
        txt_out, _ = self.cross_attn(self.txt_norm1(txt_seq), self.img_norm(img_seq), self.img_norm(img_seq))
        txt_seq = txt_seq + txt_out
        # txt FFN
        txt_seq = txt_seq + self.txt_ffn(self.txt_norm2(txt_seq))
        return img_seq, txt_seq


# ── ITMFusion ──────────────────────────────────────────────────────
class ITMFusion(nn.Module):
    """이미지-텍스트 상호 attention (ITC global loss용). SCFBlock × num_layers."""
    def __init__(self, embed_dim=768, num_heads=8, num_layers=3, ffn_ratio=4, dropout=0.1):
        super().__init__()
        self.blocks = nn.ModuleList([
            _SCFBlock(embed_dim, num_heads, ffn_ratio, dropout) for _ in range(num_layers)
        ])

    def forward(self, img_seq, txt_seq):
        for block in self.blocks:
            img_seq, txt_seq = block(img_seq, txt_seq)
        return img_seq[:, 0], txt_seq[:, 0]   # CLS token


# ── GLoRIA Local Attention ────────────────────────────────────────────────────
class GLoRIALocalAttn(nn.Module):
    """GLoRIA (Huang et al., 2021) 방식의 text-guided local attention."""
    def __init__(self, hidden_dim: int = 768, temp_attn: float = 4.0, temp_agg: float = 4.0):
        super().__init__()
        self.img_proj  = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.txt_proj  = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.temp_attn = temp_attn
        self.temp_agg  = temp_agg


def gloria_local_loss(
    img_tokens: torch.Tensor,      # (B, N, d)
    layer_tok_embs: torch.Tensor,  # (B, K, L, d)
    layer_masks: torch.Tensor,     # (B, K, L)
    local_attn: GLoRIALocalAttn,
    log_temp: torch.Tensor,
) -> torch.Tensor:
    """GLoRIA GitHub (Huang et al., 2021)와 동일한 two-step attention loss.
    Step1: softmax over words / Step2: temp-scaled softmax over patches."""
    B, N, d = img_tokens.shape
    K, L    = layer_tok_embs.shape[1], layer_tok_embs.shape[2]
    temp    = log_temp.exp().clamp(max=100)
    total   = torch.tensor(0.0, device=img_tokens.device)

    img_f = local_attn.img_proj(img_tokens.float())    # (B, N, d)

    for k in range(K):
        txt_k = layer_tok_embs[:, k].float()           # (B, L, d)
        msk_k = layer_masks[:, k]                      # (B, L)
        txt_f = local_attn.txt_proj(txt_k)             # (B, L, d)

        raw  = torch.einsum('jld,ind->jinl', txt_f, img_f)   # (B_j, B_i, N, L)
        attn = F.softmax(raw, dim=-1)
        attn = F.softmax(attn.permute(0, 1, 3, 2) * local_attn.temp_attn, dim=-1)  # (B_j,B_i,L,N)

        ctx      = torch.einsum('jiln,ind->jild', attn, img_f)              # (B_j,B_i,L,d)
        txt_e    = txt_f.unsqueeze(1).expand(-1, B, -1, -1)
        word_sim = F.cosine_similarity(txt_e, ctx, dim=-1)                   # (B_j,B_i,L)
        msk_e    = msk_k.unsqueeze(1).expand(-1, B, -1)
        word_sim = word_sim.masked_fill(msk_e == 0, -1e9)

        S      = torch.logsumexp(word_sim * local_attn.temp_agg, dim=-1).T  # (B_i,B_j)
        labels = torch.arange(B, device=img_tokens.device)
        total += (F.cross_entropy(S * temp, labels) +
                  F.cross_entropy(S.T * temp, labels)) / 2

    return total / K


# ── 헬퍼: ITC 모듈 일괄 생성 ──────────────────────────────────────────────────
def build_itc_modules(embed_dim: int = 768, proj_dim: int = 512):
    """모든 GLoRIA 학습 파일에서 동일하게 사용하는 ITC 컴포넌트를 반환.

    Returns:
        itm_fusion  : ITMFusion(embed_dim)
        itc_proj    : ProjectionHead(embed_dim → proj_dim)
        local_attn  : GLoRIALocalAttn(embed_dim)
        log_temp    : nn.Parameter (초기값 log(1/0.07))
    """
    from model import ProjectionHead   # model.py에 정의된 공유 헤드

    itm_fusion = ITMFusion(embed_dim)
    itc_proj   = ProjectionHead(embed_dim, out_dim=proj_dim)
    local_attn = GLoRIALocalAttn(hidden_dim=embed_dim)
    log_temp   = nn.Parameter(torch.tensor([math.log(1.0 / 0.07)]))
    return itm_fusion, itc_proj, local_attn, log_temp
