"""
OCT MLP Projector — ITC + DenseMatch + LM (Q-Former 제거, LLaVA-1.5 스타일)

실행 명령어:
  python OCT_LLM_Train_GLoRIA_MLP.py          # Stage 1 처음부터 (자동)
  STAGE=1 python OCT_LLM_Train_GLoRIA_MLP.py  # Stage 1 이어서
  python OCT_LLM_Train_GLoRIA_MLP.py          # Stage 2 자동 (stage1/best.pt 있으면)
  STAGE=2 python OCT_LLM_Train_GLoRIA_MLP.py  # Stage 2 이어서 (stage2/best.pt 있으면)

구조:
  BLIPImageEncoder → MLP Projector → LLM (patch token prefix)
  ITM 제거, LayerQueryRouter(Dense Matching) 추가
  Q-Former 제거 → 이미지 패치 토큰 전체를 MLP로 LLM 공간에 직접 투영

OCT desc 5개 레이어:
  1. Nerve Fiber Layer
  2. Ganglion Cell Layer
  3. Inner Plexiform Layer
  4. Outer Plexiform Layer
  5. Photoreceptor IS/OS
"""

import os
import re
import sys
import warnings
import random

warnings.filterwarnings("ignore")

os.environ.setdefault("NCCL_P2P_DISABLE", "1")
os.environ.setdefault("NCCL_IB_DISABLE", "1")
os.environ.setdefault("NCCL_DEBUG", "WARN")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model import (
    ProjectionHead,
    BERTTextEncoder, BLIPImageEncoder, info_nce_loss,
)
from gloria_itc import (
    ITMFusion, GLoRIALocalAttn, gloria_local_loss, build_itc_modules,
    _format_chat, _chat_prefix_len, lm_span_dropout,
)

# ── LM Config ─────────────────────────────────────────────────────────────────
_LM_MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
_LM_HIDDEN   = 3072
_LM_DTYPE    = torch.bfloat16
_LM_EMBED_FN  = lambda m: m.model.embed_tokens
_LM_TRUNK_FN  = lambda m: m.model
_LM_HEAD_FN   = lambda m: m.lm_head


# ── Train Config ──────────────────────────────────────────────────────────────
class Config:
    LM_MODEL    = "llama3.2_3b"
    BERT_ID = "emilyalsentzer/Bio_ClinicalBERT"
    _BASE_OUT   = "/data2/OCT_output/llama3.2_3b_GLoRIA_MLP"
    STAGE1_CKPT = f"{_BASE_OUT}/stage1/best.pt"
    STAGE2_CKPT = f"{_BASE_OUT}/stage2/best.pt"
    # STAGE env로 강제 지정 가능. 없으면 stage1/best.pt 존재 여부로 자동 결정.
    STAGE       = int(os.environ["STAGE"]) if "STAGE" in os.environ else (
                  2 if os.path.exists(f"{_BASE_OUT}/stage1/best.pt") else 1)
    OUT_DIR     = f"{_BASE_OUT}/stage{STAGE}"
    TRAIN_PATH  = str(_ROOT / "dataset/train")
    TEST_PATH   = str(_ROOT / "dataset/test")

    EPOCHS        = 30
    BATCH_SIZE    = 8 if STAGE == 1 else 4
    NUM_WORKERS   = 0
    LR_LLM        = 2e-5
    LR            = 1e-4
    LR_PRETRAINED = 2e-5
    WEIGHT_DECAY  = 0.0
    WARMUP_RATIO  = 0.03
    GRAD_CLIP     = 1.0
    MAX_ITC_LEN   = 512
    MAX_LAYER_LEN = 64
    MAX_LM_LEN    = 512 
    EARLY_STOP    = 3
    SEED          = 42

    ITC_WEIGHT   = 1.0
    LM_WEIGHT    = 1.0
    DENSE_WEIGHT = 0.5
    BLIND_WEIGHT = 1.0
    BLIND_MARGIN = 0.3
    LM_TOKEN_DROPOUT = 0.15  # 0.4→0.15: 과한 dropout은 infilling 학습→생성 붕괴. norm fix 후 비전 사용은 blind contrastive가 담당

    NUM_OCT_LAYERS = 5

    VAL_RATIO  = 0.1


# ── OCT 레이어 파서 ────────────────────────────────────────────────────────────
_LAYER_HEADERS = [
    "Nerve Fiber Layer",
    "Ganglion Cell Layer",
    "Inner Plexiform Layer",
    "Outer Plexiform Layer",
    "Photoreceptor IS/OS",
]

_LAYER_RE = re.compile(
    r'(Nerve Fiber Layer|Ganglion Cell Layer|Inner Plexiform Layer'
    r'|Outer Plexiform Layer|Photoreceptor IS/OS[^:]*)'
    r'\s*:\s*(.*?)(?=(?:Nerve Fiber Layer|Ganglion Cell Layer'
    r'|Inner Plexiform Layer|Outer Plexiform Layer|Photoreceptor IS/OS)|$)',
    re.S | re.I,
)


def split_layer_descs(desc: str) -> List[str]:
    """desc → 5개 레이어 텍스트 리스트. 없는 레이어는 빈 문자열."""
    found = {m.group(1).strip(): m.group(2).strip() for m in _LAYER_RE.finditer(desc)}
    result = []
    for header in _LAYER_HEADERS:
        matched = ""
        for key, val in found.items():
            if header.lower() in key.lower():
                matched = f"{header}: {val}".strip()
                break
        result.append(matched if matched else f"{header}: normal")
    return result


# ── OCT Dataset ────────────────────────────────────────────────────────────────
class OCTDataset(Dataset):
    def __init__(self, hf_dataset):
        self.data = hf_dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        s = self.data[idx]
        image = s['image'].convert('RGB')
        label = (s.get('label') or 'normal').strip()
        return image, (s.get('desc') or '').strip(), label


def build_oct_datasets(train_path, test_path, val_ratio=0.1, seed=42):
    from datasets import Dataset as HFDataset
    train_full = HFDataset.load_from_disk(train_path)
    test_hf    = HFDataset.load_from_disk(test_path)
    split      = train_full.train_test_split(test_size=val_ratio, seed=seed)
    train_ds   = OCTDataset(split['train'])
    val_ds     = OCTDataset(split['test'])
    test_ds    = OCTDataset(test_hf)
    print(f"[Dataset] Train: {len(train_ds):5d} | Val: {len(val_ds):5d} | Test: {len(test_ds):5d}")
    return train_ds, val_ds, test_ds


# ── Visual MLP Projector (LLaVA-1.5 스타일) ──────────────────────────────────
class VisualProjector(nn.Module):
    """각 이미지 패치 토큰을 LLM hidden space로 투영하는 2-layer MLP."""
    def __init__(self, in_dim: int = 768, out_dim: int = 4096):
        super().__init__()
        # LLaVA-1.5 projector: Linear→GELU→Linear (LayerNorm 없음).
        # LLM이 unfrozen(전체 fine-tune)이라 co-adaptation으로 안정 → frozen 모델용
        # zero-start/LayerScale 불필요. LLaVA-1.5 원본 레시피 그대로.
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim), nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )
        nn.init.trunc_normal_(self.net[0].weight, std=0.02)
        nn.init.zeros_(self.net[0].bias)
        nn.init.trunc_normal_(self.net[2].weight, std=0.02)
        nn.init.zeros_(self.net[2].bias)

    def forward(self, x):
        # x: (B, N, in_dim) → (B, N, out_dim)
        return self.net(x)


# ── Combined Model ─────────────────────────────────────────────────────────────


class CombinedModel(nn.Module):
    def __init__(self, image_encoder, text_encoder,
                 itc_proj, log_temp, proj, lm, local_attn):
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder  = text_encoder
        self.itc_proj      = itc_proj
        self.log_temp      = log_temp
        self.proj          = proj
        self.lm            = lm
        self.local_attn    = local_attn
        self.vis_scale     = nn.Parameter(torch.zeros(1))

    def _get_embed(self, ids):
        return _LM_EMBED_FN(self.lm)(ids)

    def forward(self, images, itc_ids, itc_masks, lm_ids, lm_masks,
                layer_ids, layer_masks, lm_prefix_len: int = 0):
        image_tokens = self.image_encoder(images)   # (B, N, 768)
        B, N, device = image_tokens.size(0), image_tokens.size(1), image_tokens.device
        K            = layer_ids.size(1)

        # ── ITC ────────────────────────────────────────────────────────────
        txt_cls  = self.text_encoder.encode_cls(itc_ids, itc_masks)
        img_emb  = self.itc_proj(image_tokens[:, 0])
        txt_emb  = self.itc_proj(txt_cls)
        loss_itc = info_nce_loss(img_emb, txt_emb, self.log_temp)

        # ── GLoRIA Dense Matching ───────────────────────────────────────────
        flat_ids       = layer_ids.reshape(B * K, -1)
        flat_masks_lyr = layer_masks.reshape(B * K, -1)
        layer_tok      = self.text_encoder.get_contextual_token_embeddings(
            flat_ids, flat_masks_lyr,
        )                                                        # (B*K, L, 768)
        L_tok          = layer_tok.size(1)
        layer_tok_embs = layer_tok.reshape(B, K, L_tok, 768)
        layer_masks_4d = layer_masks.reshape(B, K, -1)
        loss_dense = gloria_local_loss(
            image_tokens, layer_tok_embs, layer_masks_4d,
            self.local_attn, self.log_temp,
        )

        # ── LM (MLP patch token prefix) ────────────────────────────────────
        # 이미지 패치 토큰 전체를 MLP로 LLM 공간에 투영 (LLaVA-1.5 스타일)
        # zero-init projector → 초기 prefix≈0, 학습되며 정보화 (rescaling 불필요, LayerNorm 제거됨)
        vis        = self.proj(image_tokens) * F.softplus(self.vis_scale)  # (B, N, lm_dim)
        txt_emb_lm = self._get_embed(lm_ids)
        # span 드롭으로 findings 구절을 가려 비전 의존 강제. val은 고정 seed로 결정론적
        # 적용 → val/lm이 비전 의존 지표가 됨 (dropout OFF면 frozen LLM이 항상 ~0).
        txt_emb_lm = lm_span_dropout(txt_emb_lm, Config.LM_TOKEN_DROPOUT,
                                     seed=None if self.training else 1234)
        vis_mask   = torch.ones(B, N, dtype=lm_masks.dtype, device=device)
        full_mask  = torch.cat([vis_mask, lm_masks], dim=1)
        txt_lbl    = lm_ids.clone()
        txt_lbl[lm_masks == 0] = -100
        if lm_prefix_len > 0:
            txt_lbl[:, :lm_prefix_len] = -100
        labels = torch.cat([
            torch.full((B, N), -100, dtype=torch.long, device=device),
            txt_lbl,
        ], dim=1)

        hidden_lm = _LM_TRUNK_FN(self.lm)(
            inputs_embeds=torch.cat([vis, txt_emb_lm], dim=1),
            attention_mask=full_mask,
        ).last_hidden_state
        logits_lm = _LM_HEAD_FN(self.lm)(hidden_lm)
        # next-token prediction: logits[i] predicts labels[i+1]
        loss_lm   = F.cross_entropy(logits_lm[:, :-1].contiguous().view(-1, logits_lm.size(-1)),
                                    labels[:, 1:].contiguous().view(-1), ignore_index=-100)

        # ── Blind Contrastive ──────────────────────────────────────────────
        with torch.no_grad():
            blind_h       = _LM_TRUNK_FN(self.lm)(
                inputs_embeds=torch.cat([torch.zeros_like(vis), txt_emb_lm], dim=1),
                attention_mask=full_mask,
            ).last_hidden_state
            logits_blind  = _LM_HEAD_FN(self.lm)(blind_h)
            loss_lm_blind = F.cross_entropy(logits_blind[:, :-1].contiguous().view(-1, logits_blind.size(-1)),
                                            labels[:, 1:].contiguous().view(-1), ignore_index=-100)
        loss_blind = F.softplus(loss_lm - loss_lm_blind.detach() + Config.BLIND_MARGIN)

        total = (Config.ITC_WEIGHT   * loss_itc   +
                 Config.LM_WEIGHT    * loss_lm     +
                 Config.DENSE_WEIGHT * loss_dense  +
                 Config.BLIND_WEIGHT * loss_blind)

        return {
            "itc":   loss_itc,
            "lm":    loss_lm,
            "dense": loss_dense,
            "blind": loss_blind,
            "vis_gap": (loss_lm_blind - loss_lm).detach(),  # 비전이 절약한 nats (↑=비전 사용)
            "total": total,
        }

    @torch.no_grad()
    def generate(self, images, prompt_ids=None,
                 max_new_tokens=256, min_new_tokens=0,
                 num_beams=4, do_sample=False,
                 temperature=1.0, top_p=1.0,
                 repetition_penalty=1.2, no_repeat_ngram_size=0):
        with torch.amp.autocast("cuda", dtype=_LM_DTYPE):
            image_feats = self.image_encoder(images)            # (B, N, 768)
            vis         = self.proj(image_feats) * F.softplus(self.vis_scale)
            B           = vis.size(0)
            emb         = (torch.cat([vis, self._get_embed(prompt_ids)], dim=1)
                           if prompt_ids is not None else vis)
            attn_mask   = torch.ones(B, emb.size(1), dtype=torch.long, device=emb.device)
            out = self.lm.generate(
                inputs_embeds=emb, attention_mask=attn_mask,
                pad_token_id=self.lm.config.pad_token_id,
                max_new_tokens=max_new_tokens, max_length=None,
                min_new_tokens=min_new_tokens, num_beams=num_beams,
                do_sample=do_sample, temperature=temperature, top_p=top_p,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
            )
        return out


# ── Model Builder ──────────────────────────────────────────────────────────────
def build_combined_model():
    print("[build] BLIP 이미지 인코더 로드")
    img_enc  = BLIPImageEncoder()
    txt_enc  = BERTTextEncoder()
    bert_tok = txt_enc.tokenizer

    _, itc_proj, local_attn, log_temp = build_itc_modules()

    print(f"[build] LLM 로드: {_LM_MODEL_ID}")
    lm_tok = AutoTokenizer.from_pretrained(_LM_MODEL_ID)
    if lm_tok.pad_token is None:
        lm_tok.pad_token = lm_tok.eos_token

    lm = AutoModelForCausalLM.from_pretrained(
        _LM_MODEL_ID, torch_dtype=_LM_DTYPE, low_cpu_mem_usage=True,
    )
    lm.config.pad_token_id = lm_tok.eos_token_id
    _nb = sum(p.numel() for p in lm.parameters()) / 1e9
    if Config.STAGE == 1:
        for p in lm.parameters():
            p.requires_grad = False
        print(f"  [Stage 1] LLM Frozen — connector 정렬만 학습 ({_nb:.1f}B params)")
    else:
        print(f"  [Stage 2] LLM Unfreeze — joint fine-tune ({_nb:.1f}B params)")

    model = CombinedModel(
        image_encoder=img_enc, text_encoder=txt_enc,
        itc_proj=itc_proj, log_temp=log_temp,
        proj=VisualProjector(768, _LM_HIDDEN), lm=lm, local_attn=local_attn,
    )
    model.to(_LM_DTYPE)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  학습 파라미터: {trainable/1e6:.1f}M")
    return model, bert_tok, lm_tok


def _fit_height_crop(img):
    from PIL import Image
    w, h = img.size
    new_w = round(w * 384 / h)
    img = img.resize((new_w, 384), Image.BICUBIC)
    x = (new_w - 384) // 2
    return img.crop((x, 0, x + 384, 384))


# ── Collate ────────────────────────────────────────────────────────────────────
class DenseMatchCollate:
    def __init__(self, bert_tok, lm_tok):
        from transformers import AutoProcessor
        self.bert_tok      = bert_tok
        self.lm_tok        = lm_tok
        self.img_processor = AutoProcessor.from_pretrained("QIAIUNCC/LO-VLM")

    def __call__(self, batch):
        images, texts, labels = zip(*batch)
        pixel_values = self.img_processor(
            images=[_fit_height_crop(img) for img in images], return_tensors="pt"
        )["pixel_values"]

        itc_enc = self.bert_tok(list(texts), max_length=Config.MAX_ITC_LEN,
                               truncation=True, padding="max_length", return_tensors="pt")
        # desc에 진단명이 없어서(label 컬럼에만 존재) LM 타겟 맨 앞에 직접 주입 (공식 LO-VLM과 동일한 순서)
        lm_texts = [f"diagnosed disease: {lb}\n{t}" for t, lb in zip(texts, labels)]
        lm_enc  = self.lm_tok([_format_chat(lt, self.lm_tok) for lt in lm_texts],
                              max_length=Config.MAX_LM_LEN,
                              truncation=True, padding="max_length", return_tensors="pt")

        all_layer_ids, all_layer_masks = [], []
        for text in texts:
            layer_texts = split_layer_descs(text)
            enc = self.bert_tok(layer_texts,
                               max_length=Config.MAX_LAYER_LEN,
                               truncation=True, padding="max_length",
                               return_tensors="pt")
            all_layer_ids.append(enc.input_ids)
            all_layer_masks.append(enc.attention_mask)

        layer_ids   = torch.stack(all_layer_ids)
        layer_masks = torch.stack(all_layer_masks)

        return (pixel_values,
                itc_enc.input_ids, itc_enc.attention_mask,
                lm_enc.input_ids,  lm_enc.attention_mask,
                layer_ids,         layer_masks)


# ── Lightning DataModule ───────────────────────────────────────────────────────
class OCTDataModule(L.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.train_ds = self.val_ds = self.test_ds = None
        self.bert_tok = None
        self.lm_tok   = None

    def setup(self, stage=None):
        if self.bert_tok is None:
            from transformers import BertTokenizer
            self.bert_tok = BertTokenizer.from_pretrained(
                Config.BERT_ID, model_max_length=Config.MAX_ITC_LEN,
            )
            lm_tok = AutoTokenizer.from_pretrained(_LM_MODEL_ID)
            if lm_tok.pad_token is None:
                lm_tok.pad_token = lm_tok.eos_token
            self.lm_tok = lm_tok
        if self.train_ds is None:
            self.train_ds, self.val_ds, self.test_ds = build_oct_datasets(
                Config.TRAIN_PATH, Config.TEST_PATH,
                val_ratio=Config.VAL_RATIO, seed=Config.SEED,
            )

    def _make_loader(self, dataset, shuffle: bool) -> DataLoader:
        return DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=shuffle,
                         num_workers=Config.NUM_WORKERS, pin_memory=True,
                         drop_last=shuffle,
                         collate_fn=DenseMatchCollate(self.bert_tok, self.lm_tok))

    def train_dataloader(self):  return self._make_loader(self.train_ds, shuffle=True)
    def val_dataloader(self):    return self._make_loader(self.val_ds,   shuffle=False)
    def test_dataloader(self):   return self._make_loader(self.test_ds,  shuffle=False)


# ── Lightning Module ───────────────────────────────────────────────────────────
class OCTLightningModule(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model        = None
        self.bert_tok     = None
        self.lm_tok       = None
        self._best_val_lm = float("inf")
        self._best_epoch  = 0

    def setup(self, stage: str):
        if self.model is not None:
            return
        self.model, self.bert_tok, self.lm_tok = build_combined_model()
        self.lm_prefix_len = _chat_prefix_len(self.lm_tok)
        if Config.STAGE == 2:
            if os.path.exists(Config.STAGE2_CKPT):
                self._resume_stage2()
            else:
                self._load_stage1_connector()
        elif Config.STAGE == 1 and os.path.exists(Config.STAGE1_CKPT):
            self._resume_stage1()

    def _load_connector(self, ck):
        self.model.itc_proj.load_state_dict(ck["itc_proj"])
        self.model.proj.load_state_dict(ck["proj"])
        self.model.local_attn.load_state_dict(ck["local_attn"])
        self.model.log_temp.data.copy_(ck["log_temp"].reshape_as(self.model.log_temp))
        self.model.vis_scale.data.copy_(ck["vis_scale"].reshape_as(self.model.vis_scale))

    def _resume_stage1(self):
        ck = torch.load(Config.STAGE1_CKPT, map_location="cpu", weights_only=False)
        self._load_connector(ck)
        print(f"  [Stage 1 resume] connector 로드 완료: {Config.STAGE1_CKPT}")

    def _resume_stage2(self):
        ck = torch.load(Config.STAGE2_CKPT, map_location="cpu", weights_only=False)
        self._load_connector(ck)
        if "lm" in ck:
            self.model.lm.load_state_dict(ck["lm"])
        print(f"  [Stage 2 resume] 전체 모델 로드 완료: {Config.STAGE2_CKPT}")

    def _load_stage1_connector(self):
        if not os.path.exists(Config.STAGE1_CKPT):
            raise FileNotFoundError(
                f"[Stage 2] Stage-1 connector 없음: {Config.STAGE1_CKPT}\n"
                f"  먼저 'STAGE=1 python {os.path.basename(__file__)}' 로 Stage 1을 완료하세요.")
        ck = torch.load(Config.STAGE1_CKPT, map_location="cpu", weights_only=False)
        self._load_connector(ck)
        print(f"  [Stage 2] Stage-1 connector 로드 완료: {Config.STAGE1_CKPT}")

    def _forward_batch(self, batch):
        (pixel_values, itc_ids, itc_masks, lm_ids, lm_masks,
         layer_ids, layer_masks) = batch
        return self.model(pixel_values, itc_ids, itc_masks, lm_ids, lm_masks,
                         layer_ids, layer_masks,
                         lm_prefix_len=self.lm_prefix_len)

    def on_before_optimizer_step(self, optimizer):
        # 비유한 gradient 방어: 첫 스텝의 Inf/NaN grad가 파라미터를 NaN으로 오염시키는 것 차단.
        # (step0 정상 → step1 NaN = gradient 폭발이 원인. grad를 유한값으로 강제해 오염 차단.)
        n_bad = 0
        for group in optimizer.param_groups:
            for p in group["params"]:
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    torch.nan_to_num_(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
                    n_bad += 1
        if n_bad:
            self.log("train/nonfinite_grads", float(n_bad), on_step=True, on_epoch=False, sync_dist=True)

    def training_step(self, batch, batch_idx):
        out = self._forward_batch(batch)
        self.log("train/total", out["total"], on_step=True, on_epoch=True,
                 sync_dist=True, prog_bar=True)
        for k in ("itc", "lm", "dense", "blind"):
            self.log(f"train/{k}", out[k], on_step=False, on_epoch=True, sync_dist=True)
        return out["total"]

    def validation_step(self, batch, batch_idx):
        out = self._forward_batch(batch)
        for k in ("total", "itc", "lm", "dense", "blind", "vis_gap"):
            self.log(f"val/{k}", out[k], on_epoch=True, sync_dist=True,
                    prog_bar=(k in ("total", "lm")))

    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking:
            return
        val_lm = self.trainer.callback_metrics.get("val/lm", float("inf"))
        if isinstance(val_lm, torch.Tensor):
            val_lm = val_lm.item()

        if val_lm < self._best_val_lm:
            self._best_val_lm = val_lm
            self._best_epoch  = self.current_epoch
            self._save_best(val_lm)

    def _save_best(self, val_lm):
        if not self.trainer.is_global_zero:
            return
        sd = self.model.state_dict()

        def _ext(sub):
            return {k[len(sub)+1:]: v for k, v in sd.items() if k.startswith(sub + ".")}

        ckpt = {
            "itc_proj":   _ext("itc_proj"),
            "proj":       _ext("proj"),
            "local_attn": _ext("local_attn"),
            "log_temp":   sd["log_temp"],
            "vis_scale":  sd["vis_scale"],
            "val_lm":     val_lm,
            "epoch":      self.current_epoch,
            "stage":      Config.STAGE,
        }
        if Config.STAGE == 2:
            ckpt["lm"] = _ext("lm")
        os.makedirs(Config.OUT_DIR, exist_ok=True)
        torch.save(ckpt, os.path.join(Config.OUT_DIR, "best.pt"))
        print(f"  [best.pt] val_lm={val_lm:.4f} epoch={self.current_epoch}", flush=True)

    def configure_optimizers(self):
        core         = self.model
        total_steps  = self.trainer.estimated_stepping_batches
        warmup_steps = int(total_steps * Config.WARMUP_RATIO)

        itc_params = [p for grp in [
            core.itc_proj.parameters(),
            [core.log_temp],
        ] for p in grp if p.requires_grad]

        mlp_params = [p for grp in [
            core.proj.parameters(),
            core.local_attn.parameters(),
            [core.vis_scale],
        ] for p in grp if p.requires_grad]

        lm_params = [p for p in core.lm.parameters() if p.requires_grad]

        groups = [
            {"params": itc_params,  "lr": Config.LR_PRETRAINED, "name": "itc"},
            {"params": mlp_params,  "lr": Config.LR,            "name": "mlp"},
        ]
        if lm_params:   # Stage 2에서만 LLM 학습 (Stage 1은 frozen → lm_params 비어있음)
            groups.append({"params": lm_params, "lr": Config.LR_LLM, "name": "llm"})
        optimizer = torch.optim.AdamW(groups, weight_decay=Config.WEIGHT_DECAY)

        from transformers import get_cosine_schedule_with_warmup
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}


# ── 학습 ──────────────────────────────────────────────────────────────────────
def train():
    random.seed(Config.SEED); np.random.seed(Config.SEED)
    torch.manual_seed(Config.SEED); torch.cuda.manual_seed_all(Config.SEED)

    os.makedirs(Config.OUT_DIR, exist_ok=True)

    callbacks = [
        EarlyStopping(monitor="val/lm", patience=Config.EARLY_STOP, mode="min", verbose=True),
    ]

    trainer = L.Trainer(
        max_epochs=Config.EPOCHS, devices=4, accelerator="gpu",
        strategy="ddp",
        precision="bf16-mixed",
        callbacks=callbacks,
        log_every_n_steps=1, enable_progress_bar=True,
        default_root_dir=Config.OUT_DIR, deterministic=False,
        enable_checkpointing=False,
    )

    pl_module   = OCTLightningModule()
    data_module = OCTDataModule()

    trainer.fit(pl_module, data_module)


if __name__ == "__main__":
    train()
