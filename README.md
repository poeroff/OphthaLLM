# OphthaLLM

OCT 안저 영상 기반 망막 질환 진단 및 리포트 생성을 위한 멀티모달 LLM

https://github.com/user-attachments/assets/079565db-86ee-4535-a6de-82a4b9d48a9c

---

## Overview

OphthaLLM은 OCT(광간섭단층촬영) 이미지를 입력받아, 망막의 5개 레이어(Nerve Fiber Layer, Ganglion Cell Layer, Inner Plexiform Layer, Outer Plexiform Layer, Photoreceptor IS/OS)별 소견을 자연어로 생성하는 vision-language 모델입니다.

| 구성 요소 | 내용 |
|---|---|
| **Image Encoder** | BLIP (OCT 도메인 fine-tuned, frozen) |
| **Text Encoder** | Bio_ClinicalBERT (frozen, contrastive alignment용) |
| **Projector** | 2-layer MLP (LLaVA-1.5 스타일, Q-Former 없이 patch token 전체를 LLM 공간으로 직접 투영) |
| **LLM** | Llama-3.2-3B-Instruct |
| **Alignment** | ITC(InfoNCE) + GLoRIA 스타일 레이어 단위 local matching(dense matching) |

```
OCT Image → BLIP Image Encoder → MLP Projector → Llama-3.2-3B-Instruct → Clinical Report
                                        ↑
                    Bio_ClinicalBERT ← GLoRIA Local Attention (layer-wise alignment)
```

---

## Results

**Recall**

<img width="898" height="322" alt="recall" src="https://github.com/user-attachments/assets/d7261e82-c7cc-47a4-aee4-7fe331ffc11a" />

**Captioning**

<img width="602" height="494" alt="captioning" src="https://github.com/user-attachments/assets/3e3ab267-6983-4bc8-94f2-8a420c7baaec" />

**Disease Classification**

<img width="897" height="523" alt="disease classification" src="https://github.com/user-attachments/assets/a8d57df5-ff99-488a-9db3-0bebaf6acc8e" />

---

## Training

2-stage 학습 구조입니다. Stage 2는 `stage1/best.pt`가 있으면 자동으로 진입합니다.

```bash
# Stage 1 처음부터 (자동)
python OCT_LLM_Train_GLoRIA_MLP.py

# Stage 1 이어서
STAGE=1 python OCT_LLM_Train_GLoRIA_MLP.py

# Stage 2 자동 (stage1/best.pt 있으면)
python OCT_LLM_Train_GLoRIA_MLP.py

# Stage 2 이어서 (stage2/best.pt 있으면)
STAGE=2 python OCT_LLM_Train_GLoRIA_MLP.py
```

---

## Project Structure

```
OphthaLLM/
└── LM/
    ├── model.py
    └── GLoRIA/
        ├── gloria_itc.py
        └── OCT_LLM_Train_GLoRIA_MLP.py
```
