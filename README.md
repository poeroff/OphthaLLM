OCT 안저 영상 기반 망막 질환 진단 및 리포트 생성을 위한 멀티모달 LLM

Overview

OphthaLLM은 OCT(광간섭단층촬영) 이미지를 입력받아, 망막의 5개 레이어(Nerve Fiber Layer, Ganglion Cell Layer, Inner Plexiform Layer, Outer Plexiform Layer, Photoreceptor IS/OS)별 소견을 자연어로 생성하는 vision-language 모델입니다.

  - Image Encoder: BLIP (OCT 도메인 fine-tuned, frozen)
  - Text Encoder: Bio_ClinicalBERT (frozen, contrastive alignment용)
  - Projector: 2-layer MLP (LLaVA-1.5 스타일, Q-Former 없이 patch token 전체를 LLM 공간으로 직접 투영)
  - LLM: Llama-3.2-3B-Instruct
  - Alignment: ITC(InfoNCE) + GLoRIA 스타일 레이어 단위 local matching(dense matching)

Training
  
  python OCT_LLM_Train_GLoRIA_MLP.py          # Stage 1 처음부터 (자동)
  STAGE=1 python OCT_LLM_Train_GLoRIA_MLP.py  # Stage 1 이어서
  python OCT_LLM_Train_GLoRIA_MLP.py          # Stage 2 자동 (stage1/best.pt 있으면)
  STAGE=2 python OCT_LLM_Train_GLoRIA_MLP.py  # Stage 2 이어서 (stage2/best.pt 있으면)

https://github.com/user-attachments/assets/079565db-86ee-4535-a6de-82a4b9d48a9c





