## 1. 개요
- 실험 목적
  - ModerBERT의 batch size별 text classification 성능 검증
  - IMDB dataset을 통한 sentiment classification 수행
  - Single GPU 하에, 큰 batch size(256, 1024)를 적용하기 위해 gradient accumulation 적용
  - HuggingFace의 Accelerator 모듈 사용
## 2. Dataset
- 데이터 구성
  - IMDB 영화 리뷰: 25,000개의 train + 25,000개의 test
  - 본 실험에서는 train과 test를 concat한 후, train:valid:test = 8:1:1로 분할
- 전처리
  - 토큰화: AutoTokenizer
  - max sequence length: 128
## 3. Model
- EncoderForClassification 직접 구현
  - 사전학습된 Encoder + Dropout + Linear layer
- Hugging Face에서 불러옴
  - modernBERT-base(149M)
- Batch-size
  - 64 (accumulation step = 1)
  - 256 (accumulation step = 4)
  - 1024 (accumulation step = 16)
- Learning rate
  - 총 2가지 실험을 진행
  1) Learning rate 5e-5 고정
  2) Learning rate sqrt scaling (5e-5, 1e-4, 2e-4)
## 4. Experiment Setup
- Hyperparameter
  - Optimizer: Adam
  - Scheduler: constant
  - Epochs: 5
  - Seed: 42
- Method
  - 총 5번의 epochs 중, 마지막 모델에 대해서 test 수행
- Logging
  - Accelerator로 train_acc, train_loss, val_acc, val_loss, test_acc 기록
## 5. Result
1) 모든 모델의 learning rate 동일하게 고정
- test accuracy
- test loss
| Batch Sizse | 64 | 256 | 1024 |
|:---:|:---:|:---:|:---:|
| accuracy | 0.9036 | 0.9148 | 0.0000 |

2) Batch size 별로 learning rate를 sqrt scaling

| Batch Sizse | 64 | 256 | 1024 |
|:---:|:---:|:---:|:---:|
| accuracy | 0.9036 | 0.9148 | 0.0000 |

## 6. Discussion
- ModerBERT의 성능이 더 높았음
  1) 긴 문맥 처리 능력
  2) Atttention 설계(local+global 교차 사용)가 효율적
  3) GeGLU 활성화 함수를 사용하여 표현력을 높임
  4) bias 제거, unpadding 등으로 계산 효율성 증가
- Coding
  - token_type_ids 구현하지 않음: ModernBERT는 token_type_ids가 필요 없음, 또한 IMDB는 단일 문장이라 BERT에도 필요 없음
