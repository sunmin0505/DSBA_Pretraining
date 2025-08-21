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
1) 모든 모델의 learning rate (5e-5) 동일하게 고정
- train accuracy
  <img width="2528" height="1328" alt="1_train_acc" src="https://github.com/user-attachments/assets/331d8837-87fe-4ffd-9c25-28e86a0ab023" />
  
- train loss
  <img width="2528" height="1328" alt="1_train_loss" src="https://github.com/user-attachments/assets/07b6a6ad-7a64-4a27-9f9a-00c56cba28a8" />

- validation accuracy
  <img width="2528" height="1328" alt="1_val_acc" src="https://github.com/user-attachments/assets/620b01c0-ec1e-4127-945b-e6272e2d6396" />

- validation loss
  <img width="2528" height="1328" alt="1_val_loss" src="https://github.com/user-attachments/assets/23a6410a-fb34-48dd-9711-fabf9f0679a9" />

- test accuracy
  <img width="2528" height="1328" alt="1_test_acc" src="https://github.com/user-attachments/assets/42aa3c56-fd6c-4b3d-a5f9-8ce1954b0b7c" />

| Batch Size | 64 | 256 | 1024 |
|:---:|:---:|:---:|:---:|
| accuracy | 0.9084 | **0.9094** | 0.8902 |

2) Batch size 별로 learning rate를 sqrt scaling
- sqrt scaling을 적용한 이유
  - linear scaling을 적용했을 때, 256 batch size (lr = 2e-4)에서 성능이 급락함을 확인
    <img width="2528" height="1328" alt="test_lr" src="https://github.com/user-attachments/assets/ca1e02a6-c3cc-4227-8f18-efbe224bbd36" />
    | Learniung rate | 2e-4 | 1e-2 |
    |:---:|:---:|:---:|
    | accuracy | 0.8226 | **0.9094** |

- train accuracy
  <img width="2528" height="1328" alt="2_train_a" src="https://github.com/user-attachments/assets/b3661875-6301-435f-b29f-d1155e04bda4" />

- train loss
  <img width="2528" height="1328" alt="2_train_l" src="https://github.com/user-attachments/assets/388b7489-1f89-434d-9b94-5a7e50b26799" />

- validation accruracy
  <img width="2528" height="1328" alt="2_v_a" src="https://github.com/user-attachments/assets/5327e4c6-fbaa-41fd-afc0-3952e5bd72e5" />

- validation loss
  <img width="2528" height="1328" alt="2_v_l" src="https://github.com/user-attachments/assets/fa018921-0725-4371-afc9-aaa4e6a53b70" />

- test accuracy
  <img width="2528" height="1328" alt="2_test_a" src="https://github.com/user-attachments/assets/380f7f11-31ef-443d-82f7-01e4428d6f0c" />

| Batch Size | 64 (5e-5) | 256 (1e-4) | 1024 (2e-4) |
|:---:|:---:|:---:|:---:|
| accuracy | **0.9158** | 0.9126 | 0.8966 |

## 6. Discussion
- Batch size가 다를 경우, Learning rate의 조정이 필요함
  - Linear scaling, Sqrt scaling
- 마지막 checkpoint를 test에 적용한 이유
  - overfitting 영향까지 고려하기 위함?
