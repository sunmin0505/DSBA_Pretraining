## 1. 개요
- 실험 목적
  - BERT와 ModerBERT의 text classification 성능 검증
  - IMDB dataset을 통한 sentiment classification 수행

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
  - bert-base-uncased(110M)
  - modernBERT-base(149M)
## 4. Experiment Setup
- Hyperparameter
  - Optimizer: Adam
  - Learning rate: 5e-5
  - Scheduler: constant
  - Epochs: 5
  - Seed: 42
- Method
  - 총 5번의 epochs 중, 가장 성능이 높은 val에 대해서 test 수행
- Logging
  - Wandb로 train_acc, train_loss, val_acc, val_loss, test_acc 기록
## 5. Result
- train accuracy
<img width="600" height="300" alt="train_acc" src="https://github.com/user-attachments/assets/37e0a970-1b14-46d8-9ad3-b2273e23a389" />
- train loss
<img width="600" height="300" alt="train_loss" src="https://github.com/user-attachments/assets/791e2b4b-6b1e-4fa6-819e-a858cc531214" />
- valid accuracy
<img width="600" height="300" alt="val_acc" src="https://github.com/user-attachments/assets/71d3688d-0a41-4592-ac78-b3e2ffb1bcb6" />
- valid loss
<img width="600" height="300" alt="val_loss" src="https://github.com/user-attachments/assets/c6b5091e-408c-46bc-ac4b-b3df7d8a33e9" />
- test accuracy
<img width="600" height="300" alt="test_acc" src="https://github.com/user-attachments/assets/e440541e-b761-488c-8640-7b3b3e21475e" />
| | BERT | ModernBERT |
| accuracy | 0.9036 | 0.9148 |
## 6. Discussion
