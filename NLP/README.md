## 1. 개요
- 실험 목적
  - BERT와 ModerBERT의 text classification 성능 검증
  - IMDB dataset을 통한 sentiment classification

## 2. Dataset
- 데이터 구성
  - IMDB 영화 리뷰: 25,000개의 train + 25,000개의 test
  - 본 실험에서는 train과 test를 concat한 후, train:valid:test = 8:1:1로 분할
- 전처리
  - 토큰화: AutoTokenizer
  - max sequence length: 128
