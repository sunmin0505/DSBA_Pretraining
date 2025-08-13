from transformers import AutoTokenizer
from datasets import load_dataset, concatenate_datasets

import torch
from torch.utils.data import DataLoader
import omegaconf

from typing import List, Tuple, Literal, Optional, Dict, Any

class IMDBDataset(torch.utils.data.Dataset):
    '''
    역할:
        - IMDB 데이터셋 로드
        - __getitem__에서 text 1개를 tokenize -> dict와 label로 반환
        - batch 차원의 텐서 변환과 패딩은 collate_fn에서 처리
    '''

    # 모든 instance에서 공유되는 tokenizer (중복 방지)
    _tokenizer: Optional[AutoTokenizer] = None

    def __init__(self, data_config : omegaconf.DictConfig, split : Literal['train', 'valid', 'test']):
        self.split = split
        self.max_len = int(data_config.max_len)
        self.split_seed = int(getattr(data_config, "split_seed", 42))

        # tokenizer를 클래스 변수로 공유
        if IMDBDataset._tokenizer is None:
            IMDBDataset._tokenizer = AutoTokenizer.from_pretrained(
                data_config.model_name,
                use_fast=True
            )
        self.tokenizer: AutoTokenizer = IMDBDataset._tokenizer

        # IMDB load (train/test만 제공됨)
        raw = load_dataset('imdb')

        # train + test -> train/valid/test(8:1:1) split
        all_data = concatenate_datasets([raw["train"], raw["test"]])
        split_train_rest = all_data.train_test_split(
            test_size = 0.2,
            seed = self.split_seed,
            shuffle=True
        )
        split_valid_test = split_train_rest["test"].train_test_split(
            test_size = 0.5,
            seed = self.split_seed,
            shuffle=True
        )
        
        # 분할 데이터셋 선택
        train_set = split_train_rest['train']
        valid_set = split_valid_test['train']
        test_set = split_valid_test['test']
        if split == 'train':
            sel = train_set
        elif split == 'valid':
            sel = valid_set
        elif split == 'test':
            sel = test_set
        else:
            raise ValueError(f"Unknown split: {split}")

        # 필요한 필드만 보관
        self.texts: List[str] = sel['text']
        self.labels: List[int] = sel['label']

        # 데이터셋 크기 출력
        print(f">> SPLIT : {self.split:>5s} | num_samples : {len(self.texts)}")
        
    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, Any], int]:
        """
        역할:
            sample 하나를 tokenize하고 (input dict, 정답 label) 형태로 반환
        """
        text = self.texts[idx].strip()
        label = int(self.labels[idx])

        # text tokenize (패딩은 하지 않음)
        enc = self.tokenizer(
            text,                           
            truncation=True,                # max_len 초과시 자름
            max_length=self.max_len,        
            add_special_tokens=True,        # [CLS], [SEP] 등 추가
            return_attention_mask=True,     # attention mask 생성
            return_token_type_ids=False     # ModernBERT는 token_type_ids가 필요 없음, 또한 IMDB는 단일 문장이라 BERT에도도 필요 없음
        )

        return enc, label

    @staticmethod
    def collate_fn(batch : List[Tuple[dict, int]]) -> dict:
        """
        역할:
            - 동적 패딩 (배치 내 가장 긴 길이에 맞춤)
            - 텐서 변환 (torch.LongTensor)
            - token_type_ids가 없다면 0으로 채워서 생성
        """
        features, labels = zip(*batch)

        tokenizer = IMDBDataset._tokenizer
        assert tokenizer is not None, "Tokenizer is not initialized."

        # tokenizer가 배치 내 max_len에 맞춰 padding + 텐서로 변환
        padded = tokenizer.pad(
            features,
            padding=True,
            return_tensors='pt'
        )

        data_dict = {
            "input_ids": padded["input_ids"].long(),
            "attention_mask": padded["attention_mask"].long(),
            "label": torch.tensor(labels, dtype=torch.long),
        }

        return data_dict
    
def get_dataloader(data_config : omegaconf.DictConfig, split : Literal['train', 'valid', 'test']) -> DataLoader:
    """
    역할:
        - Dataset 생성
        - Dataloader 래핑 (train만 shuffle=True)
        - 성능 옵션 (num_workers, pin_memory)는 config로 제어
    """
    dataset = IMDBDataset(data_config, split)

    num_workers = int(getattr(data_config, "num_workers", 2))
    pin_memory = bool(getattr(data_config, "pin_memory", True))
    drop_last = bool(getattr(data_config, "drop_last", False)) if split == 'train' else False
    persistent_workers = num_workers > 0 and bool(getattr(data_config, "persistent_workers", True))
    
    loader = DataLoader(
        dataset, 
        batch_size = int(data_config.batch_size),   
        shuffle = (split=='train'),
        num_workers = num_workers,
        pin_memory = pin_memory,                    # GPU로 데이터 전송 시 성능 향상
        drop_last = drop_last,                      # train에서만 사용, 마지막 배치가 작을 경우 버림
        persistent_workers = persistent_workers,    # 에폭 간 워커 유지
        collate_fn = IMDBDataset.collate_fn
    )
    return loader