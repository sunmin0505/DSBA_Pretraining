from transformers import AutoModel
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict, Any, Optional
import omegaconf

class EncoderForClassification(nn.Module):
    def __init__(self, model_config : omegaconf.DictConfig):
        super().__init__()
        
        # pre-trained encoder 불러오기
        self.encoder = AutoModel.from_pretrained(model_config.pretrained_model_name)

        # hidden size 추출
        hidden_size = self.encoder.config.hidden_size

        # classification head 구성
        self.dropout = nn.Dropout(float(model_config.dropout))
        self.classifier = nn.Linear(hidden_size, int(model_config.num_classes))


    def forward(self, input_ids : torch.Tensor, attention_mask : torch.Tensor, token_type_ids : Optional[torch.Tensor] = None, label : Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Inputs : 
            input_ids : (batch_size, max_seq_len)
            attention_mask : (batch_size, max_seq_len)
            token_type_ids : (batch_size, max_seq_len) # only for BERT
            label : (batch_size)
        Outputs :
            logits : (batch_size, num_labels)
            loss : (1)
        """
         # ModernBERT는 token_type_ids를 받지 않으므로, BERT 계열일 때만 전달
        encoder_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if getattr(self.encoder.config, "type_vocab_size", 0) > 0 and token_type_ids is not None:
            encoder_kwargs["token_type_ids"] = token_type_ids

        enc_out = self.encoder(**encoder_kwargs)

        # [CLS] 토큰의 hidden state를 대표 벡터로 사용
        cls = enc_out.last_hidden_state[:, 0, :]

        # Dropout + Classifier
        cls = self.dropout(cls)
        logits = self.classifier(cls)

        # output 구성
        out: Dict[str, Any] = {"logits": logits}
        if label is not None:
            out["loss"] = F.cross_entropy(logits, label)

        return out