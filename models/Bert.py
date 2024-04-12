import torch.nn as nn
from transformers import BertModel
from .MutiCls import MutiCls

class BertCLS(nn.Module):
    def __init__(
        self,
        check_point_name,
        n_labels,
        freeze_pooler = True,
    ):
        super().__init__()
        self.n_labels = n_labels
        self.Bert = BertModel.from_pretrained(check_point_name)
        self.Cls = MutiCls(
            input_size = 768,
            n_labels = n_labels
        )
        
        for param in self.Bert.parameters():
            param.requires_grad = False

        if not freeze_pooler:
            for param in self.bert.pooler.parameters():
                param.requires_grad = True
        
    def forward(self, input_ids, attention_mask, token_type_ids):
        output = self.Bert(
            input_ids = input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids
        )
        x = output.pooler_output  #自带的pooler
        x = self.Cls(x)
        
        return x
        
        