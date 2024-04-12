import pandas as pd
import torch 
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

class THuCNewsDataLoader(DataLoader):
    def __init__(
        self,
        dataset,
        batch_size,
        check_point_name,
        max_length,
        shuffle = True,
        drop_last = True,
    ):
        super(THuCNewsDataLoader, self).__init__(  
            dataset,  
            batch_size=batch_size,  
            shuffle=shuffle,  
            drop_last=drop_last,  
            collate_fn=self.collate_fn   
        )  
        self.tokenizer = AutoTokenizer.from_pretrained(check_point_name)
        self.max_length = max_length
        
        self.dataloader = DataLoader(
            dataset = self.dataset,
            collate_fn = self.collate_fn,
            batch_size = self.batch_size,
            shuffle = shuffle,
            drop_last = drop_last
        )
    def collate_fn(self, data):
        labels = [data[i][0] for i in range(len(data))]
        content = [data[i][1] for i in range(len(data))]
        
        tokens = self.tokenizer.batch_encode_plus(
            batch_text_or_text_pairs = content,
            truncation=True,
            padding = 'max_length',
            max_length = self.max_length,
            return_tensors ='pt',
        )
        
        input_ids = tokens.input_ids
        attention_mask = tokens.attention_mask
        token_type_ids = tokens.token_type_ids
        labels = torch.LongTensor(labels)
        
        return input_ids, attention_mask, token_type_ids, labels
    
    def __len__(self):
        return len(self.dataloader)
    def __getitem__(self):
        for data in self.dataloader:
            yield data