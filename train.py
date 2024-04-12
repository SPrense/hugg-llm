import argparse
from tqdm import tqdm

import os
import pandas as pd
import numpy as np
import torch 
import torch.nn as nn
from transformers import AdamW

from dataset.thu_dataset import ThuCNewsdataset
from dataloader.thucn_dataloader import THuCNewsDataLoader
from models.Bert import BertCLS
from models.MutiCls import MutiCls

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='./pretrained/bert-base-chinese')
    parser.add_argument('--n_labels', type=int, default=10)
    parser.add_argument('--freeze_pooler', action='store_true')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--device', type=str, required=False)
    parser.add_argument('--project', type=str, default='bert_thucn_classification')
    parser.add_argument('--entity', type=str, default='jonahchow')
    parser.add_argument('--name', type=str, required=False)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data_path', type=str, default='/Users/hanlinwang/Documents/GitHub/hugg-llm/data/cnews/cnews_train.txt')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--bert_size', type=int, default=768)

    return parser.parse_args()

def train(configs):
    dataset = ThuCNewsdataset(configs.data_path)
    dataloader = THuCNewsDataLoader(
        dataset,
        configs.batch_size,
        configs.model_name,
        configs.max_length
    )

    model = BertCLS(
        configs.model_name,
        configs.n_labels,
        freeze_pooler = True,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=configs.lr
    )

    model.train()

    for epoch in range(configs.epochs):
        with tqdm(
            dataloader,
            total=len(dataloader),
            desc=f'Epoch {epoch + 1}/{configs.epochs}',
            unit='batch',
            ncols=100
        ) as pbar:
            for input_ids, attention_mask, token_type_ids, labels in dataloader:
                optimizer.zero_grad()
                outputs = model(input_ids, attention_mask,token_type_ids)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                outputs = outputs.argmax(dim=1)
                accuracy = (outputs == labels).float().mean()

                pbar.set_postfix(
                    loss=f'{loss.item():.3f}',
                    accuracy=f'{accuracy.item():.3f}'
                )
            # state_dict = {k: v for k, v in state_dict.items() if condition(k)}
            checkpoint_path = os.path.join(configs.checkpoint_dir, f'epoch_{epoch + 1}.pt')

if __name__ == '__main__':
    configs = argparser()
    train(configs)
        