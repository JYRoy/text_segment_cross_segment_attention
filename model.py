from dataclasses import dataclass
import transformers
from transformers import BertModel, BertTokenizer

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

bert = BertModel.from_pretrained("bert-base-uncased")
# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
MAX_TOKENS = 200


@dataclass
class CrossSegConfig:
    n_class: int = 2


class CrossSegModel(nn.Model):

    def __init__(self, config: CrossSegConfig):
        self.config = config

        self.bert = bert

        self.rule = nn.ReLU()
        self.fc = nn.Linear(768, config.n_class)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask, token_type_ids):
        bert_output = self.bert(input_ids, attention_mask)
        out = bert_output["last_hidden_state"]
        x = self.fc1(out)
        x = self.softmax(x)
        return x
