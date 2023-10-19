import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

class CustomBERTModel(nn.Module):
    def __init__(self, model_name):
        super(CustomBERTModel, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
    def forward(self, input_ids, attention_mask):
        # implement the forward pass
        outputs = self.bert(input_ids, attention_mask)
        last_hidden_state = outputs.last_hidden_state
        # last_hidden_state = (batch_size x seq_len x embed_dim)
        # attention_mask = (batch_size x seq_len)
        # mask the embeddings that are padded
        attention_mask = attention_mask.float()
        last_hidden_state = last_hidden_state * attention_mask.unsqueeze(-1)
        # sum the embeddings along the sentence length dimension
        weighted_sum = torch.sum(last_hidden_state * attention_mask.unsqueeze(-1), dim=1)
        # divide the sum by the number of non-padded tokens (i.e. the number of 1s in the attention mask)
        mean_pooled = weighted_sum / attention_mask.sum(dim=1, keepdim=True) # dimensions: (batch_size, embed_dim)
        return mean_pooled

class Tokenize(nn.Module):
    def __init__(self, model_name, max_len) -> None:
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.max_len = max_len
    def forward(self, input):
        inputs = self.tokenizer(
            input,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=self.max_len
        )
        return inputs