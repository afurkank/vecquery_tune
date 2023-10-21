# import libraries
import json
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from peft import LoraConfig, get_peft_model
# import custom modules
from vecquery_tune.loss_funcs import CosineDistanceLoss
from vecquery_tune.model import CustomBERTModel, Tokenize

def get_model_and_tokenizer(model_name, max_len):
    model = CustomBERTModel(model_name)
    # check if max_len is valid
    if max_len > model.bert.config.hidden_size:
        raise Exception(f"max_len must be less than or equal to {model.bert.config.hidden_size}")
    tokenizer = Tokenize(model_name, max_len)
    return model, tokenizer

def get_data(data_path):
    with open(data_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    # check if data is in the correct format
    for item in data:
        if 'input' not in item or 'output' not in item:
            print('The data is not in the correct format')
            exit()
    return data

class CustomDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        inputs = self.tokenizer(self.data[idx]['input'])
        correct_answer = self.tokenizer(self.data[idx]['output'])
        result_dict = {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'correct_result_input_ids': correct_answer['input_ids'].squeeze(0),
            'correct_result_attention_mask': correct_answer['attention_mask'].squeeze(0)
        }
        return result_dict

def get_data_loader(data, tokenizer, batch_size):
    dataset = CustomDataset(data, tokenizer)
    return DataLoader(dataset, batch_size=batch_size)

def train(peft_model, data_loader, loss_func, optimizer, epochs, device):
    for epoch in range(epochs):
        for batch in data_loader:
            for key in batch:
                batch[key] = batch[key].to(device)
            emb1 = peft_model(batch['input_ids'], batch['attention_mask'])
            emb2 = peft_model(batch['correct_result_input_ids'], batch['correct_result_attention_mask'])
            # emb1 = (batch_size x embed_dim)
            # emb2 = (batch_size x embed_dim)

            loss = loss_func(emb1, emb2)

            loss.backward()

            optimizer.step()

            optimizer.zero_grad()
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')

def main(model_name, data_path, path_to_save_peft_folder, epochs, batch_size, max_len, lr):
    model, tokenizer = get_model_and_tokenizer(model_name, max_len)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lora_config = LoraConfig(
        r=8,
        lora_alpha=8,
        target_modules=["query", "value"],
        lora_dropout=0.05,
        bias="none",
    )
    peft_model = get_peft_model(
        model=model,
        peft_config=lora_config,
    )
    peft_model = peft_model.to(device)
    data = get_data(data_path)
    data_loader = get_data_loader(data, tokenizer, batch_size)
    loss_func = CosineDistanceLoss()
    optimizer = Adam(peft_model.parameters(), lr=lr)
    train(peft_model, data_loader, loss_func, optimizer, epochs, device)
    peft_model.save_pretrained(path_to_save_peft_folder+'peft_model')
    print(f'Peft model adapters saved to {path_to_save_peft_folder} as peft_folder')