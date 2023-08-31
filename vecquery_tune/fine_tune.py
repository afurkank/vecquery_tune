# import libraries
import json
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
# import custom modules
from vecquery_tune.loss_funcs import CosineDistanceLoss
from vecquery_tune.model import CustomBERTModel, Tokenize

def get_model_tokenizer(model_name, max_len):
    """
    Function to get model and tokenizer
    """
    model = CustomBERTModel(model_name)
    # check if max_len is valid
    if max_len > model.bert.config.hidden_size:
        raise Exception(f"max_len must be less than or equal to {model.bert.config.hidden_size}")
    tokenizer = Tokenize(model_name, max_len)
    return model, tokenizer

def get_data(data_path):
    """
    Function to get data from JSON file
    """
    with open(data_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    # check if data is in the correct format
    for item in data:
        if 'input' not in item or 'output' not in item:
            print('The data is not in the correct format')
            exit()
    return data

# function to create iterable dataset from data
class CustomDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        inputs = self.tokenizer(self.data[idx]['input'])
        correct_answer = self.tokenizer(self.data[idx]['output'])
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'correct_result_input_ids': correct_answer['input_ids'].squeeze(0),
            'correct_result_attention_mask': correct_answer['attention_mask'].squeeze(0)
        }

def get_data_loader(data, tokenizer, batch_size):
    """
    Function to create data loader from dataset
    """
    dataset = CustomDataset(data, tokenizer)
    return DataLoader(dataset, batch_size=batch_size)

def train(model, data_loader, loss_func, optimizer, epochs, device):
    """
    Function to train model
    """
    # freeze the model's parameters except for the linear layer
    for param in model.bert.parameters():
        param.requires_grad = False
    # train model
    for epoch in range(epochs):
        for batch in data_loader:
            # put data on device
            for key in batch:
                batch[key] = batch[key].to(device)
            # get embeddings
            emb1 = model(batch['input_ids'], batch['attention_mask'])
            emb2 = model(batch['correct_result_input_ids'], batch['correct_result_attention_mask'])
            # emb1 = (batch_size x embed_dim)
            # emb2 = (batch_size x embed_dim)

            # calculate loss
            loss = loss_func(emb1, emb2)

            # backpropagate loss
            loss.backward()

            # update weights
            optimizer.step()

            # zero gradients
            optimizer.zero_grad()
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')

def main(model_name, data_path, path_to_save_model, epochs, batch_size, max_len, lr):
    """
    Main function
    """
    # get model and tokenizer
    model, tokenizer = get_model_tokenizer(model_name, max_len)
    # get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)
    # move model to device
    model.to(device)
    # get data
    data = get_data(data_path)
    # create data loader
    data_loader = get_data_loader(data, tokenizer, batch_size)
    # define loss function
    loss_func = CosineDistanceLoss()
    # define optimizer
    optimizer = Adam(model.parameters(), lr=lr)
    # train model
    train(model, data_loader, loss_func, optimizer, epochs, device)
    # save model
    torch.save(model.state_dict(), path_to_save_model + 'model.pt')
    print(f'Model saved to {path_to_save_model}model.pt')