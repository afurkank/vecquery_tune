# import libraries
import json
import torch
from argparse import ArgumentParser
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
# import custom modules
from utils.loss_funcs import CosineDistanceLoss
from utils.model import CustomBERTModel, Tokenize
from peft import LoraConfig, get_peft_model, TaskType
from transformers import TrainingArguments, Trainer

""" # define parser
parser = ArgumentParser()
parser.add_argument('--model_name', type=str, required=True,
                    help='The name of the model to use')
parser.add_argument('--data_path', type=str, required=True,
                    help='The path to the JSON file containing the data')
parser.add_argument('--path_to_save_model', type=str, default='./', required=False,
                    help='The path to save the trained model')
parser.add_argument('--epochs', type=int, default=20, required=False,
                    help='The number of epochs to train for')
parser.add_argument('--batch_size', type=int, default=32, required=False,
                    help='The batch size to use')
parser.add_argument('--max_len', type=int, default=256, required=False,
                    help='The maximum length of the input sequence')
parser.add_argument('--lr', type=float, default=0.00002, required=False,
                    help='The learning rate to use')
args = parser.parse_args()

# define constants
model_name = args.model_name
data_path = args.data_path
path_to_save_model = args.path_to_save_model
epoch = args.epochs
b_size = args.batch_size
max_len = args.max_len
lr = args.lr """

# define constants
model_name = "dbmdz/bert-base-turkish-uncased"
data_path = "scripts\data.json"
path_to_save_model = "./"
epoch = 5
b_size = 2
max_len = 256
lr = 1e-3

# define model and tokenizer
model = CustomBERTModel(model_name)
tokenizer = Tokenize(model_name, max_len)

# check if max_len is valid
if max_len > model.bert.config.hidden_size:
    raise Exception(f"max_len must be less than or equal to {model.bert.config.hidden_size}")

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
    
# load data
with open(data_path, 'r', encoding='utf-8') as file:
    data = json.load(file)
# check if data is in the correct format
for item in data:
    if 'input' not in item or 'output' not in item:
        print('The data is not in the correct format')
        exit()
dataset = CustomDataset(data, tokenizer)
data_loader = DataLoader(dataset, batch_size=b_size)

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

loss_func = CosineDistanceLoss()
optimizer = Adam(peft_model.parameters(), lr=lr)

for epoch in range(epoch):
    for batch in data_loader:
        # put data on device
        for key in batch:
            batch[key] = batch[key]
        # get embeddings
        emb1 = peft_model(batch['input_ids'], batch['attention_mask'])
        emb2 = peft_model(batch['correct_result_input_ids'], batch['correct_result_attention_mask'])
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
    print(f'Epoch {epoch+1}/{epoch}, Loss: {loss.item()}')

peft_model.save_pretrained("./")