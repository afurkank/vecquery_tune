# import libraries
import json
import torch
from argparse import ArgumentParser
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
# import custom modules
from utils.loss_funcs import CosineDistanceLoss
from utils.model import CustomBERTModel, Tokenize

# define parser
parser = ArgumentParser()
parser.add_argument('--model_name', type=str, default='bert-base-uncased', required=True,
                    help='The name of the model to use')
parser.add_argument('data_path', type=str, required=True,
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
MODEL_NAME = args.model_name
DATA_PATH = args.data_path
PATH_TO_SAVE_MODEL = args.path_to_save_model
EPOCH = args.epochs
BATCH_SIZE = args.batch_size
MAX_LEN = args.max_len
LR = args.lr

# define model
'''
The model must be one of huggingface's pretrained models.
You can find the models here:
https://huggingface.co/models

Make sure that the model you specify was trained on 
a corpus in the language you are fine-tuning for.
'''
model = CustomBERTModel(MODEL_NAME)
tokenizer = Tokenize(MODEL_NAME, MAX_LEN)

# check if max_len is valid
if MAX_LEN > model.bert.config.hidden_size:
    raise Exception(f"max_len must be less than or equal to {model.bert.config.hidden_size}")

# load data
with open(DATA_PATH, 'r', encoding='utf-8') as file:
    data = json.load(file)

# check if data is in the correct format
for item in data:
    if 'input' not in item or 'output' not in item:
        print('The data is not in the correct format')
        exit()

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

# create dataset and data loader
dataset = CustomDataset(data, tokenizer)
data_loader = DataLoader(dataset, batch_size=BATCH_SIZE)

# function to calculate loss with cosine distance
loss_func = CosineDistanceLoss()
# define optimizer
optimizer = Adam(model.parameters(), lr=LR)

# freeze the model's parameters except for the linear layer
for param in model.bert.parameters():
    param.requires_grad = False

# train model
for epoch in range(EPOCH):
    for batch in data_loader:
        # get embeddings
        emb1, mask1 = model(batch['input_ids'], batch['attention_mask'])
        emb2, mask2 = model(batch['correct_result_input_ids'], batch['correct_result_attention_mask'])

        # calculate loss
        loss = loss_func(emb1, emb2, mask1, mask2)

        # backpropagate loss
        loss.backward()

        # update weights
        optimizer.step()

        # zero gradients
        optimizer.zero_grad()
    print(f'Epoch {epoch+1}/{EPOCH}, Loss: {loss.item()}')

# save model
torch.save(model.state_dict(), PATH_TO_SAVE_MODEL + 'model.pt')