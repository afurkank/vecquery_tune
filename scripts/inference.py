import torch
import chromadb
import argparse

from utils.model import CustomBERTModel, Tokenize
from peft import PeftModel

# define parser
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, required=True)
parser.add_argument('--peft_folder_path', type=str, required=True)
parser.add_argument('--documents_col', type=str, required=True,
                    help='The name of the column in the CSV file that contains the documents')
parser.add_argument('--metadata_col', type=str, required=True,
                    help='The metadata columns you want returned')
parser.add_argument('--collection_name', type=str, default='collection', required=False,
                    help='The name of the collection')
parser.add_argument('--num_results', type=int, default=3, required=False,
                    help='The number of results returned')
parser.add_argument('--max_len', type=int, default=256, required=False,
                    help='The maximum length of the input sequence')
args = parser.parse_args()

# define constants
model_name = args.model_name
peft_folder_path = args.peft_folder_path
documents_col = args.documents_col
metadata_col = args.metadata_col
collection_name = args.collection_name
num_results = args.num_results
max_len = args.max_len
delimeter = ';' if ';' in metadata_col else ','

# get device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device)

# define model
model = CustomBERTModel(model_name)
tokenizer = Tokenize(model_name, max_len)

# check if max_len is valid
if max_len > model.bert.config.hidden_size:
    raise Exception(f"max_len must be less than or equal to {model.bert.config.hidden_size}")

peft_model = PeftModel.from_pretrained(
    model,
    peft_folder_path,
    is_trainable=False,
)

# move model to device
peft_model = peft_model.to(device)

# get client
client = chromadb.PersistentClient(path="./")

existing_collection = None
# check if collection exists
print(client.list_collections())
for collection in client.list_collections():
    if collection.name == collection_name:
        existing_collection = collection
        break
    else:
        collection = None
if existing_collection is None:
    print(f"Collection {collection_name} does not exist")
    print("Please create the collection first by running 'run_create_database.sh'")
    exit()

# get embeddings from user input
while True:
    user_input = input("Enter a sentence: ")
    if user_input == "exit":
        break
    input_ids = tokenizer(user_input)['input_ids']
    attention_mask = tokenizer(user_input)['attention_mask']
    # put data on device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    # get embeddings
    user_embedding = peft_model(input_ids, attention_mask)
    # put embeddings on cpu
    user_embedding = user_embedding.cpu()
    # convert embedding tensor to list
    user_embedding = user_embedding.detach()[0].numpy().tolist()
    # get results
    result_dict = existing_collection.query(user_embedding, n_results=num_results)
    
    metadatas = result_dict['metadatas']
    documents = result_dict['documents']
    for i in range(num_results):
        print(f'------------------ Result {i+1} ------------------')
        for metadata_column in metadata_col.split(delimeter):
            print(f'{metadata_column}: {metadatas[0][i][metadata_column]}')
        print(f'{documents_col}: {documents[0][i]}')