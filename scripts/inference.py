import torch
import chromadb
import argparse

from utils.model import CustomBERTModel, Tokenize

# define parser
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='bert-base-uncased', required=True,
                    help='The name of the model to use')
parser.add_argument('--model_weights_path', type=str, default='./model.pt', required=True,
                    help='The path to the model weights file')
parser.add_argument('--documents_column', type=str, required=True,
                    help='The name of the column in the CSV file that contains the documents')
parser.add_argument('--metadata_columns', type=str, required=True,
                    help='The metadata columns you want to return')
parser.add_argument('--collection_name', type=str, default='collection', required=False,
                    help='The name of the collection')
parser.add_argument('--num_results', type=int, default=3, required=False,
                    help='The number of results to return')
parser.add_argument('--max_len', type=int, default=256, required=False,
                    help='The maximum length of the input sequence')
args = parser.parse_args()

# define constants
MODEL_NAME = args.model_name
MODEL_WEIGHTS_PATH = args.model_weights_path
DOCUMENTS_COLUMN = args.documents_column
METADATA_COLUMNS = args.metadata_columns
COLLECTION_NAME = args.collection_name
NUM_RESULTS = args.num_results
MAX_LEN = args.max_len

# define model
model = CustomBERTModel(MODEL_NAME)
tokenizer = Tokenize(MODEL_NAME, MAX_LEN)

# check if max_len is valid
if MAX_LEN > model.bert.config.hidden_size:
    raise Exception(f"max_len must be less than or equal to {model.bert.config.hidden_size}")

# load model weights
model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH))

# get client
client = chromadb.PersistentClient(path="./")

existing_collection = None
# check if collection exists
print(client.list_collections())
for collection in client.list_collections():
    if collection.name == COLLECTION_NAME:
        existing_collection = collection
        break
    else:
        collection = None
if existing_collection is None:
    print(f"Collection {COLLECTION_NAME} does not exist")
    print("Please create the collection first by running 'run_create_database.sh'")
    exit()

# get embeddings from user input
while True:
    user_input = input("Enter a sentence: ")
    if user_input == "exit":
        break
    input_ids = tokenizer(user_input)['input_ids']
    attention_mask = tokenizer(user_input)['attention_mask']
    user_embedding = model(input_ids, attention_mask).detach()[0].numpy().tolist()

    result_dict = existing_collection.query(user_embedding, n_results=NUM_RESULTS)

    metadatas = result_dict['metadatas']
    documents = result_dict['documents']
    for i in range(NUM_RESULTS):
        print(f'------------------ Result {i+1} ------------------')
        for metadata_column in METADATA_COLUMNS.split(','):
            print(f'{metadata_column}: {metadatas[0][i][metadata_column]}')
        print(f'{DOCUMENTS_COLUMN}: {documents[0][i]}')