import torch
import chromadb
import csv
from argparse import ArgumentParser

from utils.custom_embedding import CustomEmbeddingModel
from utils.model import CustomBERTModel, Tokenize
from peft import PeftModel

# define parser
parser = ArgumentParser()
parser.add_argument('--model_name', type=str, required=True)
parser.add_argument('--csv_file_path', type=str, required=True)
parser.add_argument('--peft_folder_path', type=str, required=True)
parser.add_argument('--documents_col', type=str, required=True,
                    help='The name of the column in the CSV file that contains the documents')
parser.add_argument('--metadata_col', type=str, required=True,
                    help='The names of the columns in the CSV file that contains the metadata')
parser.add_argument('--collection_name', type=str, default='collection', required=False,
                    help='The name of the collection to create')
parser.add_argument('--max_len', type=int, default=256, required=False,
                    help='The maximum length of the input sequence')
args = parser.parse_args()

# define constants
model_name = args.model_name
data_path = args.csv_file_path
peft_folder_path = args.peft_folder_path
documents_col = args.documents_col
metadata_col = args.metadata_col
collection_name = args.collection_name
max_len = args.max_len

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

# read first row of CSV data
with open(data_path, 'r', encoding='utf-8') as csv_file:
    csv_reader = csv.reader(csv_file)
    for row in csv_reader:
        break
    delimeter = ';' if ';' in row[0] else ','

# read CSV data
datalist = []
with open(data_path, 'r', encoding='utf-8') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=delimeter)
    for row in csv_reader:
        datalist.append(row)

# get csv column names
column_names = datalist[0]
# get metadata column names
metadata_col_delim = ';' if ';' in metadata_col else ','
metadata_column_names = metadata_col.split(metadata_col_delim)
# remove column names from data
data = datalist[1:]

# check if metadata columns are in column names and documents column is in column names
for metadata_column_name in metadata_column_names:
    if metadata_column_name not in column_names:
        raise Exception(f"Metadata column name {metadata_column_name} not found in CSV column names")
if documents_col not in column_names:
    raise Exception(f"Documents column name {documents_col} not found in CSV column names")

# get data from columns
ids = []
documents = []
metadata = []
documents_column_index = column_names.index(documents_col)
for row in data:
    ids.append(row[0])
    documents.append(row[documents_column_index])
    metadata_dict = {}
    for column_name in metadata_column_names:
        col_idx = column_names.index(column_name)
        metadata_dict[column_name] = row[column_names.index(column_name)]
    metadata.append(metadata_dict)

# create embedding function
embedding_function = CustomEmbeddingModel(peft_model, tokenizer, device)

# create database
print("\nCreating database...\n")
client = chromadb.PersistentClient(path="./")
# check if collection already exists
for collection in client.list_collections():
    if collection.name == "collection":
        # delete collection
        print("Deleting collection...")
        client.delete_collection("collection")
        break
# create collection
print("Creating collection...")
collection = client.create_collection(
    name=collection_name,
    embedding_function=embedding_function,
    metadata={"hnsw:space": "cosine"}
)
# add data to collection
print("Adding data to collection, this may take a while...")
collection.add(
    ids=ids,
    documents=documents,
    metadatas=metadata
)