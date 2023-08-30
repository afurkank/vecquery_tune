import torch
import chromadb
import csv
import argparse

from utils.custom_embedding import CustomEmbeddingModel
from utils.model import CustomBERTModel, Tokenize

# define parser
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, required=True,
                    help='The name of the model to use')
parser.add_argument('--model_weights_path', type=str, required=True,
                    help='The path to the model weights file')
parser.add_argument('--data_path', type=str, required=True,
                    help='The path to the CSV file containing the data')
parser.add_argument('--documents_column', type=str, required=True,
                    help='The name of the column in the CSV file that contains the documents')
parser.add_argument('--metadata_columns', type=str, required=True,
                    help='The names of the columns in the CSV file that contain the metadata')
parser.add_argument('--collection_name', type=str, default='collection', required=False,
                    help='The name of the collection to create')
parser.add_argument('--max_len', type=int, default=256, required=False,
                    help='The maximum length of the input sequence')
args = parser.parse_args()

# define constants
MODEL_NAME = args.model_name
MODEL_WEIGHTS_PATH = args.model_weights_path
DATA_PATH = args.data_path
DOCUMENTS_COLUMN = args.documents_column
METADATA_COLUMNS = args.metadata_columns
COLLECTION_NAME = args.collection_name
MAX_LEN = args.max_len

# define model
model = CustomBERTModel(MODEL_NAME)
tokenizer = Tokenize(MODEL_NAME, MAX_LEN)

# check if max_len is valid
if MAX_LEN > model.bert.config.hidden_size:
    raise Exception(f"max_len must be less than or equal to {model.bert.config.hidden_size}")

# load model weights
model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH))

# read first row of CSV data
with open(DATA_PATH, 'r', encoding='utf-8') as csv_file:
    csv_reader = csv.reader(csv_file)
    for row in csv_reader:
        break
    delimiter = ';' if ';' in row[0] else ','

# read CSV data
datalist = []
with open(DATA_PATH, 'r', encoding='utf-8') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=delimiter)
    for row in csv_reader:
        datalist.append(row)

# get csv column names
column_names = datalist[0]
# get metadata column names
metadata_column_names = METADATA_COLUMNS.split(delimiter)
# remove column names from data
data = datalist[1:]

# check if metadata columns are in column names and documents column is in column names
for metadata_column_name in metadata_column_names:
    if metadata_column_name not in column_names:
        raise Exception(f"Metadata column name {metadata_column_name} not found in CSV column names")
if DOCUMENTS_COLUMN not in column_names:
    raise Exception(f"Documents column name {DOCUMENTS_COLUMN} not found in CSV column names")

# get data from columns
ids = []
documents = []
metadata = []
documents_column_index = column_names.index(DOCUMENTS_COLUMN)
for row in data:
    ids.append(row[0])
    documents.append(row[documents_column_index])
    metadata_dict = {}
    for column_name in metadata_column_names:
        metadata_dict[column_name] = row[column_names.index(column_name)]
    metadata.append(metadata_dict)

# create embedding function
embedding_function = CustomEmbeddingModel(model, tokenizer)

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
    name=COLLECTION_NAME,
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