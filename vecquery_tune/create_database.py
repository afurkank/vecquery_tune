import torch
import chromadb
import csv

from vecquery_tune.custom_embedding import CustomEmbeddingModel
from vecquery_tune.model import CustomBERTModel, Tokenize
from peft import PeftModel

def get_model_and_tokenizer(model_name, max_len):
    model = CustomBERTModel(model_name)
    # check if max_len is valid
    if max_len > model.bert.config.hidden_size:
        raise Exception(f"max_len must be less than or equal to {model.bert.config.hidden_size}")
    tokenizer = Tokenize(model_name, max_len)
    return model, tokenizer

def load_peft_model(model, peft_folder_path):
    peft_model = PeftModel.from_pretrained(
        model,
        peft_folder_path,
        is_trainable=False
    )
    return peft_model

def get_data(data_path):
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

    return datalist

def get_column_names(data):
    column_names = data[0]
    return column_names

def get_metadata_column_names(metadata_columns):
    splitter = ';' if ';' in metadata_columns else ','
    metadata_column_names = metadata_columns.split(splitter)
    return metadata_column_names

def get_data_from_columns(data, column_names, documents_column, metadata_columns):
    ids = []
    documents = []
    metadata = []
    documents_column_index = column_names.index(documents_column)
    for i, row in enumerate(data):
        ids.append(str(i))
        documents.append(row[documents_column_index])
        metadata_dict = {}
        for column_name in metadata_columns:
            metadata_dict[column_name] = row[column_names.index(column_name)]
        metadata.append(metadata_dict)
    return ids, documents, metadata

def check_column_parameters(column_names, documents_column, metadata_column_names):
    for metadata_column_name in metadata_column_names:
        if metadata_column_name not in column_names:
            raise Exception(f"Metadata column name {metadata_column_name} not found in CSV column names")
    if documents_column not in column_names:
        raise Exception(f"Documents column name {documents_column} not found in CSV column names")

def get_client(path):
    client = chromadb.PersistentClient(path=path)
    return client

def get_embedding_function(peft_model, tokenizer, device):
    embedding_function = CustomEmbeddingModel(peft_model, tokenizer, device)
    return embedding_function

def create_database(client, collection_name, embedding_function):
    # check if collection already exists
    for collection in client.list_collections():
        if collection.name == collection_name:
            # delete collection
            print("Deleting collection...")
            client.delete_collection(collection_name)
            break
    # create collection
    print("Creating collection...")
    collection = client.create_collection(
        name=collection_name,
        embedding_function=embedding_function,
        metadata={"hnsw:space": "cosine"}
    )
    return collection

def add_data_to_collection(collection, ids, documents, metadata):
    # add data to collection
    print("Adding data to collection, this may take a while...")
    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadata
    )

def main(model_name,
        peft_folder_path,
        data_path,
        collection_name,
        metadata_columns,
        documents_column,
        client_path,
        max_len):
    model, tokenizer = get_model_and_tokenizer(model_name, max_len)

    peft_model = load_peft_model(model, peft_folder_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    peft_model = peft_model.to(device)
    
    data = get_data(data_path)

    column_names = get_column_names(data)

    data = data[1:]

    metadata_column_names = get_metadata_column_names(metadata_columns)
    # check if metadata column names and documents column name are in column names
    check_column_parameters(column_names, documents_column, metadata_column_names)
    # get data from columns
    ids, documents, metadata = get_data_from_columns(data, column_names, documents_column, metadata_column_names)
    
    client = get_client(client_path)
    
    embedding_function = get_embedding_function(peft_model, tokenizer, device)
    # create database
    collection = create_database(client, collection_name, embedding_function)
    # add data to collection
    add_data_to_collection(collection, ids, documents, metadata)
    print("Done!")