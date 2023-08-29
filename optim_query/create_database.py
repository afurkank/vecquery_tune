import torch
import chromadb
import csv

from .custom_embedding import CustomEmbeddingModel
from .model import CustomBERTModel, Tokenize

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

def load_model(model, model_weights_path):
    """
    Function to load model weights
    """
    model.load_state_dict(torch.load(model_weights_path))
    return model

def get_data_and_delimiter(data_path):
    """
    Function to get data from CSV file
    """
    # read first row of CSV data
    with open(data_path, 'r', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            break
        delimiter = ';' if ';' in row[0] else ','
    
    # read CSV data
    datalist = []
    with open(data_path, 'r', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=delimiter)
        for row in csv_reader:
            datalist.append(row)

    return datalist, delimiter

def get_column_names(data):
    """
    Function to get column names from data
    """
    column_names = data[0]
    return column_names

def get_metadata_column_names(metadata_columns):
    """
    Function to get metadata column names
    """
    splitter = ';' if ';' in metadata_columns else ','
    metadata_column_names = metadata_columns.split(splitter)
    return metadata_column_names

def get_data_from_columns(data, column_names, documents_column, metadata_columns):
    """
    Function to get data from columns
    """
    ids = []
    documents = []
    metadata = []
    documents_column_index = column_names.index(documents_column)
    for i, row in enumerate(data):
        ids.append(i)
        documents.append(row[documents_column_index])
        metadata_dict = {}
        for column_name in metadata_columns:
            metadata_dict[column_name] = row[column_names.index(column_name)]
        metadata.append(metadata_dict)
    return ids, documents, metadata

def check_column_parameters(column_names, documents_column, metadata_column_names):
    """
    Function to check if column parameters are in column names
    """
    for metadata_column_name in metadata_column_names:
        if metadata_column_name not in column_names:
            raise Exception(f"Metadata column name {metadata_column_name} not found in CSV column names")
    if documents_column not in column_names:
        raise Exception(f"Documents column name {documents_column} not found in CSV column names")

def get_client(path):
    """
    Function to get client
    """
    client = chromadb.PersistentClient(path=path)
    return client

def get_embedding_function(model, tokenizer):
    """
    Function to get embedding function
    """
    embedding_function = CustomEmbeddingModel(model, tokenizer)
    return embedding_function

def create_database(client, collection_name, embedding_function):
    """
    Function to create database
    """
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
    """
    Function to add data to collection
    """
    # add data to collection
    print("Adding data to collection, this may take a while...")
    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadata
    )

def main(model_name,
        model_weights_path,
        data_path,
        collection_name,
        metadata_columns,
        documents_column,
        client_path,
        max_len):
    """
    Main function
    """
    # get model and tokenizer
    model, tokenizer = get_model_tokenizer(model_name, max_len)
    # load model
    model = load_model(model, model_weights_path)
    # get data
    data, delimiter = get_data_and_delimiter(data_path)
    # get column names
    column_names = get_column_names(data)
    # get metadata column names
    metadata_column_names = get_metadata_column_names(metadata_columns)
    # check if metadata column names and documents column name are in column names
    check_column_parameters(column_names, documents_column, metadata_column_names)
    # get data from columns
    ids, documents, metadata = get_data_from_columns(data, column_names, documents_column, metadata_column_names)
    # get client
    client = get_client(client_path)
    # get embedding function
    embedding_function = get_embedding_function(model, tokenizer)
    # create database
    collection = create_database(client, collection_name, embedding_function)
    # add data to collection
    add_data_to_collection(collection, ids, documents, metadata)
    print("Done!")