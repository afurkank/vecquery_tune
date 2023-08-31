import torch
import chromadb

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

def load_model(model, model_weights_path):
    """
    Function to load model weights
    """
    model.load_state_dict(torch.load(model_weights_path))
    return model

def get_client(path):
    """
    Function to get client
    """
    client = chromadb.PersistentClient(path=path)
    return client

def get_collection(client, collection_name):
    """
    Function to get collection
    """
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
    return existing_collection

def get_embeddings_from_user_input(model, tokenizer, collection, num_results, metadata_columns, documents_column, device):
    """
    Function to get embeddings from user input
    """
    delimeter = ',' if metadata_columns.find(',') != -1 else ';'
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
        user_embedding = model(input_ids, attention_mask)
        # put embeddings on cpu
        user_embedding = user_embedding.cpu()
        # convert to list
        user_embedding = user_embedding.detach()[0].numpy().tolist()
        # get results
        result_dict = collection.query(user_embedding, n_results=num_results)

        metadatas = result_dict['metadatas']
        documents = result_dict['documents']

        for i in range(num_results):
            print(f'------------------ Result {i+1} ------------------')
            for metadata_column in metadata_columns.split(delimeter):
                print(f'{metadata_column}: {metadatas[0][i][metadata_column]}')
            print(f'{documents_column}: {documents[0][i]}')
            print('--------------------------------------------------')

def main(model_name,
        model_weights_path,
        collection_name,
        num_results,
        metadata_columns,
        documents_column,
        client_path,
        max_len):
    """
    Main function
    """
    # get model and tokenizer
    model, tokenizer = get_model_tokenizer(model_name, max_len)
    # get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # move model to device
    model.to(device)
    # load model weights
    model = load_model(model, model_weights_path)
    # get client
    client = get_client(path=client_path)
    # get collection
    collection = get_collection(client, collection_name)
    # get embeddings from user input
    get_embeddings_from_user_input(model, tokenizer, collection, num_results, metadata_columns, documents_column, device)