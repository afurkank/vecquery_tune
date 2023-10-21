import torch
import chromadb
from peft import PeftModel
from model import CustomBERTModel, Tokenize

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

def get_client(path):
    client = chromadb.PersistentClient(path=path)
    return client

def get_collection(client, collection_name):
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

def get_embeddings_from_user_input(
        peft_model,
        tokenizer,
        collection,
        num_results,
        metadata_columns,
        documents_column,
        device
    ):
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
        user_embedding = peft_model(input_ids, attention_mask)
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
        peft_folder_path,
        collection_name,
        num_results,
        metadata_columns,
        documents_column,
        client_path,
        max_len):
    model, tokenizer = get_model_and_tokenizer(model_name, max_len)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    peft_model = load_peft_model(model, peft_folder_path)

    peft_model = peft_model.to(device)

    client = get_client(path=client_path)

    collection = get_collection(client, collection_name)

    get_embeddings_from_user_input(
        model,
        tokenizer,
        collection,
        num_results,
        metadata_columns,
        documents_column,
        device
    )