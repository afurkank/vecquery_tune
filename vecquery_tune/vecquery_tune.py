from fine_tune import main as fine_tune_main
from inference import main as inference_main
from create_database import main as create_database_main

# class to run fine-tuning
class FineTune:
    '''
    This class is used to run fine-tuning.

    data_path: path to your data of type JSON
    model_name: name of the pretrained BERT model
    path_to_save_model: path to save the fine-tuned model weights
    '''
    def __init__(self,
                    data_path:str,
                    model_name:str='bert-base-uncased',
                    path_to_save_model:str='./') -> None:
        self.model_name = model_name
        self.data_path = data_path
        self.path_to_save_model = path_to_save_model

    def __call__(self, epochs:int=20, batch_size:int=32, max_len:int=256, lr:float=2e-5):
        '''
        epochs: number of epochs
        batch_size: batch size
        max_len: maximum length of the input
        lr: learning rate
        '''
        fine_tune_main(
            model_name=self.model_name,
            data_path=self.data_path,
            path_to_save_model=self.path_to_save_model,
            epochs=epochs,
            batch_size=batch_size,
            max_len=max_len,
            lr=lr
        )

# class to create database
class CreateDatabase:
    '''
    This class is used to create database.

    data_path: path to your data of type CSV
    model_name: name of the pretrained BERT model
    model_weights_path: path to fine-tuned model weights
    collection_name: name of the collection (you can leave it as it is)
    client_path: path to chromadb client (you can leave it as it is)
    '''
    def __init__(self,
                    data_path:str,
                    model_weights_path:str,
                    model_name:str='bert-base-uncased',
                    collection_name:str='collection',
                    client_path:str='./') -> None:
        self.model_weights_path = model_weights_path
        self.data_path = data_path
        self.model_name = model_name
        self.collection_name = collection_name
        self.client_path = client_path

    def __call__(self, metadata_column:str, documents_column:str, max_len:int=256):
        '''
        metadata_column: metadata column names separated by comma or semicolon
        documents_column: documents column name
        max_len: maximum length of the input (you can leave it as it is)
        '''
        create_database_main(
            model_name=self.model_name,
            model_weights_path=self.model_weights_path,
            data_path=self.data_path,
            collection_name=self.collection_name,
            metadata_columns=metadata_column,
            documents_column=documents_column,
            client_path=self.client_path,
            max_len=max_len
        )

# class to run inference
class Inference:
    '''
    This class is used to run inference.

    model_weights_path: path to fine-tuned model weights
    model_name: name of the pretrained BERT model
    collection_name: name of the collection (you can leave it as it is)
    num_results: number of results to return (returns top k results)
    '''
    def __init__(self,
                model_weights_path:str,
                model_name:str='bert-base-uncased',
                collection_name:str='collection',
                client_path:str='./',
                num_results:int=5) -> None:
        self.model_name = model_name
        self.model_weights_path = model_weights_path
        self.collection_name = collection_name
        self.client_path = client_path
        self.num_results = num_results

    def __call__(self, metadata_column:str, documents_column:str, max_len:int=256):
        '''
        metadata_column: metadata column names separated by comma or semicolon
        documents_column: documents column name
        client_path: path to client (you can leave it as it is)
        max_len: maximum length of the input (you can leave it as it is)
        '''
        inference_main(
            model_name=self.model_name,
            model_weights_path=self.model_weights_path,
            collection_name=self.collection_name,
            client_path=self.client_path,
            num_results=self.num_results,
            metadata_columns=metadata_column,
            documents_column=documents_column,
            max_len=max_len
        )