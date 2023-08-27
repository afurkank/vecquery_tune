# Vector Query Optimizer

This project aims to use any pre-trained BERT model from the HuggingFace library 'transformers'
as a custom embedding function. It is useful when your query results are sometimes not related to your input.

To see the improved results, it utilizes the open-source vector database ChromaDB.

The structure of the custom embedding model is simply adding another linear layer on top of the base 
BERT model of your choice, and freezing the original BERT model's parameters before fine-tuning the 
weights of the linear layers. By doing this, it preserves the original knowledge of the BERT model, 
thus only improving the search results without causing 'forgetting'.



# Requirements for Fine-Tuning

To fine-tune the model, you will need to specify a model name to use as the base model. This model must be a pre-trained
BERT model from the HuggingFace library 'transformers'. The same model name is also used for the tokenizer.
The fine-tuning process also requires you to supply a data file which is a JSON file, including the inputs and outputs.
Here is what the structure of the file must look like:
```
[
  {
    "input": 'input for query',
    "output": 'desired output of the query search'
  },
]
```
The path to this JSON file must be supplied when running fine-tuning.

Other parameters can be found in the 'run_finetune.sh' file, but they are not required.

Note: Please make sure the pre-trained BERT model of your choice was trained on a corpus of the language you intend to query
for.



# Requirements for Creating Database

To run inference, you first need to create a database. The 'create_database.py' script utilizes the open-source vector 
database ChromaDB. You don't need to worry too much about how this vector db works, you can supply the script with 
the necessary data and use the created local database to run inference.

To run the script, you need to supply:
- a model name
- path to your fine-tuned model weights
- path to your CSV file containing your data
- document column name -the name of the column from your CSV file, which will be used to create vector embeddings-
- metadata column names -the name of the columns from your CSV file to be used as metadata (these can be all the rest of the columns except for the
document column)-
- collection name which will be the name of the newly created vector db after running the script.

Here are the specifics about your CSV file:
- The delimiter must be either ';' or ','.
- It must include an index column, starting from 1, and the column must be the first column.
- Document column is the name of the column whose values will be used to create the vector embeddings.
- Metadata columns are used for any other related data that will be saved along your vector embeddings, however,
they will not be used for embeddings. You can use them for filtering the search results afterward. For more details,
please refer to the documentation of ChromaDB: https://docs.trychroma.com/usage-guide
The metadata column parameters must be given like this:
`metadata_column1,metadata_column2,...`
- You don't need to specify a collection name, the default is 'collection'. However, the same name will be used for
running inference, so if you specify a collection name, remember to supply it to the 'inference.py' script as a parameter.
- The format should more or less look like this:
`index;metadata_column1;documents_column;metadata_column2;...`



# Requirements for Inference

'inference.py' will need: 
- a model name
- the path to your fine-tuned model weights
- the name of the document column
- the names of the metadata columns that you gave to the 'create_database.py' script
- the name of the newly created collection if you have used a different collection name other than 'collection' when running the 'create_database.py' script
- There are other parameters you can supply but they are not required. For example, you can specify the number of results to be returned for your query.



# Usage

The usage order is like this: 'fine_tune.py' -> 'create_database.py' -> 'inference.py'

First, run the 'run_finetune.sh' script to obtain trained model weights.
Then, run the 'run_create_database.sh' script to create a database that will utilize the fine-tuned model.
Finally, run the 'run_inference.sh' to query any sentence you want and see the improved results. You can write 'exit' to break out of the program.

Note: Don't forget to supply the 'run_create_database.sh' and 'run_inference.sh' scripts with the 'documents_column' and 'metadata_columns' parameters.

Note: If you can't use the 'bash _.sh' command, you can also use the 'torchrun script.py --model_name ...' command in the same directory as the scripts. Don't forget
to change the 'script' with the name of the script you want to run.










