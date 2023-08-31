# Vector Query Optimizer

## What it Does

This project aims to optimize the vector query results when you are using a custom embedding function to obtain vector embeddings from your documents. The custom embedding function can be any pre-trained BERT model from the HuggingFace library 'transformers'. 

This is useful when your query results are sometimes not related to your input.

To see the improved results, this project also allows you to run inference easily, utilizing the open-source vector database ChromaDB.

## How it Works

The structure of the custom embedding model is simply adding another linear layer on top of the base BERT model of your choice, and freezing the original BERT model's parameters before fine-tuning the weights of the linear layer. By doing this, it preserves the original knowledge of the BERT model, thus only improving the search results without causing 'forgetting'.

# How to Use

## Scripts

You can download the source code and run the scripts.

You can either run the bash scripts which are named like this: `run_script.sh` 

or 

Directly use `torchrun` command through terminal within the vecquery_tune/scripts directory. 

Here is an example usage:

```
torchrun fine_tune.py \
    --model_name bert-base-uncased \
    --data_path data.json \
    --path_to_save_model ./ \
    --epochs 20 \
    --batch_size 32 \
    --max_len 256 \
    --lr 2e-5
```

## Package

First, install the package via pip:

`pip install vecquery_tune`

Then, you can use the 'FineTune' class to define a method with which you can 
fine-tune a BERT model of your choice to better optimize the results of query 
searches on vector databases.

Here is an example usage:

```
from vecquery_tune.vecquery_tune import FineTune

# fine tune model
fine_tune = FineTune(
    data_path='data.json',
    model_name='bert-base-uncased',
    path_to_save_model='./'
)

fine_tune(
    epochs=20,
    batch_size=32,
    max_len=256,
    lr=2e-5
)
```

'data.json' file must be of the format:
```
[
    {
        'input': 'query input',
        'output': 'the desired output of the query'
    },
]
```

To see the improved results, you need to first create a database and add your data into it.
This package utilizes ChromaDB to run inference and see the results. Before using the 
'Inference' class, you need to use 'CreateDatabase' class and create a database.

Here is how you can use the 'CreateDatabase' class:

```
from vecquery_tune.vecquery_tune import CreateDatabase

# create database
create_database = CreateDatabase(
    data_path='data.csv',
    model_name='bert-base-uncased',
    model_weights_path='./model.pt',
    collection_name='collection',
    client_path='./'
)

create_database(
    metadata_column='time,author', # seperate metadata columns by either ';' or ','
    documents_column='docs',
    max_len=256
)
```

After creating the database, you can run inference via the 'Inference' class.
Keep in mind that if you supplied a collection name and a client path different than the 
default ones, you need to supply them here as well.

Here is how you can use the 'Inference' class:

```
from vecquery_tune.vecquery_tune import Inference

# inference
inference = Inference(
    model_weights_path='./model.pt',
    model_name='bert-base-uncased',
    collection_name='collection',
    client_path='./',
    num_results=5
)

inference(
    metadata_column='author',
    documents_column='docs',
    max_len=256
)
```

# What's Next
- Add JSON formatted data option for creating the database and running inference.
- Add GPU support **Done**

# Issues with Installing ChromaDB

Please note that if you encounter issues while installing ChromaDB, you may need to install Visual Studio. You can take a look at [this SO answer](https://stackoverflow.com/a/76245995) for more information.
