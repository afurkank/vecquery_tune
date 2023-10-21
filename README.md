# Vector Query Optimizer


## How it works and what it does:

This project aims to allow easy fine-tuning and improvement of BERT models that are used as embedding models utilizing a Parameter Efficient Fine-Tuning(PEFT) method Low-Rank Adaptation(LoRA).

Fine-tuning and improvement of embedding models are needed when you are querying over vector databases and the results are not what you expect. Using LoRA results in efficient fine-tuning while also preventing catastrophic forgetting.

To see the improved results, this project also lets you run inference easily, utilizing the open-source vector database ChromaDB.

If you have a GPU, running scripts or using the package will automatically use it so you don't have to worry about it.
***
# How to Use


## Scripts

You can download the source code and run the scripts.

You can either run the bash scripts which are named like this: `run_script.sh` 

or 

Directly use `torchrun` command through terminal within the vecquery_tune/scripts directory. 

Here is an example usage for running the `fine_tune.py` script:

```
torchrun fine_tune.py \
    --model_name bert-base-uncased \
    --data_path data.json \
    --path_to_save_peft_folder ./ \
    --epochs 5 \
    --batch_size 32 \
    --max_len 256 \
    --lr 2e-5
```


## Package

First, install the package via pip:

`pip install vecquery_tune`

Then, you can use the 'FineTune' class to define a method with which you can 
fine-tune a BERT model.

Here is an example usage:

```
from vecquery_tune import FineTune

# fine tune model
fine_tune = FineTune(
    data_path='data.json',
    model_name='bert-base-uncased',
    path_to_save_peft_folder='./'
)

fine_tune(
    epochs=5,
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
    peft_folder_path='./peft_model',
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
    peft_folder_path='./peft_model',
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

# Issues with Installing ChromaDB

Please note that if you encounter issues while installing ChromaDB, you may need to install Visual Studio. You can take a look at [this SO answer](https://stackoverflow.com/a/76245995) for more information.
