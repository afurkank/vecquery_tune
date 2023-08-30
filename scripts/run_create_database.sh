torchrun create_database.py \
    --model_name bert-base-uncased \
    --model_weights_path model.pt \
    --data_path data.csv \
    --documents_column document \
    --metadata_columns metadata \
    --collection_name collection \
    --max_len 256