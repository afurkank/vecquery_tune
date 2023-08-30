torchrun create_database.py \
    --model_name dbmdz/bert-base-turkish-cased \
    --model_weights_path model.pt \
    --data_path data.csv \
    --max_len 256 \
    --documents_column document \
    --metadata_columns metadata \
    --collection_name collection