torchrun inference.py \
    --model_name dbmdz/bert-base-turkish-cased \
    --model_weights_path model.pt \
    --collection_name collection \
    --documents_column documents \
    --metadata_columns metadata_column1,metadata_column2 \
    --num_results 5 \
    --max_len 256