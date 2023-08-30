python inference.py \
    --model_name bert-base-uncased \
    --model_weights_path model.pt \
    --documents_column documents \
    --metadata_columns metadata_column1,metadata_column2 \
    --collection_name collection \
    --num_results 5 \
    --max_len 256