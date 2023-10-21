from vecquery_tune.vecquery_tune import FineTune
model_name = 'dbmdz/bert-base-turkish-uncased'
fine_tune = FineTune(
    data_path='data.json',
    model_name=model_name,
    path_to_save_peft_folder='./'
)

fine_tune(
    epochs=1,
    batch_size=5,
    max_len=256,
    lr=2e-3
)