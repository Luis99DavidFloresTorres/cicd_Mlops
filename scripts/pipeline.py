import os
from scripts.preprocess import preprocess_data
from scripts.train_bert import train_model

def run_pipeline(raw_data_path, processed_data_path, model_path, experiment_name):
    print("Preprocesando datos...")
    preprocess_data(raw_data_path, processed_data_path)

    print("Entrenando modelo...")
    train_model(processed_data_path, model_path, experiment_name)

if __name__ == "__main__":
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    run_pipeline(
        raw_data_path="data/raw/dataset.csv",
        processed_data_path="data/processed/processed_data.csv",
        model_path="models/",
        experiment_name="bert-classification"
    )