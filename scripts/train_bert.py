import mlflow
import mlflow.pytorch
from transformers import BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd
import torch


def train_model(data_path, model_path, experiment_name):

    df = pd.read_csv(data_path)
    dataset = Dataset.from_pandas(df[['tokens', 'label']])


    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)


    def preprocess_batch(batch):
        return {
            "input_ids": torch.tensor(batch["tokens"]),
            "labels": torch.tensor(batch["label"])
        }

    dataset = dataset.map(preprocess_batch)


    training_args = TrainingArguments(
        output_dir=model_path,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        save_steps=10_000,
        save_total_limit=2,
        logging_dir="./logs",
        logging_steps=500
    )


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )


    mlflow.set_experiment(experiment_name)
    with mlflow.start_run():
        trainer.train()
        mlflow.pytorch.log_model(model, "model")
        mlflow.log_param("epochs", 3)
        mlflow.log_param("batch_size", 8)


if __name__ == "__main__":
    train_model("data/processed/processed_data.csv", "models/", "bert-classification")
