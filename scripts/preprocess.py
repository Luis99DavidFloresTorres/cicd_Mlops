import pandas as pd
import boto3
from transformers import BertTokenizer

s3 = boto3.client('s3')
def download_from_s3(bucket_name, s3_file_name, local_file_path):
    try:
        s3.download_file(bucket_name, s3_file_name, local_file_path)
        print(f"Archivo {s3_file_name} descargado exitosamente desde {bucket_name}")
    except Exception as e:
        print(f"Error al descargar el archivo: {e}")

def preprocess_data(bucket_name, s3_file_name, local_file_path, output_path):

    download_from_s3(bucket_name, s3_file_name, local_file_path)


    df = pd.read_csv(local_file_path)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    df['cleaned_text'] = df['text'].str.lower()
    df['tokens'] = df['cleaned_text'].apply(lambda x: tokenizer.encode(x, truncation=True, padding='max_length', max_length=128))


    df.to_csv(output_path, index=False)
    print(f"Datos preprocesados guardados en {output_path}")

if __name__ == "__main__":
    preprocess_data(
        bucket_name="bert-mlflow-bucket",
        s3_file_name="raw/dataset.csv",
        local_file_path="data/raw/dataset.csv",
        output_path="data/processed/processed_data.csv"
    )