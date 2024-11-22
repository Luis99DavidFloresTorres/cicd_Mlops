import boto3
import os

def upload_file_to_s3(local_file_path, bucket_name, s3_file_name):
    # Inicializa el cliente de S3
    s3 = boto3.client('s3')

    try:
        s3.upload_file(local_file_path, bucket_name, s3_file_name)
        print(f"Archivo {local_file_path} subido exitosamente a {bucket_name}/{s3_file_name}")
    except Exception as e:
        print(f"Error al subir el archivo: {e}")

if __name__ == "__main__":
    local_file_path = "data/dataset.csv"  # Ruta local del archivo
    bucket_name = "mlopsluis"       # Nombre del bucket S3
    s3_file_name = "mlopsluis/dataset.csv"         # Ruta en el bucket S3

    upload_file_to_s3(local_file_path, bucket_name, s3_file_name)
