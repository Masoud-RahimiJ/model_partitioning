import os, io
import boto3
from botocore.client import Config
from keras.models import load_model
from tensorflow.python.keras.saving import hdf5_format
import h5py


BUCKET="dnn-models"
OBJECT_NAME=os.getenv("OBJECT_NAME")
MIN_LAYER_SIZE = int(os.getenv("MIN_LAYER_SIZE", 20000))

s3 = boto3.resource('s3', endpoint_url='http://130.127.134.73:9000',aws_access_key_id='masoud', aws_secret_access_key='ramzminio', config=Config(signature_version='s3v4'),)
bucket = s3.Bucket(BUCKET)


def get_partition_obj_name(part):
    return OBJECT_NAME + '_' + str(part+1)

model = load_model(io.BytesIO(bucket.Object(OBJECT_NAME).get()['Body'].read()))
layers = model.layers()

partition = []
partitions_count = 0

for i in range(len(layers)):
    partition.append(layers[i])
    with h5py.File("partition.h5", 'w') as f:
        hdf5_format.save_weights_to_hdf5_group(f, partition)
    if os.stat("partition.h5").st_size / 1024 >= MIN_LAYER_SIZE or i+1==len(layers):
        obj_name = get_partition_obj_name(partitions_count)
        bucket.upload_file(Filename="partition.h5", Key=obj_name)
        partition = []
        partitions_count += 1

print(partitions_count)