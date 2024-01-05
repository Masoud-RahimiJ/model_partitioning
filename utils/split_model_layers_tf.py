import os, io
import boto3
from botocore.client import Config
from keras.models import load_model
from tensorflow.python.keras.saving import hdf5_format
import h5py
from tensorflow.python.keras.saving.hdf5_format import load_attributes_from_hdf5_group
import numpy as np


BUCKET="dnn-models"
OBJECT_NAME=os.getenv("OBJECT_NAME")
MIN_LAYER_SIZE = int(os.getenv("MIN_LAYER_SIZE", 10000))

s3 = boto3.resource('s3', endpoint_url='http://10.10.1.2:9000',aws_access_key_id='admin', aws_secret_access_key='ramzminio', config=Config(signature_version='s3v4'),)
bucket = s3.Bucket(BUCKET)


def get_partition_obj_name(part):
    return OBJECT_NAME + '_' + str(part+1) + ".h5"

bucket.download_file(Filename=f"{OBJECT_NAME}.h5", Key=f"{OBJECT_NAME}.h5")
file = h5py.File(f"{OBJECT_NAME}.h5", "r")
layer_names = load_attributes_from_hdf5_group(file, 'layer_names')


partition = h5py.File("partition.h5", 'w')
start = True
partitions_count=0



for i, layer in enumerate(layer_names):
    print(i, layer)
    if not start:
        partition.attrs["layer_names"] = np.append(partition.attrs["layer_names"], layer.encode())
    else:
        partition.attrs["layer_names"] = [layer.encode()]
        start = False
    file.copy(file[layer], partition)
    if os.stat("partition.h5").st_size / 1024 >= MIN_LAYER_SIZE or i+1==len(layer_names):
        obj_name = get_partition_obj_name(partitions_count)
        partition.close()
        bucket.upload_file(Filename="partition.h5", Key=obj_name)
        os.remove("partition.h5")
        partition = h5py.File("partition.h5", 'w')
        start=True
        partitions_count += 1

print(partitions_count)