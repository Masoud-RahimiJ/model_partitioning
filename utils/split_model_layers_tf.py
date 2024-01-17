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
partition.attrs['backend'] = 'tensorflow'
partition.attrs['keras_version'] = '2.15.0'
partitions_count=0


def extract_layer_names(weight_names):
    result = dict()
    for w in weight_names:
        layer_name = '/'.join(w.split('/')[:-1])
        if not layer_name in result:
            result[layer_name] = []
        result[layer_name].append(w)
    return result


for k, name in enumerate(layer_names):
    file.copy(file[name], partition)
    if os.stat("partition.h5").st_size / 1024 >= MIN_LAYER_SIZE:
        partition.attrs["layer_names"] = list(partition.keys())
        obj_name = get_partition_obj_name(partitions_count)
        partition.close()
        bucket.upload_file(Filename="partition.h5", Key=obj_name)
        os.remove("partition.h5")
        partitions_count += 1 
        partition = h5py.File("partition.h5", 'w')
    # partition.create_group(name)
    # wn = []
    # partition[name].attrs["weight_names"] = wn
    # g = file[name]
    # weight_names = load_attributes_from_hdf5_group(g, 'weight_names')
    # layers = extract_layer_names(weight_names)
    # for i, layer in enumerate(layers.keys()):
    #     print(name,layer)
    #     if partition[name].get('/'.join(layer.split('/')[:-1]), None) is None:
    #         partition[name].create_group('/'.join(layer.split('/')[:-1])) 
    #     file.copy(file[name][layer], partition[name]['/'.join(layer.split('/')[:-1])])
    #     for w in layers[layer]:
    #         wn.append(w)
    #         partition[name].attrs["weight_names"] = wn
    #     if os.stat("partition.h5").st_size / 1024 >= MIN_LAYER_SIZE:
    #         partition.attrs["layer_names"] = list(partition.keys())
    #         obj_name = get_partition_obj_name(partitions_count)
    #         partition.close()
    #         bucket.upload_file(Filename="partition.h5", Key=obj_name)
    #         os.remove("partition.h5")
    #         wn = []
    #         partitions_count += 1
    #         partition = h5py.File("partition.h5", 'w')
    #         partition.attrs['backend'] = 'tensorflow'
    #         partition.attrs['keras_version'] = '2.15.0'
    #         if i + 1 != len(layers.keys()):
    #             partition.create_group(name)

if len(partition.keys()) > 0:
    partition.attrs["layer_names"] = list(partition.keys())
    partition.close()
    obj_name = get_partition_obj_name(partitions_count)
    bucket.upload_file(Filename="partition.h5", Key=obj_name)
    partitions_count += 1
        
print(partitions_count)
os.remove("partition.h5")
