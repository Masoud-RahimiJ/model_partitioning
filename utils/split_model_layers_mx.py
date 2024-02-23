import os, io
import boto3
from botocore.client import Config
from keras.models import load_model
from tensorflow.python.keras.saving import hdf5_format
import h5py
from mxnet import ndarray, numpy_extension
from mxnet.utils import is_np_array

BUCKET="dnn-models"
OBJECT_NAME=os.getenv("OBJECT_NAME")
MIN_LAYER_SIZE = int(os.getenv("MIN_LAYER_SIZE", 20000))

s3 = boto3.resource('s3', endpoint_url='http://128.105.144.221:9000',aws_access_key_id='masoud', aws_secret_access_key='ramzminio', config=Config(signature_version='s3v4'),)
bucket = s3.Bucket(BUCKET)

save_fn = numpy_extension.save if is_np_array() else ndarray.save
load_fn = numpy_extension.load if is_np_array() else ndarray.load

def get_partition_obj_name(part):
    return OBJECT_NAME + '_' + str(part+1)

bucket.download_file(Filename="model.params", Key=OBJECT_NAME)
model = load_fn("model.params")
layers = model._collect_params_with_prefix()

partition = []
partitions_count = 0

for key, val in layers.items():
    partition[key] = val._reduce()
    save_fn("partition.params", partition)
    if os.stat("partition.params").st_size / 1024 >= MIN_LAYER_SIZE:
        obj_name = get_partition_obj_name(partitions_count)
        bucket.upload_file(Filename="partition.params", Key=obj_name)
        partition = []
        partitions_count += 1
        
if len(partition) > 0:
    obj_name = get_partition_obj_name(partitions_count)
    bucket.upload_file(Filename="partition.params", Key=obj_name)
    partitions_count += 1

print(partitions_count)