from torch import load, save
import os, io
import boto3
from botocore.client import Config
from collections import OrderedDict
from pympler import asizeof


BUCKET="dnn-models"
OBJECT_NAME=os.getenv("OBJECT_NAME")
MIN_LAYER_SIZE = 10*1000*1000*8

s3 = boto3.resource('s3', endpoint_url='http://10.10.1.2:9000',aws_access_key_id='masoud', aws_secret_access_key='minioadmin', config=Config(signature_version='s3v4'),)
bucket = s3.Bucket(BUCKET)

def extract_layer_name(layer):
    return '.'.join(layer.split('.')[0:-1])

def get_layer_file_name(part):
    return OBJECT_NAME + '_' + str(part+1)

model = load(io.BytesIO(bucket.Object(OBJECT_NAME).get()['Body'].read()))
splitted_model = []
previous_layer_name = ""

for key, value in  model.items():
#    print(key)
    layer_name = extract_layer_name(key)    
    print(asizeof.asizeof(splitted_model))
    if layer_name != previous_layer_name and (len(splitted_model) == 0 or asizeof.asizeof(splitted_model[-1]) > MIN_LAYER_SIZE):
        splitted_model.append(OrderedDict())
        splitted_model[-1]._metadata = getattr(model, "_metadata", None)
    splitted_model[-1][key] = value
    previous_layer_name = layer_name
    

for i in range(len(splitted_model)):
    buffer = io.BytesIO()
    save(splitted_model[i], buffer)
    buffer=buffer.getvalue()
    bucket.put_object(Key=get_layer_file_name(i), Body=buffer)
    
print(len(splitted_model))