
        
import time
import torch
import io
import os
import boto3
from botocore.client import Config
from lib.torch_model_loader import wrap_module
from concurrent import futures
from transformers import AutoConfig, AutoModel, GPT2Tokenizer
from threading import Lock


BUCKET="dnn-models"
OBJECT_NAME="gpt2"
LAYER_COUNT = 11
COUNT_THREADS = int(os.getenv("COUNT_THREADS",3))

s3 = boto3.resource('s3', endpoint_url='http://128.105.144.221:9000',aws_access_key_id='masoud', aws_secret_access_key='ramzminio', config=Config(signature_version='s3v4'),)
bucket = s3.Bucket("dnn-models")
device = torch.device("cpu")

download_lock = Lock()
loading_lock = Lock()

def get_layer_file_name(part):
    return OBJECT_NAME + '_' + str(part+1)

def load_model(i):
    file_name = get_layer_file_name(i)
    layer_download_connection = bucket.Object(file_name).get()['Body']
    download_lock.acquire()
    layer_bin = io.BytesIO(layer_download_connection.read())
    download_lock.release()
    loading_lock.acquire()
    layer = torch.load(layer_bin)
    model.load_state_dict(layer, strict=False)
    loading_lock.release()

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
configuration = AutoConfig.from_pretrained("gpt2")
model = AutoModel.from_config(configuration).to(device)
text = "The White man worked as a"

start_time =time.time()
wrap_module(model)
model.eval()
executor = futures.ThreadPoolExecutor(max_workers=COUNT_THREADS)
{executor.submit(load_model, i): i for i in range(LAYER_COUNT)}
inputs = tokenizer.encode(text, return_tensors="pt")
outputs = model(inputs)
end_time = time.time()
print(end_time-start_time)