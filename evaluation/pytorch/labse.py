import time
start_time = time.time()
from transformers import AutoConfig, BertModel, BertTokenizerFast, pipeline, set_seed
import boto3
from botocore.client import Config
import time
import torch
from accelerate import init_empty_weights
from lib.torch_model_loader import TorchModelLoader
from accelerate import init_empty_weightsModelLoader


BUCKET="dnn-models"
OBJECT_NAME="labse"
COUNT_PARTITIONS=30

s3 = boto3.resource('s3', endpoint_url='http://10.10.1.2:9000',aws_access_key_id='masoud', aws_secret_access_key='ramzminio', config=Config(signature_version='s3v4'),)
bucket = s3.Bucket("dnn-models")

set_seed(42)
tokenizer = BertTokenizerFast.from_pretrained('setu4993/LaBSE')
config=AutoConfig.from_pretrained('setu4993/LaBSE')

def init_model():
    with init_empty_weights():
        return BertModel(config)

config = {"download_delay": 6000000,
          "partition_names": [f"{OBJECT_NAME}{i}.pt" for i in range(1, COUNT_PARTITIONS)]}

model = TorchModelLoader(init_model, bucket, config).load()
model.eval()


text = "Replace me by any text you'd like."
generator = pipeline('text-generation', model=model, tokenizer=tokenizer)

output = generator("Hello, I'm a language model,", max_length=30, num_return_sequences=1)

print(output)
print("Response time is: ", time.time() - start_time)
