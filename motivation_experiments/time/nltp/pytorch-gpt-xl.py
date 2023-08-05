import time
start = time.time()
from transformers import AutoTokenizer, AutoConfig, GPT2LMHeadModel, pipeline, set_seed
import boto3
from botocore.client import Config
import time
import torch
print(time.time()-start)


BUCKET="dnn-models"
OBJECT_NAME="../models/gpt2-xl.pt"
device = torch.device("cuda")
s3 = boto3.resource('s3', endpoint_url='http://10.10.1.2:9000',aws_access_key_id='masoud', aws_secret_access_key='ramzminio', config=Config(signature_version='s3v4'),)
bucket = s3.Bucket("dnn-models")

start = time.time()
# bucket.download_file(Filename=OBJECT_NAME, Key=OBJECT_NAME)
# print(time.time()-start)

set_seed(20)
tokenizer = AutoTokenizer.from_pretrained('gpt2-xl')
config=AutoConfig.from_pretrained('gpt2-xl')
start = time.time()
model = GPT2LMHeadModel(config)
print(time.time()-start)
model.eval()

start = time.time()
state_dict = torch.load(OBJECT_NAME)
model.load_state_dict(state_dict)
device = torch.device("cuda")
# model.to(device)
print(time.time()-start)

start = time.time()
generator = pipeline('text-generation', model=model, tokenizer=tokenizer, device=0)
print(time.time()-start)

start = time.time()
output = generator("Hello, I'm a language model and I can", max_length=30, num_return_sequences=1)
print(time.time()-start)

print(output)