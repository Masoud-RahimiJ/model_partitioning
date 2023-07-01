from transformers import AutoTokenizer, AutoConfig, TFGPT2LMHeadModel, pipeline, set_seed
import boto3
from botocore.client import Config
import time


BUCKET="dnn-models"
OBJECT_NAME="gtp-xl.h5"
s3 = boto3.resource('s3', endpoint_url='http://10.10.1.2:9000',aws_access_key_id='masoud', aws_secret_access_key='ramzminio', config=Config(signature_version='s3v4'),)
bucket = s3.Bucket("dnn-models")

start = time.time()
bucket.download_file(Filename=OBJECT_NAME, Key=OBJECT_NAME)
print(time.time()-start)

start = time.time()
set_seed(42)
tokenizer = AutoTokenizer.from_pretrained('gpt2-xl')
config=AutoConfig.from_pretrained('gpt2-xl')
model = TFGPT2LMHeadModel(config)
print(time.time()-start)

text = "Replace me by any text you'd like."
start = time.time()
generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
print(time.time()-start)
generator("Hello, I'm a language model,", max_length=30, num_return_sequences=1)

start = time.time()
model.load_weights(OBJECT_NAME)
print(time.time()-start)

start = time.time()
output = generator("Hello, I'm a language model,", max_length=30, num_return_sequences=1)
print(time.time()-start)

print(output)