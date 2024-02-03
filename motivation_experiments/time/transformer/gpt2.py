times = []
import time
times.append(time.time())
from transformers import AutoTokenizer, GPT2LMHeadModel, AutoConfig, pipeline, set_seed
import torch
import boto3
from botocore.client import Config
import io
times.append(time.time())


BUCKET="dnn-models"
OBJECT_NAME="gpt2-large-torch.pth"


s3 = boto3.resource('s3', endpoint_url='http://130.127.134.75:9000',aws_access_key_id='masoud', aws_secret_access_key='ramzminio', config=Config(signature_version='s3v4'),)
bucket = s3.Bucket("dnn-models")
device = torch.device("cpu")
times.append(time.time())


text = "The White man worked as a"

tokenizer = AutoTokenizer.from_pretrained("gpt2-large")
configuration = AutoConfig.from_pretrained("gpt2-large")
model = GPT2LMHeadModel(configuration).to(device)
times.append(time.time())


model_bin = io.BytesIO(bucket.Object(OBJECT_NAME).get()['Body'].read())
times.append(time.time())
model_state_dict = torch.load(model_bin)
times.append(time.time())
model.load_state_dict(model_state_dict, strict=False)
times.append(time.time())
model.tie_weights()
times.append(time.time())

generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
times.append(time.time())

set_seed(42)
output = generator("Hello, I'm a language model,", max_length=30, num_return_sequences=5)
print(output)
times.append(time.time())

for i in range(1, len(times)):
    print(times[i] - times[i - 1])