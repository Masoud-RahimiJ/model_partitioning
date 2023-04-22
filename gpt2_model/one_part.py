import time
import torch
import io
import boto3
from botocore.client import Config
from transformers import GPT2Config, GPT2Model, GPT2Tokenizer




BUCKET="dnn-models"
OBJECT_NAME="gpt2"


s3 = boto3.resource('s3', endpoint_url='http://130.127.134.73:9000',aws_access_key_id='masoud', aws_secret_access_key='ramzminio', config=Config(signature_version='s3v4'),)
bucket = s3.Bucket("dnn-models")
device = torch.device("cpu")

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
configuration = GPT2Config()
model = GPT2Model(configuration).to(device)
text = "The White man worked as a"

start_time =time.time()
model_state_dict = torch.load(io.BytesIO(bucket.Object(OBJECT_NAME).get()['Body'].read()))
print(time.time()-start_time)
model.load_state_dict(model_state_dict)
print(time.time()-start_time)
model.eval()
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
end_time = time.time()
print(end_time-start_time)
print(outputs)