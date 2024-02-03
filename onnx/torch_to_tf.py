import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.onnx as torch_onnx
from transformers import AutoTokenizer, AutoConfig, GPT2LMHeadModel, pipeline, set_seed
import boto3
from botocore.client import Config


BUCKET="dnn-models"
OBJECT_NAME="gtp-xl.pt"
s3 = boto3.resource('s3', endpoint_url='http://130.127.134.75:9000',aws_access_key_id='masoud', aws_secret_access_key='ramzminio', config=Config(signature_version='s3v4'),)
bucket = s3.Bucket("dnn-models")

bucket.download_file(Filename=OBJECT_NAME, Key=OBJECT_NAME)

set_seed(42)
tokenizer = AutoTokenizer.from_pretrained('gpt2-xl')
config=AutoConfig.from_pretrained('gpt2-xl')
model = GPT2LMHeadModel(config)
model.eval()

state_dict = torch.load(OBJECT_NAME)
model.load_state_dict(state_dict)



# Export the model to an ONNX file
inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
output = torch_onnx.export(model, inputs["input_ids"], "gpt-xl.onnx", verbose=False,  opset_version=7)




