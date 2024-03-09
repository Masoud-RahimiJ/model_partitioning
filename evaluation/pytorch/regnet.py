import time, io
start_time = time.time()
import torch
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
import torchvision
import os, io
import boto3
from botocore.client import Config
from lib.torch_model_loader import TorchModelLoader
from utils.image_loader import image

BUCKET="dnn-models"
OBJECT_NAME="regnet"
COUNT_PARTITIONS=22
MT = os.getenv("MT", "F")


s3 = boto3.resource('s3', endpoint_url='http://127.0.0.1:9000',aws_access_key_id='admin', aws_secret_access_key='ramzminio', config=Config(signature_version='s3v4'),)
bucket = s3.Bucket("dnn-models")
device = torch.device("cuda")

def init_model():
    # with init_empty_weights():
    return torchvision.models.regnet_y_128gf(weights=None).to(device)

config = {"download_delay": 8000000,
          "partition_names": [f"{OBJECT_NAME}_{i}" for i in range(1, COUNT_PARTITIONS+1)]}

if MT == "T":
    model = TorchModelLoader(init_model, bucket, config).load()
else:
    model = init_model()
    std = torch.load(io.BytesIO(bucket.Object(OBJECT_NAME).get()['Body'].read()))
    model.load_state_dict(std)
    del std

model.eval()

image = image.to(device)
output = model.forward(image)
probabilities = torch.nn.functional.softmax(output[0], dim=0)
top5_prob, top5_catid = torch.topk(probabilities, 5)
with open("./utils/imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]
    for i in range(top5_prob.size(0)):
        print(categories[top5_catid[i]], top5_prob[i].item())
print("Response time is: ", time.time() - start_time)
