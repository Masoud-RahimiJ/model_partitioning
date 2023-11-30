import time
start_time = time.time()
import torch
from accelerate import init_empty_weights
import torchvision
import io
import boto3
from botocore.client import Config
from lib.torch_model_loader import TorchModelLoader
from utils.image_loader import image

BUCKET="dnn-models"
OBJECT_NAME="regnet_y_128gf_swag-c8ce3e52"
COUNT_PARTITIONS=50

s3 = boto3.resource('s3', endpoint_url='http://10.10.1.2:9000',aws_access_key_id='masoud', aws_secret_access_key='ramzminio', config=Config(signature_version='s3v4'),)
bucket = s3.Bucket("dnn-models")
device = torch.device("cpu")

def init_model():
    with init_empty_weights():
        return torchvision.models.regnet_y_128gf(weights=None)

config = {"download_delay": 6000000,
          "partition_names": [f"{OBJECT_NAME}{i}.pth" for i in range(1, COUNT_PARTITIONS)]}

model = TorchModelLoader(init_model, bucket, config).load()
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
