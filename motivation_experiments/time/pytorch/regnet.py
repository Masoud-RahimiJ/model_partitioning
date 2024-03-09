times = []
import time
times.append(time.time())
import torch
import torchvision
import io
import boto3
from botocore.client import Config
times.append(time.time())

BUCKET="dnn-models"
OBJECT_NAME="../models/regnet.pt"


s3 = boto3.resource('s3', endpoint_url='http://127.0.0.1:9000',aws_access_key_id='masoud', aws_secret_access_key='ramzminio', config=Config(signature_version='s3v4'),)
bucket = s3.Bucket("dnn-models")
device = torch.device("cuda")
times.append(time.time())

model = torchvision.models.regnet_y_128gf(weights=None)
times.append(time.time())

model_bin = io.BytesIO(bucket.Object(OBJECT_NAME).get()['Body'].read())
times.append(time.time())
model_state_dict = torch.load(model_bin)
model.to(device)
times.append(time.time())
model.load_state_dict(model_state_dict)
model.eval()
times.append(time.time())
from utils.image_loader import image
image
times.append(time.time())
output = model.forward(image)
probabilities = torch.nn.functional.softmax(output[0], dim=0)
top5_prob, top5_catid = torch.topk(probabilities, 5)
with open("./utils/imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]
    for i in range(top5_prob.size(0)):
        print(categories[top5_catid[i]], top5_prob[i].item())
times.append(time.time())
for i in range(1, len(times)):
    print(times[i] - times[i - 1])
