import time
import torch
import torchvision
from utils.image_loader import image
import io
import boto3
from botocore.client import Config

BUCKET="dnn-models"
OBJECT_NAME="regnet_y_128gf_swag-c8ce3e52"


s3 = boto3.resource('s3', endpoint_url='http://10.10.1.1:9000',aws_access_key_id='masoud', aws_secret_access_key='ramzminio', config=Config(signature_version='s3v4'),)
bucket = s3.Bucket("dnn-models")
device = torch.device("cpu")

model = torchvision.models.regnet_y_128gf(weights=None).to(device)

model_bin = io.BytesIO(bucket.Object(OBJECT_NAME).get()['Body'].read())
model_state_dict = torch.load(model_bin)
del model_bin
# model.load_state_dict(model_state_dict)
# del model_state_dict
# model.eval()
# output = model.forward(image)
# probabilities = torch.nn.functional.softmax(output[0], dim=0)
# top5_prob, top5_catid = torch.topk(probabilities, 5)
# with open("./utils/imagenet_classes.txt", "r") as f:
#     categories = [s.strip() for s in f.readlines()]
#     for i in range(top5_prob.size(0)):
#         print(categories[top5_catid[i]], top5_prob[i].item())