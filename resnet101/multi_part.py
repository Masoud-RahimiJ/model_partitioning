import time
import torch
import torchvision
from utils.image_loader import image
import io
import boto3
from botocore.client import Config
import os
from lib.torch_model_loader import wrap_module
from concurrent import futures


BUCKET="dnn-models"
OBJECT_NAME="resnet101-63fe2227"
LAYER_COUNT = 9
COUNT_THREADS = int(os.getenv("COUNT_THREADS",2))


device = torch.device("cpu")

s3 = boto3.resource('s3', endpoint_url='http://10.10.1.2:9000',aws_access_key_id='masoud', aws_secret_access_key='ramzminio', config=Config(signature_version='s3v4'),)
bucket = s3.Bucket("dnn-models")

model = torchvision.models.resnet101(weights=None).to(device)


def get_layer_file_name(part):
    return OBJECT_NAME + '_' + str(part+1)


def load_model(i):
    file_name = get_layer_file_name(i)
    layer = torch.load(io.BytesIO(bucket.Object(file_name).get()['Body'].read()))
    model.load_state_dict(layer, strict=False)


start_time =time.time()
wrap_module(model)
model.eval()
executor = futures.ThreadPoolExecutor(max_workers=COUNT_THREADS)
{executor.submit(load_model, i): i for i in range(LAYER_COUNT)}
output = model.forward(image)
end_time =time.time()
print(end_time-start_time)
probabilities = torch.nn.functional.softmax(output[0], dim=0)
top5_prob, top5_catid = torch.topk(probabilities, 5)
with open("./utils/imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]
    for i in range(top5_prob.size(0)):
        print(categories[top5_catid[i]], top5_prob[i].item())
        