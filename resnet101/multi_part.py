import time
import torch
import torchvision
from utils.image_loader import image
import io
import boto3
from botocore.client import Config
import os
from lib.lib import wrap_model
from concurrent import futures
from threading import Lock
from collections import OrderedDict



BUCKET="dnn-models"
OBJECT_NAME="resnet101-63fe2227"
LAYER_COUNT = 9
COUNT_THREADS = int(os.getenv("COUNT_THREADS",2))


# device = torch.device("cpu")

s3 = boto3.resource('s3', endpoint_url='http://10.10.1.2:9000',aws_access_key_id='masoud', aws_secret_access_key='minioadmin', config=Config(signature_version='s3v4'),)
bucket = s3.Bucket("dnn-models")

model = torchvision.models.resnet101(weights=None)

loading_lock = Lock()

model_dict = OrderedDict()


def get_layer_file_name(part):
    return OBJECT_NAME + '_' + str(part+1)



def load_model(i):
    print(i)
    file_name = get_layer_file_name(i)
    layer = torch.load( io.BytesIO(bucket.Object(file_name).get()['Body'].read()))
    print(layer._metadata)
    for k,v in layer.items():
        model_dict[k]=v
    if i==LAYER_COUNT-1:
        model.load_state_dict(model_dict, strict=False)
    model.eval()
    print(i)


start_time =time.time()
wrap_model(model)
executor = futures.ThreadPoolExecutor(max_workers=COUNT_THREADS)
{executor.submit(load_model, i): i for i in range(LAYER_COUNT)}
output = model.forward(image)
output2 = model.forward(image)
output3 = model.forward(image)
end_time =time.time()
# print(end_time-start_time)
probabilities = torch.nn.functional.softmax(output[0], dim=0)
top5_prob, top5_catid = torch.topk(probabilities, 5)
with open("./utils/imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]
    print(categories[top5_catid[0]], top5_prob[0].item())
probabilities = torch.nn.functional.softmax(output2[0], dim=0)
top5_prob, top5_catid = torch.topk(probabilities, 5)
with open("./utils/imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]
    print(categories[top5_catid[0]], top5_prob[0].item())
probabilities = torch.nn.functional.softmax(output3[0], dim=0)
top5_prob, top5_catid = torch.topk(probabilities, 5)
with open("./utils/imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]
    print(categories[top5_catid[0]], top5_prob[0].item())
    #     pass
    # if top5_prob[0].item() > 0.9510017634 or top5_prob[0].item() < 0.9510017632 :
    #     print("!!!!!!!!    ",top5_prob[0].item())