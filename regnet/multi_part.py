import time
import torch
import torchvision
from utils.image_loader import image
import io
import boto3
from botocore.client import Config
import os
from lib.lib import wrap_module
from concurrent import futures
from threading import Lock

BUCKET="dnn-models"
OBJECT_NAME="regnet_y_128gf_swag-c8ce3e52"
LAYER_COUNT = 22
COUNT_THREADS = int(os.getenv("COUNT_THREADS",3))

download_lock = Lock()

device = torch.device("cpu")

s3 = boto3.resource('s3', endpoint_url='http://130.127.134.73:9000',aws_access_key_id='masoud', aws_secret_access_key='ramzminio', config=Config(signature_version='s3v4'),)
bucket = s3.Bucket("dnn-models")

model = torchvision.models.regnet_y_128gf(weights=None).to(device)


def get_layer_file_name(part):
    return OBJECT_NAME + '_' + str(part+1)


def load_model(i):
    file_name = get_layer_file_name(i)
    meta_data = s3.head_object(Bucket=BUCKET, Key=file_name)
    total_length = int(meta_data.get('ContentLength', 0))
    downloaded = 0
    def progress(chunk):
        nonlocal downloaded
        downloaded += chunk
        if downloaded/total_length > 0.9 and download_lock.locked():
            download_lock.release()
    download_lock.acquire()
    layer_bin = io.BytesIO()
    s3.download_fileobj(Bucket=BUCKET, Key=file_name, Fileobj=layer_bin, Callback=progress)
    layer = torch.load(layer_bin)
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
        