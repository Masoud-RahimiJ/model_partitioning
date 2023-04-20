import time
import torch
import torchvision
from utils.image_loader import image
import io
import boto3
from botocore.client import Config

BUCKET="dnn-models"
OBJECT_NAME="regnet_y_128gf_swag-c8ce3e52"


s3 = boto3.resource('s3', endpoint_url='http://130.127.134.73:9000',aws_access_key_id='masoud', aws_secret_access_key='ramzminio', config=Config(signature_version='s3v4'),)
bucket = s3.Bucket("dnn-models")
device = torch.device("cpu")

model = torchvision.models.regnet_y_128gf(weights=None).to(device)

start_time =time.time()
model_bin = io.BytesIO(bucket.Object(OBJECT_NAME).get()['Body'].read())
print(time.time()-start_time)
model_state_dict = torch.load(model_bin)
print(time.time()-start_time)
model.load_state_dict(model_state_dict)
print(time.time()-start_time)
model.eval()
model.forward(image)
output = model.forward(image)
end_time = time.time()
print(end_time-start_time)
probabilities = torch.nn.functional.softmax(output[0], dim=0)
top5_prob, top5_catid = torch.topk(probabilities, 5)
with open("./utils/imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]
    for i in range(top5_prob.size(0)):
        print(categories[top5_catid[i]], top5_prob[i].item())