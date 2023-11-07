import time
import torch
import torchvision
from utils.image_loader import image
import io
import boto3
from botocore.client import Config

BUCKET="dnn-models"
OBJECT_NAME="vit"



device = torch.device("cpu")

model = torchvision.models.vit_l_16(weights=None).to(device)

# time.sleep(3)

model_state_dict = torch.load(OBJECT_NAME)
model.load_state_dict(model_state_dict)
del model_state_dict
model.eval()
output = model.forward(image)
probabilities = torch.nn.functional.softmax(output[0], dim=0)
top5_prob, top5_catid = torch.topk(probabilities, 5)
with open("./utils/imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]
    for i in range(top5_prob.size(0)):
        print(categories[top5_catid[i]], top5_prob[i].item())