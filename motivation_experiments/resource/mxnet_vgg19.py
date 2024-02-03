import os
from time import sleep
import mxnet as mx
from mxnet.gluon.model_zoo import vision
from utils.image_loader_mx import image
import boto3
from botocore.client import Config

BUCKET="dnn-models"
OBJECT_NAME="vgg19-ad2f660d.params"
s3 = boto3.resource('s3', endpoint_url='http://130.127.134.75:9000',aws_access_key_id='masoud', aws_secret_access_key='ramzminio', config=Config(signature_version='s3v4'),)
bucket = s3.Bucket("dnn-models")



ctx = mx.cpu()
model = vision.vgg19(pretrained=False, ctx=ctx)
bucket.download_file(Filename=OBJECT_NAME, Key=OBJECT_NAME)
sleep(10)
model.load_parameters(OBJECT_NAME)
sleep(10)
model.load_parameters(OBJECT_NAME)
sleep(10)
model.load_parameters(OBJECT_NAME)
sleep(10)
model.load_parameters(OBJECT_NAME)
sleep(10)


predictions = model(image).softmax()
top_pred = predictions.topk(k=5)[0].asnumpy()
with open("./utils/imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]
    for index in top_pred:
        probability = predictions[0][int(index)]
        category = categories[int(index)]
        print("{}: {:.2f}%".format(category, probability.asscalar()*100))
        
sleep(2)

