times = []
import time
times.append(time.time())
import mxnet as mx
from mxnet.gluon.model_zoo import vision
import boto3
from botocore.client import Config
times.append(time.time())


BUCKET="dnn-models"
OBJECT_NAME="resnet152_v2-f2695542.params"
s3 = boto3.resource('s3', endpoint_url='http://10.10.1.2:9000',aws_access_key_id='masoud', aws_secret_access_key='ramzminio', config=Config(signature_version='s3v4'),)
bucket = s3.Bucket("dnn-models")
times.append(time.time())


ctx = mx.cpu()
model = vision.resnet152_v2(pretrained=False, ctx=ctx)
times.append(time.time())
bucket.download_file(Filename=OBJECT_NAME, Key=OBJECT_NAME)
times.append(time.time())
model.load_parameters(OBJECT_NAME)
times.append(time.time())
from utils.image_loader_mx import image
times.append(time.time())


predictions = model(image).softmax()
top_pred = predictions.topk(k=5)[0].asnumpy()
with open("./utils/imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]
    for index in top_pred:
        probability = predictions[0][int(index)]
        category = categories[int(index)]
        print("{}: {:.2f}%".format(category, probability.asscalar()*100))
times.append(time.time())
for i in range(1, len(times)):
    print(times[i] - times[i - 1])
