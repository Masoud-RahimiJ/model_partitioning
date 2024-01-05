import time
start_time = time.time()
from tensorflow.keras.applications.resnet import decode_predictions, preprocess_input
from tensorflow.keras.applications import ResNet101
import tensorflow as tf
from utils.image_loader_tf import image
import boto3
from botocore.client import Config
from lib.tf_model_loader import TFModelLoader

BUCKET="dnn-models"
OBJECT_NAME="resnet"
COUNT_PARTITIONS=14


s3 = boto3.resource('s3', endpoint_url='http://10.10.1.2:9000',aws_access_key_id='admin', aws_secret_access_key='ramzminio', config=Config(signature_version='s3v4'),)
bucket = s3.Bucket("dnn-models")

def init_model():
    return ResNet101(include_top=True, weights=None, input_tensor=None, input_shape=None, pooling=None, classes=1000, classifier_activation="softmax", )

config = {"download_delay": 8000000,
          "partition_names": [f"{OBJECT_NAME}_{i}.h5" for i in range(1, COUNT_PARTITIONS+1)]}

model = TFModelLoader(init_model, bucket, config).load()


image = preprocess_input(image)

time.sleep(5)
preds = model.predict(image)
print('Predicted:', decode_predictions(preds, top=5)[0])
print("Response time is: ", time.time() - start_time)
