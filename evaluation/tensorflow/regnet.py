import time
start_time = time.time()
from transformers import AutoConfig, AutoFeatureExtractor, TFRegNetForImageClassification
from utils.image_loader_tf import image
import boto3
from botocore.client import Config
import tensorflow as tf
print(time.time() - start_time)
# from lib.tf_model_loader import TFModelLoader

BUCKET="dnn-models"
OBJECT_NAME="../models/regnet.h5"
COUNT_PARTITIONS=20

s3 = boto3.resource('s3', endpoint_url='http://10.10.1.2:9000',aws_access_key_id='masoud', aws_secret_access_key='ramzminio', config=Config(signature_version='s3v4'),)
bucket = s3.Bucket("dnn-models")

start_time = time.time()

feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/regnet-y-040")
config = AutoConfig.from_pretrained("facebook/regnet-y-040")

# def init_model():
#     return RegNetForImageClassification(config)

# config = {"download_delay": 6000000,
#           "partition_names": [f"{OBJECT_NAME}{i}.h5" for i in range(1, COUNT_PARTITIONS)]}

# model = TFModelLoader(init_model, bucket, config).load()
model=TFRegNetForImageClassification(config)
print(time.time() - start_time)

start_time = time.time()

image = feature_extractor(image, return_tensors="tf")

logits = model(**image).logits
predicted_label = int(tf.math.argmax(logits, axis=-1))
print(time.time() - start_time)

print('Predicted:', model.config.id2label[predicted_label])
print("Response time is: ", time.time() - start_time)
