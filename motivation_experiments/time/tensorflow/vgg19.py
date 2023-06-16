times = []
import time
times.append(time.time())
from tensorflow.keras.applications.vgg19 import decode_predictions, preprocess_input, VGG19
from utils.image_loader_tf import image
import boto3
from botocore.client import Config
times.append(time.time())


BUCKET="dnn-models"
OBJECT_NAME="vgg19_weights_tf_dim_ordering_tf_kernels.h5"
s3 = boto3.resource('s3', endpoint_url='http://10.10.1.1:9000',aws_access_key_id='masoud', aws_secret_access_key='ramzminio', config=Config(signature_version='s3v4'),)
bucket = s3.Bucket("dnn-models")
times.append(time.time())
model = VGG19(include_top=True, weights=None, input_tensor=None, input_shape=None, pooling=None, classes=1000, classifier_activation="softmax", )
times.append(time.time())
bucket.download_file(Filename=OBJECT_NAME, Key=OBJECT_NAME)
times.append(time.time())
model.load_weights(OBJECT_NAME)
times.append(time.time())

image = preprocess_input(image)
times.append(time.time())

preds = model.predict(image)

print('Predicted:', decode_predictions(preds, top=5)[0])
times.append(time.time())
for i in range(1, len(times)):
    print(times[i] - times[i - 1])