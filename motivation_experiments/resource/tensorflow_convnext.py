from time import sleep
from tensorflow.keras.applications.convnext import decode_predictions, preprocess_input, ConvNeXtXLarge
from utils.image_loader_tf import image
import boto3
from botocore.client import Config

BUCKET="dnn-models"
OBJECT_NAME="convnext_xlarge.h5"
s3 = boto3.resource('s3', endpoint_url='http://10.10.1.2:9000',aws_access_key_id='masoud', aws_secret_access_key='ramzminio', config=Config(signature_version='s3v4'),)
bucket = s3.Bucket("dnn-models")

model = ConvNeXtXLarge( model_name="convnext_xlarge", include_top=True, include_preprocessing=True, weights=None, input_tensor=None, input_shape=None, pooling=None, classes=1000, classifier_activation="softmax", )
bucket.download_file(Filename=OBJECT_NAME, Key=OBJECT_NAME)
model.load_weights(OBJECT_NAME)

image = preprocess_input(image)

preds = model.predict(image)
print('Predicted:', decode_predictions(preds, top=5)[0])
sleep(5)