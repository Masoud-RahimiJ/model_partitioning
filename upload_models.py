from torch import load, save
import os, io
import boto3
from botocore.client import Config
import torchvision
from transformers import TFRegNetForImageClassification, TFGPT2LMHeadModel, TFBertLMHeadModel, GPT2LMHeadModel, BertLMHeadModel, Wav2Vec2ForCTC, WhisperForConditionalGeneration
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.applications.vgg19 import VGG19

BUCKET="dnn-models"
s3 = boto3.resource('s3', endpoint_url='http://128.110.219.188:9000',aws_access_key_id='admin', aws_secret_access_key='ramzminio', config=Config(signature_version='s3v4'),)
bucket = s3.Bucket(BUCKET)

def upload_model(model,name):
    buffer = io.BytesIO()
    save(model.state_dict(), buffer)
    buffer=buffer.getvalue()
    bucket.put_object(Key=name, Body=buffer)

model = torchvision.models.resnet101(pretrained=True)
upload_model(model, "resnet")
model = torchvision.models.vgg19(pretrained=True)
upload_model(model, "vgg19")
# model = torchvision.models.regnet_y_128gf(weights=torchvision.models.R
# upload_model(model, "regnet")
model = GPT2LMHeadModel.from_pretrained("gpt2")
upload_model(model, "gpt2")
model = BertLMHeadModel.from_pretrained("setu4993/LaBSE")
upload_model(model, "labse")
model = GPT2LMHeadModel.from_pretrained("gpt2-xl")
upload_model(model, "gpt2-xl")
model = Wav2Vec2ForCTC.from_pretrained('facebook/wav2vec2-base-960h')
upload_model(model, "wav")
model = WhisperForConditionalGeneration.from_pretrained('openai/whisper-medium')
upload_model(model, "whisper-md")
model = WhisperForConditionalGeneration.from_pretrained('openai/whisper-large-v2')
upload_model(model, "whisper-large")


def upload_model2(model,name):
    model.save_weights(name+".h5")
    bucket.upload_file(Key=name+".h5", Filename=name+".h5")

model = ResNet101(include_top=True, weights="imagenet", input_tensor=None, input_shape=None, pooling=None, classes=1000, classifier_activation="softmax", )
upload_model2(model, "resnet")
model = VGG19(include_top=True, weights="imagenet", input_tensor=None, input_shape=None, pooling=None, classes=1000, classifier_activation="softmax", )
upload_model2(model, "vgg19")
model = TFRegNetForImageClassification.from_pretrained("facebook/regnet-y-1280-seer-in1k")
upload_model2(model, "regnet")
model = TFGPT2LMHeadModel.from_pretrained("gpt2")
upload_model2(model, "gpt2")
model = TFBertLMHeadModel.from_pretrained('setu4993/LaBSE')
upload_model2(model, "labse")
model = TFGPT2LMHeadModel.from_pretrained("gpt2-xl")
upload_model2(model, "gpt2-xl")


["resnet" "vgg19" "regnet" "gpt2" "gpt2-xl" "labse" "wav" "whisper-md" "whisper-large"]