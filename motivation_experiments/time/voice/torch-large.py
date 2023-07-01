import time
start = time.time()
from transformers import WhisperForConditionalGeneration, WhisperProcessor, AutoFeatureExtractor, AutoConfig, pipeline, set_seed
import boto3
from botocore.client import Config
import time
import torch
print(time.time()-start)


BUCKET="dnn-models"
OBJECT_NAME="whisper-large.pt"
s3 = boto3.resource('s3', endpoint_url='http://10.10.1.2:9000',aws_access_key_id='masoud', aws_secret_access_key='ramzminio', config=Config(signature_version='s3v4'),)
bucket = s3.Bucket("dnn-models")

start = time.time()
bucket.download_file(Filename=OBJECT_NAME, Key=OBJECT_NAME)
print(time.time()-start)

start = time.time()
set_seed(42)
processor = WhisperProcessor.from_pretrained('openai/whisper-large-v2')
feature_extractor = AutoFeatureExtractor.from_pretrained("openai/whisper-large-v2")
config = AutoConfig.from_pretrained('openai/whisper-large-v2')
model = WhisperForConditionalGeneration(config)
model.config.forced_decoder_ids = None
model.eval()
print(time.time()-start)

start = time.time()
state_dict = torch.load(OBJECT_NAME)
model.load_state_dict(state_dict, strict=False)
model.tie_weights()
print(time.time()-start)

start = time.time()
generator = pipeline('automatic-speech-recognition', model=model, tokenizer=processor, feature_extractor=feature_extractor)
print(time.time()-start)

start = time.time()
output = generator("sample2.flac")
print(time.time()-start)

print(output)