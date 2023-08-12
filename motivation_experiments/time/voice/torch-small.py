import time
start = time.time()
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, AutoFeatureExtractor, AutoConfig, pipeline, set_seed
import boto3
from botocore.client import Config
import time
import torch
from accelerate import init_empty_weights
print(time.time()-start)


# BUCKET="dnn-models"
# OBJECT_NAME="../models/wav.pt"
# s3 = boto3.resource('s3', endpoint_url='http://10.10.1.2:9000',aws_access_key_id='masoud', aws_secret_access_key='ramzminio', config=Config(signature_version='s3v4'),)
# bucket = s3.Bucket("dnn-models")

# start = time.time()
# bucket.download_file(Filename=OBJECT_NAME, Key=OBJECT_NAME)
# print(time.time()-start)

start = time.time()
set_seed(42)
processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h')
feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")
config = AutoConfig.from_pretrained('facebook/wav2vec2-base-960h')
with init_empty_weights():
    model = Wav2Vec2ForCTC(config)
model.eval()
print(time.time()-start)

# start = time.time()
# state_dict = torch.load(OBJECT_NAME)
# model.load_state_dict(state_dict, strict=False)
# model.tie_weights()
# print(time.time()-start)

# start = time.time()
# generator = pipeline('automatic-speech-recognition', model=model, tokenizer=processor, feature_extractor=feature_extractor, device=torch.device("cuda"))
# print(time.time()-start)

# start = time.time()
# output = generator("sample2.flac")
# print(time.time()-start)

# print(output)