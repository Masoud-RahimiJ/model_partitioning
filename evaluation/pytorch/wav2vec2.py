import time
start_time = time.time()
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, AutoFeatureExtractor, AutoConfig, pipeline, set_seed
import boto3
from botocore.client import Config
from lib.torch_model_loader import TorchModelLoader
import time
import torch, os
from accelerate import init_empty_weights


BUCKET="dnn-models"
OBJECT_NAME="wav"
COUNT_PARTITIONS=8

s3 = boto3.resource('s3', endpoint_url='http://10.10.1.2:9000',aws_access_key_id='admin', aws_secret_access_key='ramzminio', config=Config(signature_version='s3v4'),)
bucket = s3.Bucket("dnn-models")


set_seed(42)
processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h')
feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")

def init_model():
    # with init_empty_weights():
    config = AutoConfig.from_pretrained('facebook/wav2vec2-base-960h')
    return Wav2Vec2ForCTC(config)


config = {"download_delay": 6000000,
          "partition_names": [f"{OBJECT_NAME}_{i}" for i in range(1, COUNT_PARTITIONS+1)]}

model = TorchModelLoader(init_model, bucket, config).load()

# model=init_model()
# bucket.download_file(Key = OBJECT_NAME, Filename = OBJECT_NAME)
# std = torch.load(OBJECT_NAME)
# model.load_state_dict(std)
# del std
# os.remove(OBJECT_NAME)

model.eval()

generator = pipeline('automatic-speech-recognition', model=model, tokenizer=processor, feature_extractor=feature_extractor)

inp = []
for i in range(1, int(os.getenv('BS', 1))+1):
    inp .append(f"sample{i}.flac")

output = generator(inp)

print(output)
print("Response time is: ", time.time() - start_time)
