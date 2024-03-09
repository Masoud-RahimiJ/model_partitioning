import time, io
start_time = time.time()
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, AutoFeatureExtractor, AutoConfig, pipeline, set_seed
import boto3
from botocore.client import Config
from lib.torch_model_loader import TorchModelLoader
import time, io
import torch, os
from accelerate import init_empty_weights


BUCKET="dnn-models"
OBJECT_NAME="wav"
COUNT_PARTITIONS=8
MT = os.getenv("MT", "F")


s3 = boto3.resource('s3', endpoint_url='http://128.110.219.188:9000',aws_access_key_id='admin', aws_secret_access_key='ramzminio', config=Config(signature_version='s3v4'),)
bucket = s3.Bucket("dnn-models")
device = torch.device("cpu")


set_seed(42)
processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h')
feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")

def init_model():
    # with init_empty_weights():
    config = AutoConfig.from_pretrained('facebook/wav2vec2-base-960h')
    return Wav2Vec2ForCTC(config).to(device)


config = {"download_delay": 8000000,
          "partition_names": [f"{OBJECT_NAME}_{i}" for i in range(1, COUNT_PARTITIONS+1)]}

if MT == "T":
    model = TorchModelLoader(init_model, bucket, config).load()
else:
    model = init_model()
    std = torch.load(io.BytesIO(bucket.Object(OBJECT_NAME).get()['Body'].read()))
    model.load_state_dict(std)
    del std

model.eval()

generator = pipeline('automatic-speech-recognition', model=model, tokenizer=processor, feature_extractor=feature_extractor)

inp = []
for i in range(1, int(os.getenv('BS', 1))+1):
    inp .append(f"sample1.flac")

output = generator(inp)

print(output)
print("Response time is: ", time.time() - start_time)
