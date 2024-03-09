import time, io
start_time = time.time()
from transformers import WhisperForConditionalGeneration, WhisperProcessor, AutoFeatureExtractor, AutoConfig, pipeline, set_seed
import boto3
from botocore.client import Config
from lib.torch_model_loader import TorchModelLoader
import time, io, subprocess
import numpy as np
import torch, os
from accelerate import init_empty_weights


BUCKET="dnn-models"
OBJECT_NAME="whisper-large"
COUNT_PARTITIONS=46
MT = os.getenv("MT", "T")
device = torch.device("cpu")



s3 = boto3.resource('s3', endpoint_url='http://128.110.219.188:9000',aws_access_key_id='admin', aws_secret_access_key='ramzminio', config=Config(signature_version='s3v4'),)
bucket = s3.Bucket("dnn-models")


def ffmpeg_read(bpayload: bytes, sampling_rate: int) -> np.array:
    ar = f"{sampling_rate}"
    ac = "1"
    format_for_conversion = "f32le"
    ffmpeg_command = [
        "ffmpeg",
        "-i",
        "pipe:0",
        "-ac",
        ac,
        "-ar",
        ar,
        "-f",
        format_for_conversion,
        "-hide_banner",
        "-loglevel",
        "quiet",
        "pipe:1",
    ]
    try:
        ffmpeg_process = subprocess.Popen(ffmpeg_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    except FileNotFoundError:
        raise ValueError("ffmpeg was not found but is required to load audio files from filename")
    output_stream = ffmpeg_process.communicate(bpayload)
    out_bytes = output_stream[0]
    audio = np.frombuffer(out_bytes, np.float32)
    if audio.shape[0] == 0:
        raise ValueError("Malformed soundfile")
    return audio


def load_audio(feature_extractor):
    inp = []
    for i in range(1, int(os.getenv('BS', 1))+1):
        tmp = 0
        with open(f"sample1.flac", "rb") as f:
            tmp = f.read()
        tmp = ffmpeg_read(tmp, feature_extractor.sampling_rate)
        inp.append(tmp)
    processed = feature_extractor(inp, sampling_rate=feature_extractor.sampling_rate, return_tensors="pt").input_features
    return processed.to(device)


set_seed(42)
processor = WhisperProcessor.from_pretrained('openai/whisper-large-v2')
feature_extractor = AutoFeatureExtractor.from_pretrained("openai/whisper-large-v2")

def init_model():
    # with init_empty_weights():
    config = AutoConfig.from_pretrained('openai/whisper-large-v2')
    return WhisperForConditionalGeneration(config).to(device)

config = {"download_delay": 8000000,
          "partition_names": [f"{OBJECT_NAME}_{i}" for i in range(1, COUNT_PARTITIONS+1)]}

if MT == "T":
    model = TorchModelLoader(init_model, bucket, config).load()
else:
    model = init_model()
    std = torch.load(io.BytesIO(bucket.Object(OBJECT_NAME).get()['Body'].read()))
    model.load_state_dict(std)
    del std

model.config.forced_decoder_ids = None
model.eval()
model.tie_weights()

audio = load_audio(feature_extractor)
predicted_ids = model.generate(audio)
output = processor.batch_decode(predicted_ids, skip_special_tokens=True)


print(output)
print("Response time is: ", time.time() - start_time)
