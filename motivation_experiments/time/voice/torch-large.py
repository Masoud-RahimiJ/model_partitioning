import time
start = time.time()
from transformers import WhisperForConditionalGeneration, WhisperProcessor, AutoFeatureExtractor, AutoConfig, pipeline, set_seed
import boto3
from botocore.client import Config
import time, subprocess
import numpy as np
import torch
print(time.time()-start)


BUCKET="dnn-models"
OBJECT_NAME="whisper-large.pt"
s3 = boto3.resource('s3', endpoint_url='http://10.10.1.2:9000',aws_access_key_id='masoud', aws_secret_access_key='ramzminio', config=Config(signature_version='s3v4'),)
bucket = s3.Bucket("dnn-models")

start = time.time()
bucket.download_file(Filename=OBJECT_NAME, Key=OBJECT_NAME)
print(time.time()-start)

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


def load_audio(inputs, feature_extractor):
    with open(inputs, "rb") as f:
        inputs = f.read()
    inputs = ffmpeg_read(inputs, feature_extractor.sampling_rate)
    processed = feature_extractor(inputs, sampling_rate=feature_extractor.sampling_rate, return_tensors="pt").input_features
    return processed


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
audio = load_audio("sample2.flac", feature_extractor)
predicted_ids = model.generate(audio)
output = processor.batch_decode(predicted_ids, skip_special_tokens=True)
print(time.time()-start)


print(output)