import time
start = time.time()
from transformers import Wav2Vec2Processor, TFWav2Vec2ForCTC, AutoFeatureExtractor, AutoConfig, pipeline, set_seed
import boto3
from botocore.client import Config
import numpy as np
import time
import subprocess
import tensorflow as tf
import soundfile as sf

print(time.time()-start)


def ffmpeg_read(bpayload: bytes, sampling_rate: int) -> np.array:
    """
    Helper function to read an audio file through ffmpeg.
    """
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

    processed = feature_extractor(
        inputs, sampling_rate=feature_extractor.sampling_rate, return_tensors="tf"
    )
    return processed.input_values[0]



BUCKET="dnn-models"
OBJECT_NAME="../models/wav.h5"
s3 = boto3.resource('s3', endpoint_url='http://10.10.1.2:9000',aws_access_key_id='masoud', aws_secret_access_key='ramzminio', config=Config(signature_version='s3v4'),)
bucket = s3.Bucket("dnn-models")

# start = time.time()
# bucket.download_file(Filename=OBJECT_NAME, Key=OBJECT_NAME)
# print(time.time()-start)

start = time.time()
set_seed(42)
processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h')
feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")
config = AutoConfig.from_pretrained('facebook/wav2vec2-base-960h')
model = TFWav2Vec2ForCTC(config)
print(time.time()-start)

start = time.time()
audio = sf.read("sample2.flac")
audio = processor(audio, return_tensors="tf").input_values
print(time.time()-start)


model(audio)

start = time.time()
model.load_weights(OBJECT_NAME)
print(time.time()-start)


start = time.time()
logits = model(audio).logits[0]
pred_ids = tf.math.argmax(logits)
output = processor.batch_decode(pred_ids, skip_special_tokens=True)
print(time.time()-start)

print(pred_ids)