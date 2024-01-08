import time
start_time = time.time()
from transformers import Wav2Vec2Processor, TFWav2Vec2ForCTC, AutoFeatureExtractor, AutoConfig, pipeline, set_seed
import boto3
from botocore.client import Config
import numpy as np
import time
import subprocess
from lib.tf_model_loader import TFModelLoader
import tensorflow as tf

BUCKET="dnn-models"
OBJECT_NAME="wav"
COUNT_PARTITIONS=10

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
        inputs, sampling_rate=feature_extractor.sampling_rate, return_tensors="np"
    )
    return processed


s3 = boto3.resource('s3', endpoint_url='http://10.10.1.2:9000',aws_access_key_id='admin', aws_secret_access_key='ramzminio', config=Config(signature_version='s3v4'),)
bucket = s3.Bucket("dnn-models")

set_seed(42)
processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h')
feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")
config = AutoConfig.from_pretrained('facebook/wav2vec2-base-960h')

def init_model():
    return TFWav2Vec2ForCTC(config)

config = {"download_delay": 6000000,
          "partition_names": [f"{OBJECT_NAME}{i}.h5" for i in range(1, COUNT_PARTITIONS)]}

model = TFModelLoader(init_model, bucket, config).load()


audio = load_audio("sample2.flac", feature_extractor)

model(audio)
logits = model(audio).logits[0]
pred_ids = tf.math.argmax(logits)
output = processor.batch_decode(pred_ids, skip_special_tokens=True)
print(pred_ids)
print("Response time is: ", time.time() - start_time)
