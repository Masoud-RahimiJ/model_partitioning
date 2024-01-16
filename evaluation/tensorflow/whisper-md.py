import time, os
start_time = time.time()
from transformers import TFWhisperForConditionalGeneration, WhisperProcessor, AutoFeatureExtractor, AutoConfig, pipeline, set_seed
import boto3
from botocore.client import Config
from lib.tf_model_loader import TFModelLoader
import time, subprocess
import numpy as np
import tensorflow as tf


BUCKET="dnn-models"
OBJECT_NAME="whisper-medium"
COUNT_PARTITIONS=121
MT = os.getenv("MT", "F")

s3 = boto3.resource('s3', endpoint_url='http://10.10.1.2:9000',aws_access_key_id='admin', aws_secret_access_key='ramzminio', config=Config(signature_version='s3v4'),)
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


def load_audio(inputs, feature_extractor):
    with open(inputs, "rb") as f:
        inputs = f.read()
    inputs = ffmpeg_read(inputs, feature_extractor.sampling_rate)
    processed = feature_extractor(inputs, sampling_rate=feature_extractor.sampling_rate, return_tensors="np").input_features
    return processed


set_seed(42)
processor = WhisperProcessor.from_pretrained('openai/whisper-medium')
feature_extractor = AutoFeatureExtractor.from_pretrained("openai/whisper-medium")

def init_model():
    config = AutoConfig.from_pretrained('openai/whisper-medium')
    model = TFWhisperForConditionalGeneration(config)
    model.build((1500, 1024))
    return model

config = {"download_delay": 6000000,
          "partition_names": [f"{OBJECT_NAME}_{i}.h5" for i in range(1, COUNT_PARTITIONS+1)]}

if MT == "T":
    model = TFModelLoader(init_model, bucket, config).load()
else:
    model = init_model()
    bucket.download_file(Filename = f"{OBJECT_NAME}.h5", Key= f"{OBJECT_NAME}.h5")
    model.load_weights(f"{OBJECT_NAME}.h5")
    os.remove(f"{OBJECT_NAME}.h5")

model.config.forced_decoder_ids = None



# audios = []
# for i in range(int(os.getenv('BS', 1))):
#     audios.append(load_audio("sample1.flac", feature_extractor)[0])
# audios = tf.convert_to_tensor(audios)
audios = np.random.randn(int(os.getenv('BS', 1)), 800, 300)


logits = model.generate(audios, max_new_tokens=1)

# print(pred_ids, output)
print("Response time is: ", time.time() - start_time)
