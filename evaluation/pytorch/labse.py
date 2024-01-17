import time, io
start_time = time.time()
from transformers import AutoConfig, BertLMHeadModel, BertTokenizerFast, pipeline, set_seed
import boto3
from botocore.client import Config
import time, io
import torch, os
from accelerate import init_empty_weights
from lib.torch_model_loader import TorchModelLoader


BUCKET="dnn-models"
OBJECT_NAME="labse"
COUNT_PARTITIONS=8

session = boto3.Session()
s3 = session.resource('s3', endpoint_url='http://10.10.1.2:9000',aws_access_key_id='admin', aws_secret_access_key='ramzminio', config=Config(signature_version='s3v4'),)
bucket = s3.Bucket("dnn-models")

set_seed(42)
tokenizer = BertTokenizerFast.from_pretrained('setu4993/LaBSE')

def init_model():
    # with init_empty_weights():
    config=AutoConfig.from_pretrained('setu4993/LaBSE')
    return BertLMHeadModel(config)

config = {"download_delay": 8000000,
          "partition_names": [f"{OBJECT_NAME}_{i}" for i in range(1, COUNT_PARTITIONS+1)]}

model = TorchModelLoader(init_model, bucket, config).load()


# model=init_model()
# stt = time.time()
# std = torch.load(io.BytesIO(bucket.Object(OBJECT_NAME).get()['Body'].read()))
# model.load_state_dict(std)
# del std

model.eval()


text = ["The cat jumped onto the kitchen counter and meowed", "I can't believe how fast time is flying by", "The sun was setting over the horizon, painting the sky", "She smiled and waved as she walked away", "The smell of fresh coffee filled the air", "He ran through the park, feeling the wind in his hair", "The book fell off the shelf with a loud thud", "The flowers bloomed in vibrant colors in the garden", "I closed my eyes and took a deep breath", "The sound of laughter echoed through the room", "The waves crashed against the rocky shore relentlessly", "She twirled around in her new dress, feeling happy", "The old man sat on the porch, watching the sunset", "I reached out to grab his hand and held it tightly", "The rain poured down from the dark clouds above", "The children played happily in the playground together", "I could hear birds chirping outside my window", "He gazed at her with admiration in his eyes", "The smell of freshly baked bread wafted through the air", "She danced gracefully across the stage with confidence", "I felt a sense of peace wash over me suddenly", "The fire crackled and popped in the fireplace warmly", "We hiked up to the top of the mountain togethe", "The stars twinkled brightly in the night sky above u", "The baby giggled and clapped his hands excitedl", "I watched as she carefully arranged flowers in a vas", "The music blared loudly from speakers at the part", "I sipped on my hot tea as I watched it snow outsid", "The scent of pine trees filled my lungs as I hike", "He smiled warmly at me and offered his han", "I felt a surge of adrenaline as I jumped of", "She laughed heartily at his silly joke and hugged hi"]

generator = pipeline('text-generation', model=model, tokenizer=tokenizer, pad_token_id=50256)

output = generator(text[:int(os.getenv('BS', 1))], max_new_tokens=1, num_return_sequences=1)

print(output)
print("Response time is: ", time.time() - start_time)
