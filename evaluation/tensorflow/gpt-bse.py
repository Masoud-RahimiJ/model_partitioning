import time
start_time = time.time()
from transformers import AutoTokenizer, AutoConfig, TFGPT2LMHeadModel, pipeline, set_seed
import boto3
from botocore.client import Config
import time
from lib.tf_model_loader import TFModelLoader


BUCKET="dnn-models"
OBJECT_NAME="gpt"
COUNT_PARTITIONS=10

s3 = boto3.resource('s3', endpoint_url='http://10.10.1.2:9000',aws_access_key_id='admin', aws_secret_access_key='ramzminio', config=Config(signature_version='s3v4'),)
bucket = s3.Bucket("dnn-models")

set_seed(42)
tokenizer = AutoTokenizer.from_pretrained('gpt2')
config=AutoConfig.from_pretrained('gpt2')

def init_model():
    return TFGPT2LMHeadModel(config)

config = {"download_delay": 6000000,
          "partition_names": [f"{OBJECT_NAME}{i}.h5" for i in range(1, COUNT_PARTITIONS)]}

model = TFModelLoader(init_model, bucket, config).load()

model.eval()


text = "Replace me by any text you'd like."
generator = pipeline('text-generation', model=model, tokenizer=tokenizer)

output = generator("Hello, I'm a language model,", max_length=30, num_return_sequences=1)

print(output)
print("Response time is: ", time.time() - start_time)
