import time, os
start_time = time.time()
from transformers import AutoTokenizer, AutoConfig, TFGPT2LMHeadModel, pipeline, set_seed
import boto3
from botocore.client import Config
import time
from lib.tf_model_loader import TFModelLoader


BUCKET="dnn-models"
OBJECT_NAME="gpt2-xl"
COUNT_PARTITIONS=194

s3 = boto3.resource('s3', endpoint_url='http://10.10.1.2:9000',aws_access_key_id='admin', aws_secret_access_key='ramzminio', config=Config(signature_version='s3v4'),)
bucket = s3.Bucket("dnn-models")

set_seed(42)



def init_model():
    config=AutoConfig.from_pretrained('gpt2-xl')
    tokenizer = AutoTokenizer.from_pretrained('gpt2-xl')
    model = TFGPT2LMHeadModel(config)
    generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
    generator("Hello", max_new_tokens=1, num_return_sequences=1)
    return model, generator
    

config = {"download_delay": 6000000,
          "partition_names": [f"{OBJECT_NAME}_{i}.h5" for i in range(1, COUNT_PARTITIONS+1)]}

model, generator = TFModelLoader(init_model, bucket, config).load()

# model, generator = init_model()
# bucket.download_file(Filename = f"{OBJECT_NAME}.h5", Key= f"{OBJECT_NAME}.h5")
# model.load_weights(f"{OBJECT_NAME}.h5")
# os.remove(f"{OBJECT_NAME}.h5")

output = generator("Hello, I'm a language model,", max_new_tokens=30, num_return_sequences=1)

print(output)
print("Response time is: ", time.time() - start_time)
