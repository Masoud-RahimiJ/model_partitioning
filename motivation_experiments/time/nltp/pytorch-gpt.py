import time
import torchvision
import time
import torch
from accelerate import init_empty_weights
# print(time.time()-start)


# BUCKET="dnn-models"
# OBJECT_NAME="../models/gpt2.pt"
# device = torch.device("cuda")
# s3 = boto3.resource('s3', endpoint_url='http://10.10.1.2:9000',aws_access_key_id='masoud', aws_secret_access_key='ramzminio', config=Config(signature_version='s3v4'),)
# bucket = s3.Bucket("dnn-models")

start = time.time()
# bucket.download_file(Filename=OBJECT_NAME, Key=OBJECT_NAME)
# print(time.time()-start)

start = time.time()
# set_seed(42)
# tokenizer = AutoTokenizer.from_pretrained('gpt2-xl')
# config=AutoConfig.from_pretrained('gpt2-xl')
with init_empty_weights():
    model = torchvision.models.vgg19(weights=None)
model.eval()
print(time.time()-start)

# start = time.time()
# state_dict = torch.load(OBJECT_NAME)
# model.load_state_dict(state_dict)
# print(time.time()-start)

# text = "Replace me by any text you'd like."
# start = time.time()
# generator = pipeline('text-generation', model=model, tokenizer=tokenizer, device=torch.device("cuda"))
# print(time.time()-start)

# start = time.time()
# output = generator("Hello, I'm a language model,", max_length=30, num_return_sequences=1)
# print(time.time()-start)

# print(output)