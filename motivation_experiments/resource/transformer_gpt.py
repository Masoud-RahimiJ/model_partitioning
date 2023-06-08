from transformers import pipeline, set_seed
from time import sleep

text = "The White man worked as a"

generator = pipeline('text-generation', model='gpt2-large')
set_seed(42)
output = generator("Hello, I'm a language model,", max_length=30, num_return_sequences=5)
print(output)
sleep(5)