from transformers import AutoTokenizer, AutoConfig, GPT2Model, pipeline, set_seed
set_seed(42)
tokenizer = AutoTokenizer.from_pretrained('gpt2-xl')
config=AutoConfig.from_pretrained('gpt2-xl')
model = GPT2Model(config)
text = "Replace me by any text you'd like."
generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
output = generator("Hello, I'm a language model,", max_length=30, num_return_sequences=5)