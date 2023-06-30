from transformers import GPT2Tokenizer, GPT2Model, pipeline, set_seed
set_seed(42)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-xl')
model = GPT2Model.from_config('gpt2-xl')
text = "Replace me by any text you'd like."
generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
output = generator("Hello, I'm a language model,", max_length=30, num_return_sequences=5)