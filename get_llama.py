from transformers import GPT2Tokenizer, GPT2Model

# Assuming you have already loaded the model and tokenizer
# tokenizer = GPT2Tokenizer.from_pretrained('meta-llama/Llama-2-13b-hf', low_cpu_mem_usage=True)
model = GPT2Model.from_pretrained('meta-llama/Llama-2-13b-hf', low_cpu_mem_usage=True)

# Specify the directory path where you want to save the model and tokenizer
directory_path = './Llama-2-13b-hf'

# Save the tokenizer and model
# tokenizer.save_pretrained(directory_path)
model.save_pretrained(directory_path)

# This will create the directory (if it doesn't already exist) and save the model and tokenizer there.
