import os
from transformers import AutoModelForCausalLM, AutoTokenizer

# Specify the directory path where you want to save the model and tokenizer
save_pretrained = True
org_name = 'meta-llama'; model_name = 'Llama-2-7b-hf'
# org_name = 'google'; model_name = 'gemma-2b'
directory_path = os.path.join('.', model_name)

def save_model():
    model = AutoModelForCausalLM.from_pretrained(f"{org_name}/{model_name}",  device_map='auto')
    if save_pretrained:
        print('saving model')
        model.save_pretrained(directory_path)

def save_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(f"{org_name}/{model_name}", device_map='auto')
    if save_pretrained:
        tokenizer.save_pretrained(directory_path)

if  __name__ == "__main__":

    print("saving tokenizer...")
    save_tokenizer()

    print("saving model...")
    save_model()

    print("done")