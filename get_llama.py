import os
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer

# load environment variable HUGGINGFACE_API_KEY
access_token = os.environ['HUGGINGFACE_API_KEY']

# Specify the directory path where you want to save the model and tokenizer
save_pretrained = False
org_name = 'meta-llama'; model_name = 'Llama-2-13b-hf'
# org_name = 'google'; model_name = 'gemma-2b'
directory_path = os.path.join('.', model_name)

def save_model():
    model = AutoModelForCausalLM.from_pretrained(f"{org_name}/{model_name}",  device_map='auto', token=access_token)
    if save_pretrained:
        print('saving model')
        model.save_pretrained(directory_path)

def save_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(f"{org_name}/{model_name}", device_map='auto', token=access_token)
    if save_pretrained:
        tokenizer.save_pretrained(directory_path)

if  __name__ == "__main__":

    print("saving tokenizer...")
    save_tokenizer()

    print("saving model...")
    save_model()

    print("done")