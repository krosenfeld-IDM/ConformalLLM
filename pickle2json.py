"""
convert the pkls to jsonl
"""
import pickle
import json

def write_json(data, filename):
    # Writing the dictionary to a file
    with open(filename+'.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

for filename in ["accuracy_gpt_prompts_10", "accuracy_mmlu_prompts_10"]:
    with open(filename+".pkl","rb") as fid:
        data = pickle.load(fid)
    write_json(data, filename)