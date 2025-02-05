# ConformalLLM
## Extending Conformal Prediction to LLMs 
Read our paper here [Conformal Prediction with Large Language Models for Multi-Choice Question Answering
](https://arxiv.org/abs/2305.18404)
### Code Contributors: Charles Lu and Bhawesh Kumar 
## Code Organization
conformal_llm_scores.py contains the python script for classification using 1-shot question prompts. It outputs three files
1) The softmax scores corresponding to each subjects for each of the 10 prompts
2) The accuracy for each subject prompt for mmlu-based 1-shot question as a dictionary where the key is the subject name and value is a list containing accuracy for each of the 10 prompts.
3) The accuracy for each subject prompt for gpt4-based 1-shot question as a dictionary where the key is the subject name and value is a list containing accuracy for each of the 10 prompts.

In conformal.ipynb, we have results for all conformal prediction experiments and gpt4 vs mmlu based prompt comparison. It requires the three files outputted by conformal_llm_scores.py to work. To run the experiment, download the llm_probs_gpt.zip file, unzip it and save it in your working directory and then run the conformal.ipynb file.

If you would like to run the experiments from scratch, apply for LLaMA [access here](https://docs.google.com/forms/d/e/1FAIpQLSfqNECQnMkycAp2jP4Z9TFX0cGR4uf7b_fBxjY_OjhJILlKGA/viewform) and then use the hugging face version of LLaMA by converting original LLaMA weights to hugging face version [refer here for instructions](https://huggingface.co/docs/transformers/main/model_doc/llama) and then run the conformal_llm_scores.py script. Requirements can be installed:
```
pip install -r requirements.txt
```

## Fork notes
- The original repository contained `.pkl` files which have been converted to `.json` by `pickle2json.py`.
- Hugging face models can fill the `.cache`, keep an eye on that.