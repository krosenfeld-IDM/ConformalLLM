import prompt_questions as p
import numpy as np
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import pickle

from functions import (
    get_max_size_prompt_len,
    modify_task_data,
    get_prompt,
    get_question_dict,
    to_tokens_and_logprobs,
    extract_answer,
    accuracy,
    get_prediction_list
)

# List of task we consider
task_list = ['college_computer_science', 'formal_logic', 'high_school_computer_science',
             'computer_security', 'machine_learning',

             'clinical_knowledge', 'high_school_biology', 'anatomy', 'college_chemistry',
             'college_medicine', 'professional_medicine',

             'business_ethics', 'professional_accounting', 'public_relations',
             'management', 'marketing'
             ]


token_limit = 1500  # Maximum size of tokens used in forward pass.
n = 10 # number of different MMLU based prompts used.
task_list = task_list

max_size_prompt_len_dict = {}
prompt_question_ids_dict = {}
for subject_name in task_list:
    task_data = load_dataset('lukaemon/mmlu', subject_name)
    max_len, prompt_question_ids = get_max_size_prompt_len(task_data, subject_name, n=n,
                                                          max_allowed_prompt_len=700)
    max_size_prompt_len_dict[subject_name] = max_len
    prompt_question_ids_dict[subject_name] = prompt_question_ids

tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2", device_map='auto')

# Get prediction for subjects with MMLU based prompts

acc_dicts = {}

for subject_name in task_list:
    task_data = load_dataset('lukaemon/mmlu', subject_name)
    new_task_data = modify_task_data(task_data, token_limit, max_size_prompt_len_dict[subject_name])

    acc_dicts[subject_name] = []
    print(f'generating predictions for the subject {subject_name}')
    for j, question_num in enumerate(prompt_question_ids_dict[subject_name]):
        preds = []
        targets = []
        print(f'Running experiments with test set question_id {question_num}')
        prompt_add = get_prompt(task_data, task=subject_name, question_num=question_num, prompt_q=None)
        if j % 5 == 0:
            print(prompt_add)
        questions, answers = get_question_dict(new_task_data, prompt_q_id=question_num, prompt_add=prompt_add)
        for i, (question, answer) in enumerate(zip(questions, answers)):
            batch = to_tokens_and_logprobs(model, tokenizer, [v for v in question.values()])
            torch.cuda.empty_cache()
            preds.append(extract_answer(batch))
            targets.append(answer)
        print(f'Predictions Generated for {subject_name} for iteration {j}')
        print('Calculating accuracy')
        acc = round(accuracy(preds, targets), 3)
        acc_dicts[subject_name].append(acc)
        print(f'Accuracy on {subject_name} for iteration {j} is {acc:.2f} ')
    print('*****************************************************************************************')
    print(f'calculating average accuracy on {subject_name}')
    print(f'Average accuracy on {subject_name} is {np.mean(np.array(acc_dicts[subject_name])):.3f}')
    with open("distilgpt2_accuracy_mmlu_prompts_10.pkl", "wb") as f:
        pickle.dump(acc_dicts, f)


# Import GPT-4 based question prompts
prompt_list = [p.prompt_q_list_college_cs, p.prompt_q_list_formal_logic, p.prompt_q_list_high_school_cs,
               p.prompt_q_list_computer_security, p.prompt_q_list_machine_learning,

               p.prompt_q_list_clinical_knowledge, p.prompt_q_list_high_school_bio, p.prompt_q_list_anatomy,
               p.promtp_q_list_college_chemistry, p.prompt_q_list_college_medicine,
               p.prompt_q_list_professional_medicine,

               p.prompt_q_list_business_ethics, p.prompt_q_list_professional_accounting, p.prompt_q_list_pr,
               p.prompt_q_list_management, p.prompt_q_list_marketing
               ]


prompt_list = prompt_list

# Get predictions for each subject using GPT-4 based prompts

acc_dicts_mmlu = {}
for task, prompt in zip(task_list, prompt_list):
    prediction_lists, solution_answers, acc_list = get_prediction_list(task, prompt, token_limit, model, tokenizer)
    avg_acc = np.mean(np.array(acc_list))
    print('*****************************************************************************************')
    print(f'calculating average accuracy on {task}')
    print(f'Average accuracy on {task} is {avg_acc:.3f}')
    acc_dicts_mmlu[task] = acc_list
    with open("accuracy_gpt_prompts_10.pkl", "wb") as f:
        pickle.dump(acc_dicts_mmlu, f)
    scores = np.array([[[a[1] for a in p] for p in predictions] for predictions in prediction_lists])

    answer_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    targets = np.array(list(map(lambda x: answer_map[x], solution_answers)))
    np.save(f'distilgpt2_{task}_scores.npy', scores)
    np.save(f'distilgpt2_{task}_targets.npy', targets)


