"""
GPT version of conformal_llm_scores.py
"""

from collections import defaultdict

# List of task we consider
task_list = ['college_computer_science', 'formal_logic', 'high_school_computer_science',
             'computer_security', 'machine_learning',

             'clinical_knowledge', 'high_school_biology', 'anatomy', 'college_chemistry',
             'college_medicine', 'professional_medicine',

             'business_ethics', 'professional_accounting', 'public_relations',
             'management', 'marketing'
             ]


def modify_task_data(task_data, token_limit, max_size_prompt_len):
    '''
    task_data: load_dataset('lukaemon/mmlu', subject_name), i.e., comes from mmlu subject
    token_limit: the maximum sized token used in forward pass (some questions are too large and thus
    are difficult to fit into memory given we use a single A100, thus we keep a token_limit of 1500 tokens.)
    max_size_prompt_len: Since we use 10 different prompts for one question which all differ in their one-shot
    question, the number of total questions may become different for each prompt. Thus we chose the
    max_size_prompt_len, which is the largest of 10 prompts, to remove questions that exceed token_limit,
    This results in same count of questions across all 10 prompts.

    Returns task_data with questions exceeding (token_limit-max_size_prompt_len) length tokens removed.
    '''
    new_task_data = {
        'train': defaultdict(list),
        'validation': defaultdict(list),
        'test': defaultdict(list),
    }
    for split in new_task_data.keys():
        for i in range(len(task_data[split])):
            q = task_data[split]['input'][i]
            a = task_data[split]['A'][i]
            b = task_data[split]['B'][i]
            c = task_data[split]['C'][i]
            d = task_data[split]['D'][i]
            target = task_data[split]['target'][i]
            if len(q) + max(map(len, [a, b, c, d])) + max_size_prompt_len < token_limit:
                new_task_data[split]['input'].append(q)
                new_task_data[split]['A'].append(a)
                new_task_data[split]['B'].append(b)
                new_task_data[split]['C'].append(c)
                new_task_data[split]['D'].append(d)
                new_task_data[split]['target'].append(target)
    return new_task_data
