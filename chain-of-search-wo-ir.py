# %%
import sys
sys.argv = ['']
sys.path.append('/root/autocot/')
from string import Template
import numpy as np

from utils import *
from api import cot
import re
import time
import argparse
import pandas as pd
from openai import OpenAI

import openai
openai.api_key = "sk-proj-xxxxx"
client = OpenAI()

parser = argparse.ArgumentParser(description="Zero-shot-CoT")
parser.add_argument("--dataset", type=str, default='bigbench_truth',
                    help="maximum number of workers for dataloader")
parser.add_argument("--dataset_path", type=str, default='/root/datasets/truthful_qa/task_mc.json',
                    help="maximum number of workers for dataloader")
parser.add_argument("--method", type=str, default='cos',
                    help="maximum number of workers for dataloader")





def check_result(query, model = 'gpt-3.5-turbo', max_tokens=1000, temperature=0):
  message = []
  message.append({'role': 'user', 'content': query})  
  completion = client.chat.completions.create(model = model, 
                                            messages = message, 
                                            temperature = temperature,
                                            max_tokens = max_tokens)
  return completion.choices[0].message.content

def extract_answer(generated):
    if '\n' not in generated:
        last_line =  generated
    else: 
        last_line = generated.split('\n')[-1]

    if ':' not in last_line:
        after_colon = last_line
    else:
        after_colon = generated.split(':')[-1]
    if after_colon:
        if ' ' == after_colon[0]:
            after_colon = after_colon[1:]
        if '.' == after_colon[-1]:
            after_colon = after_colon[:-1]

    return after_colon
 
args = parser.parse_args()

# %%
data = pd.read_csv('/home/hlv8980/dd/final_data_questions_100.csv')



# %%
questions = data['input'].to_list()


# %%
answers = data['answer1'].to_list()


answers2 = data['answer2'].to_list()


  


i = 0

question_temp = [questions[:3], answers[:3], answers2[:3]]
answers = answers[3:]
questions = questions[3:]
answers2 = answers2[3:]
template = []


# %%

predicts = []
true = 0
total = 0
pre = []
unsupport = 0
temp = []   
real_answers = []
real_answers2 = []
while i < len(questions):
    if i % 3 != 0:
        i += 1
        continue
    question = questions[i]
    response = cot(method="tot", question=question,
                    question_temp=question_temp)
    choice = extract_answer(response)
    predicts.append(choice)
    real_answers.append(answers[i])
    real_answers2.append(answers2[i])
    i += 1

from rouge_score import rouge_scorer
def calculate_rouge_l(string1, string2):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(string1, string2)
    return scores['rougeL'].fmeasure * 100
def get_max_rouge_l(A, B, C):
    rouge_l_ab = calculate_rouge_l(A, B)
    rouge_l_ac = calculate_rouge_l(A, C)
    return max(rouge_l_ab, rouge_l_ac)


result = []
for i in range(len(real_answers)):
    result.append(get_max_rouge_l(predicts[i], real_answers[i], real_answers2[i]))

print('Average Rouge-L:', sum(result)/len(result))
# args.direct_answer_trigger_for_fewshot = "The answer is"
# predict = []
# pattern = r'\((.)\)'  # Matches a single character inside parentheses
# for pred in predicts:
#     # match = re.search(pattern, pred)
#     # if match:
#     #     result = match.group(1)
#     #     predict.append(result)
#     # else:
#     #     print(pred)
#     pred = pred.split('.')[0]
#     pred = answer_cleansing(args, pred, must_choice=True)
#     predict.append(pred)


# final_predictions = [1 for true, pred in zip(
#     real_answers, predict) if str(true) == str(pred)]
# correct_predictions = sum(final_predictions)

# # Calculate accuracy
# accuracy = correct_predictions / len(real_answers) * 100
# final_predictions = pd.DataFrame(final_predictions)
# print(final_predictions)

# print(f"Accuracy: {accuracy}%")




# %%
# Calculate accuracy

print('done')
# %%
