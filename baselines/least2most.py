# %%
import sys
sys.argv = ['']
sys.path.append('/root/autocot/')
from rouge_score import rouge_scorer

from utils import *
from api import cot
import re
import time
import argparse
from string import Template
import numpy as np
import pandas as pd
from openai import OpenAI

parser = argparse.ArgumentParser(description="Zero-shot-CoT")
parser.add_argument("--dataset", type=str, default='strategyqa',
                    help="maximum number of workers for dataloader")
parser.add_argument("--dataset_path", type=str, default='/root/datasets/strategy/task.json',
                    help="maximum number of workers for dataloader")
parser.add_argument("--method", type=str, default='tot',
                    help="maximum number of workers for dataloader")


# 36439 truth

# 46795 general
def check_result(query, model = 'gpt-3.5-turbo', max_tokens=1000, temperature=0):
  message = []
  message.append({'role': 'user', 'content': query})  
  completion = client.chat.completions.create(
    model=model,
    messages=message,
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
import openai
openai.api_key = "sk-proj-xxxx"
client = OpenAI()

# %%
predicts = []
real_answers = []
real_answers2 = []
steps = []
count = 0
true = 0
total = 0
pre = []
unsupport = 0
while i < len(questions):
    # if i % 2 != 0:
    #     i += 1
    #     continue
    question = questions[i]
    completion = client.chat.completions.create(
        model = "gpt-3.5-turbo",
        temperature = 0.8,
        max_tokens = 2000,
        messages = [
            {"role": "system", "content": "Construct an two step reasoning chain for complex question. For each step of the reasoning chain, generate a sub-question (Sub). Please provide the subquestion only and each sub split with '\n'"},
            {"role": "user", "content": "The question is " + question},
        ]
    )
    sub_questions = completion.choices[0].message.content.split('\n')[1:]
    response = ''
    j = 0
    print(i)
    if len(sub_questions) > 3:
        i += 1
        continue
    steps.append(len(sub_questions))
    sub_count= 0
    
    while j < len(sub_questions):
        try:
            sub_question = response + sub_questions[j]
            response = cot(method="zero_shot", question=sub_question,
                            question_temp=question_temp)
            response = sub_question + response
            j += 1
        except:
            j += 1
    response = cot(method="zero_shot", question=response + question,
                            question_temp=question_temp)
    predicts.append(response)
    real_answers.append(answers[i])
    real_answers2.append(answers2[i])
    i += 1


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