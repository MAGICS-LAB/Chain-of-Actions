# %%
import sys
sys.argv = ['']
sys.path.append('/root/autocot/')

import numpy as np

from utils import *
from api import cot
import re
import time
import argparse
import pandas as pd
from rouge_score import rouge_scorer

parser = argparse.ArgumentParser(description="Zero-shot-CoT")
parser.add_argument("--dataset", type=str, default='bigbench_general',
                    help="maximum number of workers for dataloader")
parser.add_argument("--dataset_path", type=str, default='/root/datasets/General QA/task.json',
                    help="maximum number of workers for dataloader")
parser.add_argument("--method", type=str, default='cos',
                    help="maximum number of workers for dataloader")







 
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
real_answers = []
real_answers2 = []
predicts = []
while i < len(questions):
    try:
        if i % 3 != 0:
          i += 1
          continue
        question = questions[i]
        response = cot(method="few_shot", question=question,
                        question_temp=question_temp)
        predicts.append(response)
        real_answers.append(answers[i])
        real_answers2.append(answers2[i])
        if i % 10 == 0:
            print(str(i//10) + '/100')
        i += 1
    except:
        i += 1


# %%
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

# %%
