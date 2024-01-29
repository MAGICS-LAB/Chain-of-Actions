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


parser = argparse.ArgumentParser(description="Zero-shot-CoT")
parser.add_argument("--dataset", type=str, default='bigbench_general'
                    help="maximum number of workers for dataloader")
parser.add_argument("--dataset_path", type=str, default='/root/datasets/General QA/task.json',
                    help="maximum number of workers for dataloader")
parser.add_argument("--method", type=str, default='cos',
                    help="maximum number of workers for dataloader")







 
args = parser.parse_args()

# %%
data = pd.read_csv('/root/experiment_datasets/webqa.csv')



# %%
questions = data['question'].to_list()


# %%
answers_temp = data['answers'].to_list()
answers = []
for item in answers_temp:
  item = item[2:-2]
  temp = item.split(',')
  answers.append([e.strip(' ').replace("'","") for e in temp])
  

  

# %%
i = 0

question_temp = [questions[:3], answers[:3]]
answers = answers[3:]
questions = questions[3:]


predicts = []
while i < len(questions):
    try:
        question = questions[i]
        response = cot(method="few_shot", question=question,
                        question_temp=question_temp)
        predicts.append(response)
        if i % 10 == 0:
            print(str(i//10) + '/100')
        i += 1
    except:
        time.sleep(10)


# %%
results = []
for sentence, answer_list in zip(predicts, answers):
    # Check if any word in the answer_list is in the sentence
    result = 1 if any(word in sentence for word in answer_list) else 0
    results.append(result)

# %%
# Calculate accuracy
accuracy = np.sum(results) / len(results) * 100

print(f"Accuracy: {accuracy}%")

# %%
