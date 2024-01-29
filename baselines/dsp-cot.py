#!/usr/bin/env python
# coding: utf-8

# %%
import sys
import os
import dspy
import sys
sys.argv = ['']
sys.path.append('/root/autocot/')
import openai 
openai.api_key = "sk-xxxxx"
import pandas as pd
from utils import *
from api import cot
import re
import time
import argparse

import numpy as np

# In[23]:


turbo = dspy.OpenAI(model='gpt-3.5-turbo')
colbertv2_wiki17_abstracts = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')

dspy.settings.configure(lm=turbo, rm=colbertv2_wiki17_abstracts)


# In[24]:


from dspy.datasets import HotPotQA

# Load the dataset.
dataset = HotPotQA(train_seed=1, train_size=20, eval_seed=2023, dev_size=50, test_size=0)

# Tell DSPy that the 'question' field is the input. Any other fields are labels and/or metadata.
trainset = [x.with_inputs('question') for x in dataset.train]
devset = [x.with_inputs('question') for x in dataset.dev]


# In[25]:


class GenerateAnswer(dspy.Signature):
    """Answer questions with short factoid answers."""

    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")


# In[26]:


class RAG(dspy.Module):
    def __init__(self, num_passages=3):
        super().__init__()

        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)
    
    def forward(self, question):
        context = self.retrieve(question).passages
        prediction = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=prediction.answer)


# In[27]:


from dspy.teleprompt import BootstrapFewShot

# Validation logic: check that the predicted answer is correct.
# Also check that the retrieved context does actually contain that answer.
def validate_context_and_answer(example, pred, trace=None):
    answer_EM = dspy.evaluate.answer_exact_match(example, pred)
    answer_PM = dspy.evaluate.answer_passage_match(example, pred)
    return answer_EM and answer_PM

# Set up a basic teleprompter, which will compile our RAG program.
teleprompter = BootstrapFewShot(metric=validate_context_and_answer)

# Compile!
compiled_rag = teleprompter.compile(RAG(), trainset=trainset)


# In[28]:

parser = argparse.ArgumentParser(description="Zero-shot-CoT")
parser.add_argument("--dataset", type=str, default='bigbench_truth',
                    help="maximum number of workers for dataloader")
parser.add_argument("--dataset_path", type=str, default='/root/datasets/BIG-bench/bigbench/benchmark_tasks/truthful_qa/task_mc.json',
                    help="maximum number of workers for dataloader")
parser.add_argument("--method", type=str, default='zero_shot',
                    help="maximum number of workers for dataloader")





args = parser.parse_args()

data = pd.read_csv('/root/experiment_datasets/webqa.csv')



questions = data['question'].to_list()


answers_temp = data['answers'].to_list()
answers = []
for item in answers_temp:
  item = item[2:-2]
  temp = item.split(',')
  answers.append([e.strip(' ').replace("'","") for e in temp])

question_temp = [questions[:3], answers[:3]]
answers = answers[3:]
questions = questions[3:]
predicts = []
real_answers = []
api = []
for i in range(20):
    global count
    count = 0
    try:
        my_question = questions[i]

        # Get the prediction. This contains `pred.context` and `pred.answer`.
        pred = compiled_rag(my_question)
        real_answers.append(answers[i])
        # Print the contexts and the answer.
        prompts = "I have the answer of the problem" + pred.answer + '. And provide me the choice from question '
        response = cot(method="zero_shot", question=prompts + questions[i],
                                question_temp=question_temp)
        count += 1
        predicts.append(response)
        api.append(count)
        print('-------------------')
    except:
        continue


# In[ ]:
results = []
for sentence, answer_list in zip(predicts, answers):
    # Check if any word in the answer_list is in the sentence
    result = 1 if any(word in sentence for word in answer_list) else 0
    results.append(result)

# %%
# Calculate accuracy
accuracy = np.sum(results) / len(results) * 100
print(np.mean(np.array(api)))
print(f"Accuracy: {accuracy}%")





