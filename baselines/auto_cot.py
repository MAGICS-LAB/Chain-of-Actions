import pandas as pd
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
import openai
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

model = SentenceTransformer('all-MiniLM-L6-v2')

questions_final = []
answers_final = []

# data = pd.read_csv('/home/hlv8980/dd/final_data_sub_100.csv')

# questions = data['input'].to_list()


# # %%
# answers_temp = data['answer'].to_list()

# answers = []
# for item in answers_temp:
#   item = item[2:-2]
#   temp = item.split(',')
#   answers.append([e.strip(' ').replace("'","") for e in temp])

# questions_final.extend(questions[:3])
# answers_final.extend(answers[:3])





# parser = argparse.ArgumentParser(description="Zero-shot-CoT")
# parser.add_argument("--dataset", type=str, default='bigbench_truth',
#                     help="maximum number of workers for dataloader")
# parser.add_argument("--dataset_path", type=str, default='/root/datasets/truthful_qa/task_mc.json',
#                     help="maximum number of workers for dataloader")
# parser.add_argument("--method", type=str, default='cos',
#                     help="maximum number of workers for dataloader")

# args = parser.parse_args()


# args.dataset = 'strategyqa'
# args.dataset_path = '/root/datasets/strategy/task.json'
# questions, answers = data_reader(args)

# questions_final.extend(questions[:3])
# answers_final.extend(answers[:3])

# args.dataset = 'bigbench_social'
# args.dataset_path = '/root/datasets/socialQA/task.json'
# questions, answers = data_reader(args)

# questions_final.extend(questions[:3])
# answers_final.extend(answers[:3])

# args.dataset = 'bigbench_truth'
# args.dataset_path = '/root/datasets/truthful_qa/task_mc.json'
# questions, answers = data_reader(args)

# questions_final.extend(questions[:3])
# answers_final.extend(answers[:3])

# args.dataset = 'bigbench_general'
# args.dataset_path = '/root/datasets/General QA/task.json'
# questions, answers = data_reader(args)

# questions_final.extend(questions[:3])
# answers_final.extend(answers[:3])

data = pd.read_csv('/home/hlv8980/dd/final_data_sub_100.csv')

questions = data['input'].to_list()


# %%
answers_temp = data['answer'].to_list()
answers = []
for item in answers_temp:
  item = item[2:-2]
  temp = item.split(',')
  answers.append([e.strip(' ').replace("'","") for e in temp])
  
questions_final.extend(questions[:3])
answers_final.extend(answers[:3])


import openai
openai.api_key = "sk-proj-xxxx"
client = OpenAI()


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

# %%
data = pd.read_csv('/home/hlv8980/dd/final_data_sub_100.csv')



# %%
questions = data['input'].to_list()


# %%
answers_temp = data['answer'].to_list()
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
real_answers = []

predicts = []
true = 0
total = 0
pre = []
unsupport = 0
temp = []
path = []
real_answers = []



while i < len(questions):
    if i % 10 != 0:
        i += 1
        continue
    question = questions[i]
    question = questions[i]
    question_embeddings = model.encode(question).reshape(1, -1)
    questions_embeddings = model.encode(questions_final)
    questions_embeddings = np.concatenate([question_embeddings, questions_embeddings])


    kmeans = KMeans(n_clusters=3, random_state=0).fit(questions_embeddings)
    labels = kmeans.labels_
    final_template = [questions_final[i] for i in range(len(questions_final)-1) if labels[i] == labels[-1]]
    path.append(len(final_template))
    for i in range(len(final_template)):
      question = final_template[i] + ' ' + question
    response = cot(method="zero_shot_cot", question=question,
                    question_temp=question_temp)
    choice = extract_answer(response)
    predicts.append(choice)
    real_answers.append(answers[i])
    i += 1
    print(i)
    
    # prompts = "I have the answer of the problem" + choice + '. And provide me the choice from question. Provide A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P only'
    # choice = cot(method="zero_shot", question=prompts + questions[i],
    #                         question_temp=question_temp)
    # print('choice: ',choice, 'true_label: ', str(answers[i]))
    # print('Now is number: ', total)
    # temp.append('choice: '+choice+'true_label: '+ str(answers[i]))
    # template_judge = Template("""
    # If the meaning of the answer "$ANSWER" is similar or equal to the meaning of "($AN)", is this statement true? If true, reply [True], if false , reply [False]  
    # """).substitute(ANSWER= choice, AN = str(answers[i])) 
    # # print(template_judge)
    # try:
    #     print(template_judge)
    #     judge_result = check_result(query=template_judge)
    #     print('judge_result: ', judge_result)
    #     if judge_result.find('True') != -1:
    #         true += 1
    #     total += 1
    #     ii = 0
    #     i += 1
    #     pre.append([response, judge_result])  
    #     print('accuracy: ', true / total * 100, '%') 
    # except:
    #     try:
    #         print(template_judge)
    #         judge_result = check_result(query=template_judge)
    #         print('judge_result: ', judge_result)
    #         if judge_result.find('True') != -1:
    #             true += 1
    #         total += 1
    #         ii = 0
    #         i += 1
    #         pre.append([response, judge_result])  
    #         print('accuracy: ', true / total * 100, '%')
    #     except:
    #         i += 1
    #         continue


results = []
for sentence, answer_list in zip(predicts, real_answers):
    # Check if any word in the answer_list is in the sentence
    result = 1 if any(word in sentence for word in answer_list) else 0
    results.append(result)

# %%
# Calculate accuracy
accuracy = np.sum(results) / len(results) * 100

print(f"Accuracy: {accuracy}%")

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
