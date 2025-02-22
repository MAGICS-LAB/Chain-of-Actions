# %%
import sys
sys.argv = ['']
sys.path.append('/root/autocot/')
import numpy as np
from string import Template
from utils import *
from api import cot
import re
import time
import argparse
import tqdm
import pandas as pd
from openai import OpenAI


parser = argparse.ArgumentParser(description="Zero-shot-CoT")
parser.add_argument("--dataset", type=str, default='bigbench_date',
                    help="maximum number of workers for dataloader")
parser.add_argument("--dataset_path", type=str, default='/root/datasets/DATE/task.json',
                    help="maximum number of workers for dataloader")
parser.add_argument("--method", type=str, default='zero_shot',
                    help="maximum number of workers for dataloader")


#47458 general

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
  
  
# data = pd.read_csv('/root/datasets/fever/subset_fever.csv')


# questions = data['claim'] + " Please choice from following: (A)REFUTES (B)NOT ENOUGH INFO (C)SUPPORTS"
# questions = questions.to_list()


# answers_temp = data['label'].to_list()
# answers = []
# for a in answers_temp:
#     if a == 'REFUTES':
#         answers.append('A')
#     elif a == 'NOT ENOUGH INFO':
#         answers.append('B')
#     elif a == 'SUPPORTS':
#         answers.append('C')




# %%
import openai
openai.api_key = "sk-proj-xxxxx"
client = OpenAI()


# %%
predicts = []
real_answers = []
real_answers2 = []
prd_path = []
true = 0
pre = []
total = 0
temp = []
while i < len(questions):
    if i % 3 != 0:
        i += 1
        continue

    question = questions[i]
    completion = client.chat.completions.create(
        model = "gpt-3.5-turbo",
        temperature = 0.8,
        max_tokens = 2000,
        messages = [
            {"role": "system", "content": "Construct an reasoning chain for complex question. Generate sub-question (Sub) with three diverse reasoning paths. Please provide the subquestion only and each sub split with '\n'"},
            {"role": "user", "content": "The question is " + question},
        ]
    )
    sub_questions = completion.choices[0].message.content.split('\n')[1:]
    sub_predicts = ''
    j = 0
    if len(sub_questions) > 3:
        i += 1
        continue
    while j < len(sub_questions):
        try:
            sub_question = sub_questions[j]
            response = cot(method="zero_shot", question=sub_question,
                            question_temp=question_temp)
            sub_predicts +=  response + '\n'
            j += 1
        except:
            j += 1
            continue
    prd_path.append(len(sub_questions))
    prompts = "I have serval response of diverse set of reasoning paths of this question, each are devide with '\n'." + sub_predicts + 'Choose the majority response to answer the below question '
    response = cot(method="zero_shot", question=prompts + question,
                            question_temp=question_temp)
    predicts.append(response)
    real_answers.append(answers[i])
    real_answers2.append(answers2[i])
    i +=  1
    #print(i)
    #choice = extract_answer(response)

    
from rouge_score import rouge_scorer
def calculate_rouge_l(string1, string2):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(string1, string2)
    return scores['rougeL'].fmeasure * 100
def get_max_rouge_l(A, B, C):
    rouge_l_ab = calculate_rouge_l(A, B)
    rouge_l_ac = calculate_rouge_l(A, C)
    return max(rouge_l_ab, rouge_l_ac)

print(len(predicts))
print(len(real_answers))
print(len(real_answers2))

result = []
for i in range(len(predicts)):
    result.append(get_max_rouge_l(predicts[i], real_answers[i], real_answers2[i]))

print('Average Rouge-L:', sum(result)/len(result))



# %%
# Calculate accuracy

print('done')


#     prompts = "I have the answer of the problem" + choice + '. And provide me the choice from question. Provide A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P only'
#     choice = cot(method="zero_shot", question=prompts + questions[i],
#                             question_temp=question_temp)
#     print('choice: ',choice, 'true_label: ', str(answers[i]))
#     print('Now is number: ', total)
#     temp.append('choice: '+choice+'true_label: '+ str(answers[i]))
#     template_judge = Template("""
#     If the meaning of the answer "$ANSWER" is similar or equal to the meaning of "($AN)", is this statement true? If true, reply [True], if false , reply [False]  
#     """).substitute(ANSWER= choice, AN = str(answers[i])) 
#     # print(template_judge)
#     try:
        
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
#         try:
            
#             judge_result = check_result(query=template_judge)
#             print('judge_result: ', judge_result)
#             if judge_result.find('True') != -1:
#                 true += 1
#             total += 1
#             ii = 0
#             i += 1
#             pre.append([response, judge_result])  
#             print('accuracy: ', true / total * 100, '%')
#         except:
#             i += 1
#             continue

# print(temp)

# # args.direct_answer_trigger_for_fewshot = "The answer is"
# # predict = []
# # pattern = r'\((.)\)'  # Matches a single character inside parentheses
# # for pred in predicts:
# #     # match = re.search(pattern, pred)
# #     # if match:
# #     #     result = match.group(1)
# #     #     predict.append(result)
# #     # else:
# #     #     print(pred)
# #     pred = pred.split('.')[0]
# #     pred = answer_cleansing(args, pred, must_choice=True)
# #     predict.append(pred)


# # final_predictions = [1 for true, pred in zip(
# #     real_answers, predict) if str(true) == str(pred)]
# # correct_predictions = sum(final_predictions)

# # # Calculate accuracy
# # accuracy = correct_predictions / len(real_answers) * 100
# # final_predictions = pd.DataFrame(final_predictions)
# # print(final_predictions)

# # print(f"Accuracy: {accuracy}%")




# # %%
# # Calculate accuracy
# print('done')
# # %%
