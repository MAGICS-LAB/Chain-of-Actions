# %%
import openai
import os
from langchain_community.llms import OpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain.agents import load_tools
from langchain.tools import BaseTool, StructuredTool, Tool, tool
from langchain.agents import initialize_agent
from dotenv import load_dotenv
load_dotenv()
from langchain.agents import AgentExecutor
from langchain.agents.output_parsers import ReActJsonSingleInputOutputParser
from string import Template
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain_community.chat_models import ChatOpenAI

from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain import hub
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.tools.render import render_text_description
import pandas as pd
import sys
import numpy as np
from langchain_community.callbacks.manager import get_openai_callback
from openai import OpenAI

sys.argv = ['']
sys.path.append('/root/autocot/')


from utils import *
from api import cot
import re
import time
import argparse
from langchain import hub
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.tools.render import render_text_description
from langchain.agents import AgentExecutor

# load API keys; you will need to obtain these if you haven't yet
os.environ["OPENAI_API_KEY"] = 'sk-proj-xxxx-xxxx'
os.environ["GOOGLE_API_KEY"] = 'xxxx'
os.environ["GOOGLE_CSE_ID"] = "xxxx"

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
 

search = GoogleSearchAPIWrapper()

tool = Tool(
    name="Google Search",
    description="Search Google for recent results.",
    func=search.run,
)

llm = ChatOpenAI(model_name="gpt-3.5-turbo" ,temperature=0)

tools = load_tools(["llm-math"], llm=llm)
tools.append(tool)



# e4 social
#47326 truth


#/root/datasets/BIG-bench/bigbench/benchmark_tasks/truthful_qa/task_mc.json

parser = argparse.ArgumentParser(description="Zero-shot-CoT")
parser.add_argument("--dataset", type=str, default='strategyqa',
                    help="maximum number of workers for dataloader")
parser.add_argument("--dataset_path", type=str, default='/home/hlv8980/datasets/strategy/task.json',
                    help="maximum number of workers for dataloader")
parser.add_argument("--method", type=str, default='zero_shot',
                    help="maximum number of workers for dataloader")


args = parser.parse_args()

# data = pd.read_csv('/home/hlv8980/datasets/fever/subset_fever.csv')


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


# data = pd.read_csv('/home/hlv8980/experiment_datasets/webqa.csv')

data = pd.read_json('/home/hlv8980/Cov-COA/datasets/topiocqa/dev.json')
print(data.columns)
count = 0
for i in range(len(data)):
    if Turn_no == 0:
        count += 1
    if count > 20:
        break
    
exit(0)


# # %%
# questions = data['question'].to_list()


# # # %%
# answers_temp = data['answers'].to_list()
# answers = []
# for item in answers_temp:
#   item = item[2:-2]
#   temp = item.split(',')
#   answers.append([e.strip(' ').replace("'","") for e in temp])

# data = pd.read_csv('/home/hlv8980/dd/final_data_questions_100.csv')



# # %%
# questions = data['input'].to_list()


# # %%
# answers = data['answer1'].to_list()


# answers2 = data['answer2'].to_list()




# args.dataset = 'strategyqa'
# args.dataset_path = '/home/hlv8980/datasets/strategy/task.json'
# questions, answers = data_reader(args)

# questions_final.extend(questions[:3])
# answers_final.extend(answers[:3])

# args.dataset = 'bigbench_social'
# args.dataset_path = '/home/hlv8980/datasets/socialQA/task.json'
# questions, answers = data_reader(args)

# questions_final.extend(questions[:3])
# answers_final.extend(answers[:3])

# args.dataset = 'bigbench_truth'
# args.dataset_path = '/home/hlv8980/datasets/truthful_qa/task_mc.json'
# questions, answers = data_reader(args)

# questions_final.extend(questions[:3])
# answers_final.extend(answers[:3])

# args.dataset = 'bigbench_general'
# args.dataset_path = '/home/hlv8980/datasets/General QA/task.json'
# questions, answers = data_reader(args)

# questions_final.extend(questions[:3])
# answers_final.extend(answers[:3])



# i = 0

# question_temp = [questions[:3], answers[:3]]
# answers = answers[3:]
# questions = questions[3:]

# template = []




prompt = hub.pull("hwchase17/react")
prompt = prompt.partial(
    tools=render_text_description(tools),
    tool_names=", ".join([t.name for t in tools]),
)

llm_with_stop = llm.bind(stop=["\nObservation"])




agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_log_to_str(x["intermediate_steps"]),
    }
    | prompt
    | llm_with_stop
    | ReActSingleInputOutputParser()
)


i = 0 

predicts = []
real_answers = []
real_answers2 = []
j = 0
true = 0
total = 0
args.direct_answer_trigger_for_fewshot = "The answer is"
predict = []
temp = []
pre = []
api = []
tokens = []
before = []
after = []
current_time = time.time()
while i < 20:
    count = 0
    # if i % 10 != 0:
    #     i += 1
    #     continue
    try:
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True,  stream_runnable=False)
        #print(i)
        # response = agent_executor.invoke(
        #         {
        #             "input": questions[i]
        #         }
        #     )
        # print(response.usage_metadata)
        with get_openai_callback() as cb:
            response = agent_executor.invoke(
                {
                    "input": questions[i]
                }
            )
            print(f"Total Tokens: {cb.total_tokens}")
            print(f"Prompt Tokens: {cb.prompt_tokens}")
            print(f"Completion Tokens: {cb.completion_tokens}")
            print(f"Total Cost (USD): ${cb.total_cost}")
            tokens.append(cb.total_tokens)
            before.append(cb.prompt_tokens)
            after.append(cb.completion_tokens)
            #count += cb.successful_requests
            
        
        prompts = "I have the answer of the problem" + response['output'] + '. And provide me the choice from question '
        response = cot(method="zero_shot", question=prompts + questions[i],
                                question_temp=question_temp)
        count += 1
        predicts.append(response)
        real_answers.append(answers[i])
        # choice = extract_answer(response)
        # predicts.append(choice)
        #real_answers2.append(answers2[i])
        i += 1
        
    except:
        i += 1
        continue

print((time.time() - current_time)/count)
# from rouge_score import rouge_scorer
# def calculate_rouge_l(string1, string2):
#     scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
#     scores = scorer.score(string1, string2)
#     return scores['rougeL'].fmeasure * 100
# def get_max_rouge_l(A, B, C):
#     rouge_l_ab = calculate_rouge_l(A, B)
#     rouge_l_ac = calculate_rouge_l(A, C)
#     return max(rouge_l_ab, rouge_l_ac)


# print(len(predicts))
# print(len(real_answers))
# print(len(real_answers2))


# result = []
# for i in range(len(predicts)):
#     result.append(get_max_rouge_l(predicts[i], real_answers[i], real_answers2[i]))

# print('Average Rouge-L:', sum(result)/len(result))