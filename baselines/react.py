# %%
import openai
import os
from langchain.llms import OpenAI
from langchain.agents import load_tools
from langchain.tools import BaseTool, StructuredTool, Tool, tool
from langchain.agents import initialize_agent
from dotenv import load_dotenv
load_dotenv()
from langchain.agents import AgentExecutor
from langchain.agents.output_parsers import ReActJsonSingleInputOutputParser
from string import Template
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.chat_models import ChatOpenAI

from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.utilities import GoogleSearchAPIWrapper
from langchain import hub
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.tools.render import render_text_description
import pandas as pd
import sys
import numpy as np
from langchain.callbacks import get_openai_callback
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
os.environ["OPENAI_API_KEY"] = 'sk-xxxxx'
os.environ["GOOGLE_API_KEY"] = 'xxxxx'
os.environ["GOOGLE_CSE_ID"] = "xxxx"

def check_result(query, model = 'gpt-3.5-turbo', max_tokens=1000, temperature=0):
  message = []
  message.append({'role': 'user', 'content': query})  
  completion = openai.ChatCompletion.create(model = model, 
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
parser.add_argument("--dataset_path", type=str, default='/root/datasets/strategy/task.json',
                    help="maximum number of workers for dataloader")
parser.add_argument("--method", type=str, default='zero_shot',
                    help="maximum number of workers for dataloader")


args = parser.parse_args()


# data = pd.read_csv('/root/experiment_datasets/webqa.csv')



# # %%
# questions = data['question'].to_list()


# # # %%
# answers_temp = data['answers'].to_list()
# answers = []
# for item in answers_temp:
#   item = item[2:-2]
#   temp = item.split(',')
#   answers.append([e.strip(' ').replace("'","") for e in temp])

questions, answers = data_reader(args)

# %%
question_temp = [questions[:3], answers[:3]]
answers = answers[3:]
questions = questions[3:]




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
j = 0
true = 0
total = 0
args.direct_answer_trigger_for_fewshot = "The answer is"
predict = []
temp = []
pre = []
api = []
while i < 20:
    count = 0
    # if i % 10 != 0:
    #     i += 1
    #     continue
    try:
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
        print(i)
        
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
            
            count += cb.successful_requests
            
        
        prompts = "I have the answer of the problem" + response['output'] + '. And provide me the choice from question '
        response = cot(method="zero_shot", question=prompts + questions[i],
                                question_temp=question_temp)
        count += 1
        predicts.append(response)
        real_answers.append(answers[i])
        choice = extract_answer(response)
        predicts.append(choice)
        real_answers.append(answers[i])
        
        prompts = "I have the answer of the problem" + choice + '. And provide me the choice from question. Provide A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P only'
        choice = cot(method="zero_shot", question=prompts + questions[i],
                                question_temp=question_temp)
        count += 1
        print('choice: ',choice, 'true_label: ', str(answers[i]))
        print('Now is number: ', total)
        temp.append('choice: '+choice+'true_label: '+ str(answers[i]))
        template_judge = Template("""
        If the meaning of the answer "$ANSWER" is similar or equal to the meaning of "($AN)", is this statement true? If true, reply [True], if false , reply [False]  
        """).substitute(ANSWER= choice, AN = str(answers[i])) 
        # print(template_judge)
        try:
            
            judge_result = check_result(query=template_judge)
            print('judge_result: ', judge_result)
            if judge_result.find('True') != -1:
                true += 1
            total += 1
            ii = 0
            i += 1
            count += 1
            pre.append([response, judge_result])  
            api.append(count)
            print('accuracy: ', true / total * 100, '%') 
        except:
            try:
                
                judge_result = check_result(query=template_judge)
                print('judge_result: ', judge_result)
                if judge_result.find('True') != -1:
                    true += 1
                total += 1
                ii = 0
                i += 1
                count += 1
                pre.append([response, judge_result])  
                api.append(count)
                print('accuracy: ', true / total * 100, '%')
            except:
                i += 1
                continue
    except:
        i += 1
        continue
    
print(np.mean(np.array(api)))  
print(temp)