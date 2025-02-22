# %%
from openai import OpenAI
import openai
# from helicone.globals import helicone_global

from googleapiclient.discovery import build
from sklearn.metrics.pairwise import cosine_similarity
from langchain.document_loaders import UnstructuredURLLoader

import asyncio
from string import Template
import pandas as pd
import numpy as np
import time
import json
import logging
import math
import tiktoken
import requests

def calculate_metrics(generated_text, reference_text):
    """
    Returns:
        tuple: (precision, recall, awl)
    """
    # Tokenize the strings
    A_tokens = generated_text.split()
    R_tokens = reference_text.split()

    # Find relevant tokens
    relevant_tokens = set(A_tokens).intersection(R_tokens)

    # Calculate Precision
    precision = len(relevant_tokens) / len(A_tokens) if A_tokens else 0

    # Calculate Recall
    recall = len(relevant_tokens) / len(R_tokens) if R_tokens else 0

    # Calculate Average Word Length (AWL)
    awl = sum(len(word) for word in A_tokens) / len(A_tokens) if A_tokens else 0

    return 0.3 * precision + 0.3 * recall + 0.4 * awl


def retrieve_top_k(query_item, k=1, api_url="http://172.20.213.2:8893/api/search"):
    url = f"{api_url}?query={query_item}&k={k}"
    try:
        response = requests.get(url=url)
        response.raise_for_status()  
        res_dic = response.json()
        return res_dic.get('topk', [])  
    except requests.exceptions.RequestException as e:
        print(f"Error during retrieval: {e}")
        return []

# retrieved_docs = retrieve_top_k(query_item=query, k=1)


GOOGLE_API_KEY="xxxx"
GOOGLE_ENGINEER_ID="xxx"
OPENAI_API_KEY="sk-proj-xxxxx"
# HELICONE_API_KEY="sk-xxxxx"
# AZURE_OPENAI_KEY="xxxxx"
# AZURE_OPENAI_BASE="https://faq-ai.openai.azure.com/"
# AZURE_DEPLOYMENT_NAME="faq-ai-3' | 'faq-ai-35' | 'faq-ai-embedding"
# AZURE_OPENAI_VERSION="2023-06-01-preview"
# helicone_global.api_key = HELICONE_API_KEY

googleService = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)
client = OpenAI()  

def split_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
def setOpenAi():
  openai.api_type = "open_ai"
  openai.api_key = OPENAI_API_KEY
  openai.api_base = "https://api.openai.com/v1"
  openai.api_version = None
  
  
def setAzure():
  # openai.api_type = "azure"
  # openai.api_key = AZURE_OPENAI_KEY
  # openai.api_base = AZURE_OPENAI_BASE
  # openai.api_version = AZURE_OPENAI_VERSION
  pass


def check_result(query, model = 'gpt-4o-2024-11-20', max_tokens=1000, temperature=0):
  message = []
  message.append({'role': 'user', 'content': query})  
  completion = client.chat.completions.create(model = model, 
                                            messages = message, 
                                            temperature = temperature,
                                            max_tokens = max_tokens)
  return completion.choices[0].message.content
  
def truncateTextTokens(content, max_tokens):
  encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
  exact = encoding.encode(content)[:max_tokens]
  exact_text = encoding.decode(exact)
  return exact_text  
  
def num_tokens_from_string(string: str):
  encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
  num_tokens = len(encoding.encode(string))
  return num_tokens

def embeddings(input: list[str], useAzure = False) -> list[list[str]]:
    response = openai.Embedding.create(model="text-embedding-ada-002", input=input)
    return [data.embedding for data in response.data]
  
def embeddings_content(content):
    nums = num_tokens_from_string(content)
    if nums > 8190:
      encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
      tokens = encoding.encode(content)
      token_chunks = list(split_list(tokens, 8190))  
      chunkTexts = [encoding.decode(chunk) for chunk in token_chunks]
      lens = [len(chunk) for chunk in token_chunks]
      emb_chunks = embeddings(chunkTexts)
      average_emb_content = sum([np.array(emb_chunk) / length for emb_chunk, length in zip(emb_chunks,lens)]).tolist()
      return average_emb_content
    else:
      return embeddings(content)[0]

  
def getSimilarities(hypo_answer, strings):
  emb_strings = embeddings(strings)
  emb_hypo_answer = embeddings(hypo_answer)
  cosine_similarities = cosine_similarity(emb_strings, emb_hypo_answer).flatten()
  return cosine_similarities

def getSimilaritiesContents(hypo_answer, emb_contents):
  emb_hypo_answer = embeddings(hypo_answer)
  cosine_similarities = cosine_similarity(emb_contents, emb_hypo_answer).flatten()
  return cosine_similarities
  
def onceChatCompletion(logger, query, model = 'gpt-3.5-turbo', max_tokens=1000, system='', jsonResponse=False, useAzure=False, temperature=0):
  message = []
  if jsonResponse:
    message.append({'role':'system', 'content': 'Output only valid LIST'})
  if system != '':
    message.append({'role':'system', 'content': system})
  message.append({'role': 'user', 'content': query})
  if useAzure:
    setAzure()
    completion = client.chat.completions.create(messages=message,deployment_id="faq-ai-35")    
  else:
    setOpenAi()
    completion = client.chat.completions.create(model = model, 
                                            messages = message, 
                                            temperature = temperature,
                                            max_tokens = max_tokens)

  return completion.choices[0].message.content, completion.usage.prompt_tokens, completion.usage.completion_tokens, completion.usage.total_tokens
  
# def hypotheticalAnswer(logger, query, temperature = 0.5, max_tokens = 500, model = 'gpt-3.5-turbo'):
#   logger.info(f'hypothetical answer: {query}')
#   template_text = Template("""
#         Generate a hypothetical answer related to *WEB3* or *CRYPTO* to the user's question. This answer will be used to rank search results. 
#         Pretend you have all the information you need to answer, but don't use any actual facts. Instead, use placeholders
#         like NAME did something, or NAME said something at PLACE. 

#         User question: $USER_QUESTION
#         ANSWER:
#       """).substitute(USER_QUESTION=query)
#   result = onceChatCompletion(logger=logger,query=template_text, temperature=temperature, max_tokens=max_tokens, model=model)
#   logger.info(f'hypothetical answer result: {result}')
#   return result 

# def extractQueries(logger, query, limit=5, jsonResponse=True, system = '', temperature = 0.5, max_tokens = 500, model = 'gpt-3.5-turbo'):
#   setOpenAi()  
#   logger.info(f"extract queries by: {query}")
#   limit = limit
#   template_text = Template("""
#       You have access to a search API that returns recent news articles.
#       Generate an array of search queries that are relevant to this question. 
#       If the context is not specific (e.g., whether it is a 'project' or something else), try to be as general as possible but avoid standalone keywords that are prone to ambiguity.
#       Also, avoid queries that are too long or too short.
#       Context should wrap in double quotes, e.g., "project name".
#       Use a variation of related keywords and phrases, aiming to cover a broad range of possibilities.
#       Include as many queries as you can think of, but aim to remove or refine those that are too ambiguous without context.
#       For example, include queries like ['"context" keyword_1 keyword_2', '"context" keyword_1', '"context" keyword_2'] but post-process to remove or refine ambiguous terms and context must be provided. Do not include query like '"keyword_1 context keyword_2"'
#       Be creative. The more queries you include, the more likely you are to find relevant results.

#       Response only top $LIMIT queries.
      
#       User question: $USER_QUESTION
      
#       {{{{raw}}}}
#       Format: {{"queries": ["query_1", "query_2", "query_3"]}}
#       {{{{/raw}}}}
#   """).substitute(LIMIT=limit, USER_QUESTION=query) 
#   result = onceChatCompletion(logger=logger,query=template_text, temperature=temperature, max_tokens=max_tokens, model=model)
#   logger.info(f'extracted queries result: ${result}');
#   return json.loads(result)



# def extract_sub_queries(logger, query, limit = 10, jsonResponse = True, system = '', temperature = 0.5, max_tokens = 500, model = 'gpt-3.5-turbo'):
#   setOpenAi()    
#   logger.info(f"extract queries by: {query}")
#   template_text = Template("""
#   You have access to a google search API that returns articles.

#   According to the question, analyze what basic information is needed to answer the question and convert this information into sub-questions.

#   Split then question into multiple sub-questions, Each sub-question should be independent of each other, avoid generating multiple similar questions.
#   If multiple entities are mentioned in the question, each entity should be split into a separate question.

#   Each sub-question should have a length and format that is suitable for use as a Google search keyword.

#   For example, when the question is “When will the bull market come?”, you need to know “what is a cryptocurrency bull market?”, “What are the characteristics of a bull market?”, and “How was the cryptocurrency market situation in the past year?“

#   Output only the keywords array.
#   OUTPUT Format: ["Keyword1", "keyword2", ...]

#   Limit: Response only top $LIMIT queries.

#   User question: $USER_QUESTION

#   {{{{raw}}}}
#   Format: ["query_1", "query_2", "query_3"]
#   {{{{/raw}}}}
#   """).substitute(LIMIT=limit, USER_QUESTION=query)  
#   result,  = onceChatCompletion(logger=logger,query=template_text, temperature=temperature, max_tokens=max_tokens, model=model)
#   logger.info(f'extracted sub queries result: ${result}');
#   return json.loads(result)


# def extractSearchKeywords(logger, query, model = 'gpt-3.5-turbo'):
#   setAzure()    
#   template_text = Template("""
#       According to the user’s question({$question}), provide the configuration for calling the Google Search API in the following format:
#       '''
#       {
#           "keyword": the plain-text english sentence you will input into Google to answer the question,
#           "date_restrict": Set the time range limit for search results based on the user's query, the format is y[n] for n years, w[n] for n weeks, d[n] for n days, m[n] for n months, for example, y1 means search results within 1 year, d7 means search results within 7 days, and so on. If the user's query does not contain a time range, the default is m3.
#       }
#       '''
      
#       RULES:
#       '''
#       - Don't directly answer the question, just make the google search keyword.
#       - Do not include punctuation marks in the keyword.
#       - If you cannot generate a Google search keyword based on the above question, please directly output the original question in the "keyword".
#       - **The keyword should only contain the information that users want to know, without any requirements regarding format, language, time, etc.**
#       - **When users mention “latest” or “recent”, the date_restrict should be limited to within the m1 or w1.**
#       '''
      
#       Output:
#   """).substitute(question=query)  
  
#   result, prompt_token, completion_token, total_token = onceChatCompletion(logger = logger, query = template_text, jsonResponse=True, useAzure = True, model = model)
  
#   queries = extract_sub_queries(logger = logger, query = query,limit=5)
#   queries.append(json.loads(result)['keyword'])
#   num = min(10, max(3, math.floor(30 / len(queries))))  
#   return {"queries":queries, "date_restrict":json.loads(result)['date_restrict'], "num":num}


def create_summary(content, question):
  t = time.asctime()
  qu = Template("""
            The relevant content below should be used to generate a best answer of the user's question shortly.
            question is:
            '''
            $question
            '''
            
            relevant content:
            '''
            $content
            '''    
  """).substitute(content = content, question = question) 
  result, prompt_token, completion_token, total_token = onceChatCompletion(logger=logger, query=qu,model="gpt-3.5-turbo-16k")
  return result, prompt_token, completion_token, total_token 



import re
# from utils import *
logging.basicConfig(level=logging.ERROR,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(),
                              logging.FileHandler('search.log')])
logger = logging.getLogger("main")






# %%
def get_answer(question, true_answer, k, successful_request):
  prompt_tokens, completion_tokens ,total_tokens = 0, 0, 0
  template_text = Template("""
  Construct an action reasoning chain for this complex [Question]: “$QUESTION”. For each step of the reasoning chain, choose an action from three choses: [Web-querying Engine(search real-time news or new words), Information-analyzing Engine (search existing domain information in local knowledge base) , Data-analyzing Engine (query real-value data and calculate some results) ] as value of element "Action", and also generate a sub-question for each action to get one of [web-search keywords, needed information description, data features description] as value of element "Sub". Also, generate answer for each Sub as value of element "Guess Answer". 
  You need to try to generate the final answer for the [Question] by referring to the "Action"-"Sub"-"Guess Answer" tuples in "Chain", as the value of element "Final Answer". By the way, forgive the three actions choses, and only use Web-querying engine and Information-analyzing Engine.
    For example:

  {
    "Question": "Is it good to invest in bitcoin now? A. It is a good time. B. It is not a good time.",
    "Chain": [
      {
        "Action": "Information-analyzing engine",
        "Sub": "what is bitcoin",
        "Guess_Answer": "Bitcoin is one of the cryptocurrencies."
      },
      {
        "Action": "Data-analyzing Engine",
        "Sub": "What is the recent price trend of bitcoin?",
        "Guess_Answer": "The price of Bitcoin increases from XX to YY…"
      },
      {
        "Action": "Web-querying Engine",
        "Sub": "bitcoin news",
        "Guess_Answer": "One news shows that ..."
      }
    ],
    "Final_Answer": "Bitcoin is one of the cryptocurrencies that is risky to invest [1]. And its price become more and more high recently [2]. Also, there are lot of news to promote Bitcoin. So, it is a good time to invest in Bitcoin now."
  }

  """).substitute(QUESTION=question) 

  result, prompt_token, completion_token, total_token = onceChatCompletion(logger=logger,query=template_text)
  #successful_request += 1
  prompt_tokens += prompt_token
  completion_tokens += completion_token
  total_tokens += total_token
  print(result)
  json_result = json.loads(result)
  for i in range(len(json_result['Chain'])):
    single_step = json_result['Chain'][i]
    action = single_step['Action']
    print('action is:', action)
    sub_question = single_step['Sub']
    guess_answer = single_step['Guess_Answer']
    if action == 'Web-querying Engine':
      pages = (googleService.cse().list(q=sub_question, cx=GOOGLE_ENGINEER_ID).execute())['items'][:k]
      successful_request += 1
      urls = [page['link'] for page in pages]
      loader = UnstructuredURLLoader(urls=urls)
      datas = loader.load()
      contents = [content.page_content for content in datas]
      individual_results = ''
      mrfs = []
      for qq in range(len(contents)):
        print('contents is ', contents)
        print('content is', contents[qq])
        mrfs.append(calculate_metrics(contents[qq], true_answer))
      print('--------------------------------')
      print('mrf is:', max(mrfs))
      print('--------------------------------')
      if max(mrfs) > 2:
         continue
      for j in range(len(contents)):
        try:
          c_f = contents[j]
          datil, prompt_token, completion_token, total_token = create_summary(c_f, question)
          #successful_request += 1
          individual_results = individual_results + '\n\n---\n\n---\n\n' + datil
          prompt_tokens += prompt_token
          completion_tokens += completion_token
          total_tokens += total_token
        except:
          continue
      final_sub_result, prompt_token, completion_token, total_token = create_summary(individual_results, question)
      #successful_request += 1    
      prompt_tokens += prompt_token
      completion_tokens += completion_token
      total_tokens += total_token

      # print('original guess answer is:' , guess_answer)
      json_result['Chain'][i]['Guess_Answer'] = final_sub_result
      # print('changed guess answer is:', final_sub_result)
    elif action == 'Information-analyzing Engine':
      top_k_result = retrieve_top_k(sub_question, k=1)
      # print('test 1', top_k_result)
      mrfs = calculate_metrics(top_k_result[0]['text'], true_answer)
      # print('test 2')
      print('--------------------------------')
      print('mrf is:', mrfs)
      print('--------------------------------')
      if mrfs > 2:
         continue
      else:
        final_sub_result, prompt_token, completion_token, total_token = create_summary(top_k_results, question)
        json_result['Chain'][i]['Guess_Answer'] = final_sub_result
  #   else if action == 'Data-analyzing engine':
    else:
      print('action is:', action)
      print('error about the sub action, pls double check')
      continue

  template_final = Template("""
  In the following reasoning chain, based on the Sub-question (Sub) and Guess_Answer, answer the question [Question]: "$QUESTION" and give me the final_answer. ---Reason Chain --- "$CHAIN" ---End of Chain---
  """).substitute(CHAIN= str(json_result['Chain']), QUESTION = question) 
  final_response, prompt_token, completion_token, total_token  = onceChatCompletion(logger=logger,query=template_final)
  
  #successful_request += 1
  prompt_tokens += prompt_token
  completion_tokens += completion_token
  total_tokens += total_token

  template_judge = Template("""
  For question [Question]: "$QUESTION". Judge if the given answer [Answer]: "$ANSWER" is similar or cover the meaning of given choice [Choice]: "$CHOICE". If yes, reply [Yes]. If not, reply [No].
  """).substitute(ANSWER= final_response, CHOICE= true_answer, QUESTION = question) 
  # print('after actions:', json_result)
  # print('final answer:', final_response)
  # print('true answer:', true_answer)
  judge_result, _, _, _ = onceChatCompletion(logger=logger,query=template_judge)
  return final_response, judge_result, successful_request, prompt_tokens, completion_tokens ,total_tokens

# %%


count = 0
real_answers = []
predicts = []
pre = []
not_first = False
tokens = []
befores = []
final_responses = []
judge_results = []
afters = []
total = 0
true = 0
correct = 0
total2 = 0.000001
successful_requests = []

data_csv = pd.read_csv('/projects/p32364/coa/topiocqa/test.csv')
data = pd.read_json('/projects/p32364/coa/topiocqa/dev.json')
#data = pd.read_json('/home/hlv8980/Cov-COA/datasets/qrecc/test.json')
current_time = time.time()

for i in range(len(data)):
  record = data.iloc[i]
  if record['Turn_no'] == 1:
      count += 1
      if not_first:
          real_answers.append(answer)
          predicts.append(final_responses)
          tokens.append(token)
          befores.append(before)
          afters.append(after)
          successful_requests.append(successful_request)
      not_first = True
      token = 0
      before = 0
      after = 0
      logs = ''
      successful_request = 0
  # if count > 2:
  #     break
  question = data_csv.iloc[i]['rewrite_question']
  #question = record['Rewrite']
  answer = record['Answer']
  # print('now is conversition: ', count, ', question content:', question, '; answer is: ', answer)
  try:
    final_response, judge_result, successful_request, prompt_tokens, completion_tokens ,total_tokens = get_answer(question, answer, 2, successful_request)
    final_responses.append(final_response)
    judge_results.append(judge_result)
    # print('judge_result: ', judge_result , '!!!!')
    if judge_result.find('Yes') != -1:
      correct += 1
    total += 1
    print('now the acc is: ', correct / total)
    before += prompt_tokens
    after += completion_tokens
    token += total_tokens
    
    checker_template = Template("""
      If the answer "$ANSWER" is contradict with details in previous conversation "$AN", is this statement true? If true, reply [True], if false , reply [False]  
      """).substitute(ANSWER= final_response, AN = logs)
      #try:
    judge_result = check_result(query=checker_template)
    print('judge_result: ', judge_result)
    if judge_result.find('True') != -1:
        true += 1
        total2 += 1
        pre.append([final_response, judge_result])  
        print('contradict rate: ', true / total2 * 100, '%') 
    else:
      total2 += 1
    logs += 'Question: ' + question + " Response: " + final_response
  except:
    pass


count -= 1
print('time:',(time.time() - current_time)/count)
print('tokens:',np.mean(tokens))
print('after:',np.mean(afters))
print('before:',np.mean(befores))
print('SR:',np.mean(successful_requests))
print(count)
print('contradict rate: ', true / total2 * 100, '%') 


for i in range(count):
    template_judge = Template("""
    If the meaning of the answer "$ANSWER" is similar or equal to the meaning of "$AN", is this statement true? If true, reply [True], if false , reply [False]  
    """).substitute(ANSWER= predicts[i], AN = str(real_answers[i])) 
    # print(template_judge)
    try:
        judge_result = check_result(query=template_judge)
        count += 1
        print('judge_result: ', judge_result)
        if judge_result.find('True') != -1:
            true += 1
            total += 1
            ii = 0
            pre.append([ret, judge_result])  
            print('accuracy: ', true / total * 100, '%') 
    except:
        time.sleep(10)
        try:
            judge_result = check_result(query=template_judge)
            count += 1
            print('judge_result: ', judge_result)
            if judge_result.find('True') != -1:
                true += 1
                total += 1
                ii = 0
                pre.append([ret, judge_result])  
                print('accuracy: ', true / total * 100, '%')
        except:
            continue




