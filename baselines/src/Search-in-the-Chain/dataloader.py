import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from datasets import load_dataset, Dataset
#from OPENAI 
import pandas as pd
import time
import requests
from bs4 import BeautifulSoup

def get_wiki_content(url):
    # Send a GET request to the Wikipedia page
    try:
      response = requests.get(url)
      
      # Check if the request was successful
      if response.status_code == 200:
          # Parse the content of the page using BeautifulSoup
          
        soup = BeautifulSoup(response.text, 'html.parser')
        text_list = [p.text for p in soup.find_all('p')]
        return text_list
      else:
        # If the request was not successful, return None or raise an exception
        return None
    except Exception as e:
      time.sleep(10)



class coverational_dataset(Dataset):
  def __init__(self, datapath, flag, max_length=512):
    self.datapath = datapath
    self.flag = flag

    # Load the dataset from the CSV file
    self.dataset = load_dataset('csv', data_files={'data': self.datapath + '/' + self.flag + '.csv'})

  def __len__(self):
    return len(self.dataset['data'])

  def __getitem__(self, idx):
    item = self.dataset['data'][idx]
    return item



class coverational_dataloader(DataLoader):
  def __init__(self, datapath, flag, batch_size, max_length=512):
    self.datapath = datapath
    self.flag = flag
    self.batch_size = batch_size
    self.max_length = max_length
    self.dataset = coverational_dataset(self.datapath, self.flag, self.max_length)
    super().__init__(self.dataset, batch_size=self.batch_size, shuffle=True)
  
  def __len__(self):
    return len(self.dataset)
  

class Prepocress:
  def __init__(self, datapath, flag,data_type = 'topicqa'):
    self.datapath = datapath
    self.flag = flag
    if data_type == 'orconvqa':
      self.dataset = load_dataset('json', data_files={'train': self.datapath + '/' + self.flag + '.txt'})
    else:
      self.dataset = load_dataset('json', data_files={'train': self.datapath + '/' + self.flag + '.json'})
    self.openai = OpenAILLM('gpt-3.5-turbo-0125', openai_key)
    self.data_type = data_type
    
  def preprocess(self):
    data = {'rewrite_question': [], 'answer': [], 'context': []}
    count = 0
    for i in range(len(self.dataset['train'])):
      item = self.dataset['train'][i]
      #print(item)
      question = ''
      if self.data_type == 'topicqa':
        for j in range(len(item['Context'])):
          question += item['Context'][j] + ' '
        question += item['Question']
        rewrite_queation = self.openai.generate(question)[0]
        data['rewrite_question'].append(rewrite_queation)
        data['answer'].append(item['Answer'])
        data['context'].append(item['Rationale'])
      elif self.data_type == 'qrecc':
        print(count)
        data['rewrite_question'].append(item['Rewrite'])
        data['answer'].append(item['Answer'])
        #print(item['Answer_URL'])
        data['context'].append(item['Answer_URL'])
        count += 1
      elif self.data_type == 'orconvqa':
        data['rewrite_question'].append(item['rewrite'])
        data['answer'].append(item['answer']['text'])
        data['context'].append(item['evidences'])
    df = pd.DataFrame(data)
    return df

 
# topicaqa_preprocess = Prepocress('/home/hlv8980/Cov-COA/datasets/topiocqa', 'dev', 'topicqa')
# topicaqa = topicaqa_preprocess.preprocess()
# topicaqa.to_csv('/home/hlv8980/Cov-COA/datasets/topiocqa/test.csv', index=False)
# qrecc_preprocess = Prepocress('/home/hlv8980/Cov-COA/datasets/qrecc', 'test', 'qrecc')
# qrecc = qrecc_preprocess.preprocess()
# qrecc.to_csv('/home/hlv8980/Cov-COA/datasets/qrecc/test.csv', index=False)
# ORConvqa_preprocess = Prepocress('/home/hlv8980/Cov-COA/datasets/ORConvQA/preprocessed', 'test', 'orconvqa')
# ORConvqa = ORConvqa_preprocess.preprocess()
# ORConvqa.to_csv('/home/hlv8980/Cov-COA/datasets/ORConvQA/preprocessed/test.csv', index=False)
# topicaqa = pd.read_csv('/home/hlv8980/Cov-COA/datasets/topiocqa/test.csv')
# ORConvqa = pd.read_csv('/home/hlv8980/Cov-COA/datasets/ORConvQA/preprocessed/test.csv')
# total = pd.concat([topicaqa , ORConvqa])
# total.to_csv('/home/hlv8980/Cov-COA/datasets/test.csv', index=False)

# data = pd.read_csv('/home/hlv8980/Cov-COA/datasets/train.csv')
# for i in range(len(data['rewrite_question'])):
#   if data['rewrite_question'][i][0] == '[':
#     data['rewrite_question'][i] = data['rewrite_question'][i][2:-2]



# for i in range(len(data['context'])):
#   if str(data['context'][i])[0] == '[':
#     data['context'][i] = data['context'][i][2:-2]


# data.to_csv('/home/hlv8980/Cov-COA/datasets/train.csv', index=False)

