import openai
import json
import os
import string
import regex
import time
from collections import Counter
import joblib
from tqdm import tqdm
import pandas as pd
import sys
sys.argv = ['']
sys.path.append('/root/autocot')
from string import Template
import numpy as np

from utils import *
from api import cot
import re
import time
import argparse
import openai

import socket
HOST = '127.0.0.1'
PORT = 50007
openai.api_key = 'sk-xxxx'

def excute(args,start_idx):
    questios, answers = data_reader(args)
    for k in range(len(questios)):
        if k < start_idx:
            continue
        print(k)
        q = questios[k]
        answer = answers[k]
        round_count = 0
        message_keys_list = [{"role": "user", "content":
                    """Construct a global reasoning chain for this complex [Question] : " {} " You should generate a query to the search engine based on
                        what you already know at each step of the reasoning chain, starting with [Query].
                        If you know the answer for [Query], generate it starting with [Answer].
                        You can try to generate the final answer for the [Question] by referring to the [Query]-[Answer] pairs, starting with [Final
                        Content].
                        If you don't know the answer, generate a query to search engine based on what you already know and do not know, starting with
                        [Unsolved Query].
                        For example:
                        [Question]: "Where do greyhound buses that are in the birthplace of Spirit If...'s performer leave from? "
                        [Query 1]: Who is the performer of Spirit If... ?
                        If you don't know the answer:
                        [Unsolved Query]: Who is the performer of Spirit If... ?
                        If you know the answer:
                        [Answer 1]: The performer of Spirit If... is Kevin Drew.
                        [Query 2]: Where was Kevin Drew born?
                        If you don't know the answer:
                        [Unsolved Query]: Where was Kevin Drew born?
                        If you know the answer:
                        [Answer 2]: Toronto.
                        [Query 3]: Where do greyhound buses in Toronto leave from?
                        If you don't know the answer:
                        [Unsolved Query]: Where do greyhound buses in Toronto leave from?
                        If you know the answer:
                        [Answer 3]: Toronto Coach Terminal.
                        [Final Content]: The performer of Spirit If... is Kevin Drew [1]. Kevin Drew was born in Toronto [2]. Greyhound buses in
                        Toronto leave from Toronto
                        Coach Terminal [3]. So the final answer is Toronto Coach Terminal.
                        
                        [Question]:"Which magazine was started first Arthur’s Magazine or First for Women?"
                        [Query 1]: When was Arthur’s Magazine started?
                        [Answer 1]: 1844.
                        [Query 2]: When was First for Women started?
                        [Answer 2]: 1989
                        [Final Content]: Arthur’s Magazine started in 1844 [1]. First for Women started in 1989 [2]. So Arthur’s Magazine was started
                        first. So the answer is Arthur’s Magazi
                        [Question]: {}
                    """.format(q,q)}]
        feedback_answer = 'continue'
        predict_answer = ''
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((HOST, PORT))
        while round_count < 5 and not feedback_answer == 'end':
            print('round is {}'.format(round_count))
            try:
                time.sleep(0.5)
                rsp = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=message_keys_list
                )
                round_count += 1
                input_str = rsp.get("choices")[0]["message"]["content"]
                message_keys_list.append({"role": "assistant", "content": input_str})
                print('solving......')
                predict_answer += input_str
                sock.send(input_str.encode())
                print('send message {}'.format(input_str))
                feedback = sock.recv(10240).decode()
                print('feedback is '+feedback)
                if feedback == 'end':
                    break
                #[Query]:xxxx<SEP>[Answer]:xxxx<SEP>[Reference]:xxxx<SEP>
                feedback_list = feedback.split('<SEP>')
                if not 'Unsolved Query' in feedback:
                    new_prompt = """
                    According to this Reference, the answer for "{}" should be "{}",  
                    you can change your answer based on the Reference and continue constructing the reasoning chain to give the final answer for [Question]:{}
                    Reference: {}
                    """.format(feedback_list[0],feedback_list[1],q,feedback_list[2])
                else:
                    new_prompt = """
                    According to this Reference, the answer for "{}" should be "{}",  
                    you can give your answer based on the Reference and continue constructing the reasoning chain to give the final answer for [Question]：{}
                    Reference: {}
                    """.format(feedback_list[0],feedback_list[1],q,feedback_list[2])
                message_keys_list.append({"role": "user", "content":new_prompt})
            except:
                print('start_idx is {}'.format(k))
                sock.send('end'.encode())
                sock.close()
                return k
    
        if not feedback_answer == 'end':
            sock.send('end'.encode())
        sock.close()
        print(message_keys_list)
        break

    return -1

if __name__ == '__main__':
    start_idx = 0
    parser = argparse.ArgumentParser(description="Zero-shot-CoT")
    parser.add_argument("--dataset", type=str, default='bigbench_truth',
                        help="maximum number of workers for dataloader")
    parser.add_argument("--dataset_path", type=str, default='/root/datasets/truthful_qa/task_mc.json',
                        help="maximum number of workers for dataloader")
    parser.add_argument("--method", type=str, default='cos',
                        help="maximum number of workers for dataloader")
    args = parser.parse_args()
    while not start_idx == -1:
        start_idx = excute(args,
               start_idx=start_idx)