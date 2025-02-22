# Chain-of-Action: Faithful and Multimodal Question Answering through Large Language Models
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/chain-of-action-faithful-and-multimodal/question-answering-on-fever)](https://paperswithcode.com/sota/question-answering-on-fever?p=chain-of-action-faithful-and-multimodal)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/chain-of-action-faithful-and-multimodal/question-answering-on-strategyqa)](https://paperswithcode.com/sota/question-answering-on-strategyqa?p=chain-of-action-faithful-and-multimodal)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/chain-of-action-faithful-and-multimodal/question-answering-on-truthfulqa)](https://paperswithcode.com/sota/question-answering-on-truthfulqa?p=chain-of-action-faithful-and-multimodal)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/chain-of-action-faithful-and-multimodal/question-answering-on-webquestions)](https://paperswithcode.com/sota/question-answering-on-webquestions?p=chain-of-action-faithful-and-multimodal)

This is the code of our ICLR 2025 paper [Chain-of-Action](https://openreview.net/forum?id=1BdPHbuimc). You can use this repo to reproduce the results in the paper.

* You can try to run our project by following the steps below, running in different environments may encounter various problems. We are still working hard to make it robust and bug-free.
* You should use your own **OpenAI API** and **Google search API**, which is required in our baseline and paper code.


## Datasets
Download the datasets from the following:

https://github.com/google/BIG-bench/tree/main/bigbench/benchmark_tasks

https://fever.ai/dataset/fever.html

https://huggingface.co/datasets/Stanford/web_questions


## Chain of Action
### Environmental Setup
`pip install -r requirements.txt`

### Run Experiments
An example on Dataset in the setting without IR:

`python chain-of-search-wo-ir.py`

An example on Dataset in the setting with IR:

`python chain-of-search.py`



## Baseline
### Environmental Setup
You can set up the experimental environment by running the following command line:

```shell
$ cd baselines/src
$ pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
$ pip3 install -r requirements.txt
$ export PYTHONPATH=$PYTHONPATH:$PWD
```


### Instructions
You can run any baseline with the code we provide.

An example on Dataset in the setting with **React**:

`python react.py`



## Acknowledgment
The experiments in this work benefit from the following open-source codes:

https://github.com/xsc1234/Search-in-the-Chain

https://github.com/amazon-science/auto-cot

https://python.langchain.com/v0.1/docs/modules/agents/

https://github.com/stanfordnlp/dspy

https://github.com/lucidrains/toolformer-pytorch

https://github.com/princeton-nlp/tree-of-thought-llm

https://www.promptingguide.ai/

## Citation
If you find our work useful, please consider citing our paper:
```
@inproceedings{
pan2025chainofaction,
title={Chain-of-Action: Faithful and Multimodal Question Answering through Large Language Models},
author={Zhenyu Pan and Haozheng Luo and Manling Li and Han Liu},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=1BdPHbuimc}
}
```
