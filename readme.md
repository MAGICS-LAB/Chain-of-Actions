# Chain-of-Actions (CoA)
This is the code of the paper Chain-of-Action. You can use this repo to reproduce the results in the paper.


## Chain of Action
### Environmental Setup
`pip install -r requirements.txt`

### Run Experiments
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


### Datasets
Download the datasets from the following:

https://github.com/kojima-takeshi188/zero_shot_cot/tree/main/dataset

https://github.com/kojima-takeshi188/zero_shot_cot/tree/main/log

### Instructions
You can run any baseline with our code provide, suchas auto_cot.py, few_shot.py, sc.py,....

#### You can use your own openai API and google seacrch API, which is required in our baseline code.


