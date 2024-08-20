#!/bin/bash
export PYTHONPATH=$PYTHONPATH: 'pwd' ;

export GCR_GPT_URL=<your azure openai endpoint>
export GCR_GPT_KEY=<your azure openai key>

python_path=/bin/python3
exp_folder=experiments

$python_path src/main.py --experiments_config $exp_folder/summarization-0.json
$python_path src/main.py --experiments_config $exp_folder/summarization-1.json
$python_path src/main.py --experiments_config $exp_folder/summarization-2.json
