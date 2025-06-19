
from datasets import load_dataset
import random

from pathlib import Path
import argparse
import yaml

import torch
from transformers import PreTrainedTokenizerFast, LlamaTokenizer, AutoModelForCausalLM, LlamaConfig, AutoConfig

parser = argparse.ArgumentParser(
    description='Pytorch Imagenet Training')
parser.add_argument('--config', default='config_server.yaml')
args = parser.parse_args()

def get_wikitext2_testloader(nsamples, seed, seqlen, tokenizer):
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    input_ids = testenc['input_ids'][0]
    testloader = []

    for i in range(0, input_ids.shape[0] - seqlen, seqlen):
        inp = input_ids[i:i+seqlen].unsqueeze(0)  # Add batch dim
        tar = inp.clone()
        tar[:, :-1] = -100
        testloader.append((inp, tar))

    return testloader

def get_wikitext2_random_test_stream(nsamples, seed, seqlen, tokenizer):
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    random.seed(seed)

    data_lines = [line for line in testdata['text'] if line.strip() != '']
    n_lines = len(data_lines)

    testloader = []

    for _ in range(nsamples):
        cur_text = ''
        while len(tokenizer(cur_text)['input_ids']) < seqlen + 1:
            rand_idx = random.randint(0, n_lines - 1)
            cur_text += ' ' + data_lines[rand_idx]

        input_ids = tokenizer(cur_text, return_tensors='pt', truncation=True, max_length=seqlen + 1)['input_ids']
        input_ids = input_ids[:, :seqlen]

        inp = input_ids.clone()
        tar = input_ids.clone()
        tar[:, :-1] = -100

        testloader.append((inp, tar))

    return testloader


if __name__ == '__main__':
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    for key in config:
        for k, v in config[key].items():
            setattr(args, k, v)

    tokenizer = LlamaTokenizer.from_pretrained(args.ckpt_dir_hf_sep, use_fast=False)
    test_loader = get_wikitext2_testloader(1024, tokenizer)

    for data in test_loader:
        print('data: ', data)