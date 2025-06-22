
from datasets import load_dataset
import random
from datetime import datetime

from pathlib import Path
import argparse
import yaml

import torch
from transformers import PreTrainedTokenizerFast, LlamaTokenizer, AutoModelForCausalLM, LlamaConfig, AutoConfig

'''parser = argparse.ArgumentParser(
    description='Pytorch Imagenet Training')
parser.add_argument('--config', default='config_server.yaml')
args = parser.parse_args()'''

def get_wikitext2_testloader_full(tokenizer):
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    return testenc

def get_wikitext2_testloader(nsamples, seed, seqlen, tokenizer, device):
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    #input_ids = testenc['input_ids'][0]  # assume batch dim = 1
    random.seed(seed)
    testloader = []

    for _ in range(nsamples):
        i = random.randint(0, testenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = testenc.input_ids[:, i:j].to(device)
        testloader.append(inp)

    return testloader
def get_wikitext2_random_test_stream(nsamples, seed, seqlen, tokenizer, device):
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    lines = [line for line in testdata['text'] if line.strip() != '']
    random.seed(seed)

    testloader = []

    for _ in range(nsamples):
        cur_text = ''
        while len(tokenizer(cur_text)['input_ids']) < seqlen + 1:
            cur_text += ' ' + random.choice(lines)

        tokens = tokenizer(cur_text, return_tensors='pt', truncation=True, max_length=seqlen + 1)['input_ids']
        input_ids = tokens[:, :seqlen].to(device)

        inp = input_ids.clone()
        tar = input_ids.clone()
        tar[:, :-1] = -100  # mask non-target tokens

        #testloader.append((inp, tar))
        testloader.append(inp)
    return testloader


'''if __name__ == '__main__':
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    for key in config:
        for k, v in config[key].items():
            setattr(args, k, v)


    tokenizer = LlamaTokenizer.from_pretrained(args.ckpt_dir_hf_sep, use_fast=False)

    seed = random.seed(datetime.now().timestamp())
    #test_loader = get_wikitext2_testloader(3, seed, 1024, tokenizer)
    test_loader = get_wikitext2_random_test_stream(3, seed, 1024, tokenizer)

    for data in test_loader:
        print('data: ', data)

    print('================================================')
    seed = random.seed(datetime.now().timestamp())
    #test_loader = get_wikitext2_testloader(3, seed, 1024, tokenizer)
    test_loader = get_wikitext2_random_test_stream(3, seed, 1024, tokenizer)

    for data in test_loader:
        print('data: ', data)

    print('================================================')
    seed = random.seed(datetime.now().timestamp())
    #test_loader = get_wikitext2_testloader(3, seed, 1024, tokenizer)
    test_loader = get_wikitext2_random_test_stream(3, seed, 1024, tokenizer)

    for data in test_loader:
        print('data: ', data)'''


