# this code is used for seperating the weights into small pieces and store them into seperated .pt files. One time usage.
import random
from typing import Optional

import numpy as np
import torch
import time
from pathlib import Path
import json
from sentencepiece import SentencePieceProcessor
from tqdm import tqdm
import argparse
from data import get_wikitext2_testloader_full, get_wikitext2_trainloader
import torch.nn as nn
import safetensors
from natsort import natsorted

from safetensors.torch import save_file
from transformers import PreTrainedTokenizerFast, LlamaTokenizer, AutoModelForCausalLM, LlamaConfig, AutoConfig

import sys

from early_exit import early_exit_lm_head, early_exit_regression, early_exit_lm_head_shift
from model_hf import LlamaForCausalLM, LlamaForCausalLM_emb, LlamaForCausalLM_layer_0, LlamaForCausalLM_norm, \
    LlamaForCausalLM_linear
import yaml


parser = argparse.ArgumentParser(
    description='Pytorch Imagenet Training')
parser.add_argument('--config', default='config_server.yaml')
parser.add_argument('--head', type=int)
parser.add_argument('--ppl', type=int)
args = parser.parse_args()


def get_llm(model, cache_dir="llm_weights"):
    model = AutoModelForCausalLM.from_pretrained(
        model,
        torch_dtype=torch.float16,
        cache_dir=cache_dir,
        low_cpu_mem_usage=True,
        device_map="auto"
    )

    model.seqlen = 1024
    return model


def load_model(checkpoints_dir, start_idx, end_idx, device):
    config, kwargs = AutoConfig.from_pretrained(
        args.ckpt_dir_hf,
        return_unused_kwargs=True
    )
    print('config: ', config)

    checkpoint_list = []
    checkpoints = sorted(Path(checkpoints_dir).glob("consolidated.*.pth"))
    assert len(checkpoints) > 0, f"no checkpoint files found in {checkpoints_dir}"

    checkpoint_idx = 0
    for checkpoint in checkpoints:
        ckpt_path = checkpoint
        print(f'Loading checkpoint "{ckpt_path}"')

        checkpoint_list.append(torch.load(ckpt_path, map_location="cpu"))
        checkpoint_idx = checkpoint_idx + 1
        if checkpoint_idx > end_idx:
            break

    if device.type == 'cuda':
        torch.set_default_tensor_type(torch.cuda.HalfTensor)
    else:
        torch.set_default_tensor_type(torch.BFloat16Tensor)

    models = []
    for i in range(start_idx, end_idx + 1):
        print('i: ', i)
        if i == 0:
            models.append(LlamaForCausalLM_emb(config))
            models[i].load_state_dict(checkpoint_list[i], strict=True)
            #models[0].model.embed_tokens.weight = nn.Parameter(checkpoint_list[0]['model.embed_tokens.weight'])
            models[0].to(device)
        elif i == 33:
            models.append((LlamaForCausalLM_norm(config)))
            models[i].load_state_dict(checkpoint_list[i], strict=True)
            #models[33].model.norm.weight = nn.Parameter(checkpoint_list[33]['model.norm.weight'])
            models[33].to(device)

        elif i == 34:
            models.append((LlamaForCausalLM_linear(config)))
            models[i].load_state_dict(checkpoint_list[i], strict=True)
            #models[34].lm_head.weight = nn.Parameter(checkpoint_list[34]['lm_head.weight'])
            models[34].to(device)
        else:
            models.append(LlamaForCausalLM_layer_0(config))
            models[i].load_state_dict(checkpoint_list[i], strict=True)

            models[i].to(device)


    return models



def load_lm_head(checkpoints_dir, end_idx, device, cache_dir="llm_weights"):
    config, kwargs = AutoConfig.from_pretrained(
        args.ckpt_dir_hf,
        return_unused_kwargs=True
    )

    lm_head, lm_head_idx = get_lm_head_idx(end_idx)
    print()


    checkpoint_list = []
    checkpoints = sorted(Path(checkpoints_dir).glob("lm_head.*.pth"))
    checkpoints = natsorted(checkpoints)
    assert len(checkpoints) > 0, f"no checkpoint files found in {checkpoints_dir}"


    for i in range(0, len(checkpoints)):
        if i == 0 or i == lm_head_idx:
            ckpt_path = checkpoints[i]
            print(f'Loading checkpoint "{ckpt_path}"')

            checkpoint_list.append(torch.load(ckpt_path, map_location="cpu"))



    if device.type == 'cuda':
        torch.set_default_tensor_type(torch.cuda.HalfTensor)
    else:
        torch.set_default_tensor_type(torch.BFloat16Tensor)

    lm_models = []

    for i in range(0, len(checkpoint_list)):
        if i == 0:
            lm_models.append((LlamaForCausalLM_norm(config)))
            lm_models[i].load_state_dict(checkpoint_list[i], strict=True)
            lm_models[i].to(device)

        else:
            lm_models.append((LlamaForCausalLM_linear(config)))
            lm_models[i].load_state_dict(checkpoint_list[i], strict=True)
            lm_models[i].to(device)

    return lm_head, lm_models


def get_lm_head_idx(end_idx):

    lm_heads = [1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    lm_head = 1
    lm_head_idx = 0

    for i in range(0, len(lm_heads)):
        if lm_heads[i] > end_idx:
            #lm_head = lm_heads[i - 1]
            #lm_head_idx = lm_head_idx - 1
            break
        elif lm_heads[i] == end_idx:
            lm_head = lm_heads[i]
            lm_head_idx = i
            break

        lm_head = lm_heads[i]
        lm_head_idx = i

    lm_head_idx = lm_head_idx + 1


    return lm_head, lm_head_idx

if __name__ == '__main__':
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    for key in config:
        for k, v in config[key].items():
            setattr(args, k, v)


    #print('config type: ', args.config)
    torch.manual_seed(0)


    #allow_cuda = False
    device = torch.device("cuda")

    head_idx = args.head
    #head_idx = 10


    models = load_model(args.ckpt_dir_hf_sep, 0, 34, device)
    tokenizer = LlamaTokenizer.from_pretrained(args.ckpt_dir_hf, use_fast=False)
    _, lm_models = load_lm_head(args.ckpt_dir_hf_sep, head_idx, device, cache_dir="llm_weights")

    print("loading success")


    bs = 1


    '''# test loader
    # loading inputs data
    seqlen = 1024
    # Get input IDs
    test_loader = get_wikitext2_testloader_full(tokenizer)
    #testenc = test_loader.input_ids
    testenc = test_loader.input_ids

    # Calculate number of samples
    nsamples = testenc.numel() // seqlen'''

    nsamples = 500
    #seed = random.seed(time.time())
    seed = 1
    seqlen = 1024

    testenc = get_wikitext2_trainloader(nsamples, seed, seqlen, tokenizer, device)

    # List to store negative log likelihoods
    nlls = []
    early_count = 0
    print(f"nsamples {nsamples}")
    # Loop through each batch
    for i in range(0, nsamples, bs):
        if i % 50 == 0:
            print(f"sample {i}")

        # Calculate end index
        j = min(i + bs, nsamples)

        # Prepare inputs and move to device
        #inputs = testenc[:, (i * seqlen):(j * seqlen)].to(device)
        #inputs = inputs.reshape(j - i, seqlen)
        inputs = testenc[i]

        lm_logits = None
        is_early_exit = False
        with torch.no_grad():
            out, ids, mask = models[0](inputs)
            for k in range(1, len(models) - 2):
                out, ids, mask = models[k](out.last_hidden_state, position_ids=ids, attention_mask=mask)
                if k == head_idx:
                    try:
                        is_early_exit, lm_logits = early_exit_lm_head_shift(lm_models, out, head_idx, args.ppl)
                        break
                        #is_early_exit, lm_logits = early_exit_lm_head(lm_models, out, head_idx, args.ppl)
                        # print('is early: ', is_early_exit)
                    except Exception as e:
                        print('early oom!')
                        is_oom = True
                        is_early_exit = False

                        end_idx = k

                    if is_early_exit:
                        early_count = early_count + 1
                        is_early_exit = True
                        break

            if not is_early_exit:
                lm_logits = models[33](out.last_hidden_state)
                lm_logits = models[34](lm_logits)


            # Shift logits and labels for next token prediction
            shift_logits = lm_logits[:, :-1, :].contiguous()
            shift_labels = inputs[:, 1:]

            # Compute loss
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

            # Calculate negative log likelihood
            neg_log_likelihood = loss.detach().float() * seqlen * (j - i)

            # Append to list of negative log likelihoods
            nlls.append(neg_log_likelihood)

            sys.stdout.flush()

    print('begin calcualte ppl')
    # Compute perplexity
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))
    # Empty CUDA cache to save memory
    torch.cuda.empty_cache()

    print('ppl: ', ppl.item())
    print('early count: ', early_count)