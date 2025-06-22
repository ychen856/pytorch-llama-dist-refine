import threading
import queue
import time
import random
import torch
from natsort import natsorted
from queue import Queue
import numpy as np

from pathlib import Path
import argparse
import yaml
from transformers import PreTrainedTokenizerFast, LlamaTokenizer, AutoModelForCausalLM, LlamaConfig, AutoConfig

from calculate_opt import *
from model_hf import LlamaForCausalLM_emb, LlamaForCausalLM_linear, LlamaForCausalLM_layer_0, LlamaForCausalLM_norm
from data import get_wikitext2_testloader, get_wikitext2_random_test_stream, get_wikitext2_testloader_full
from timestamp_manager import Timestamp_manager

parser = argparse.ArgumentParser(
    description='Pytorch Imagenet Training')
parser.add_argument('--config', default='config_server.yaml')
args = parser.parse_args()

data_collector = ServerClientDataCollector()
input_queue = Queue()
timestamp_manager = Timestamp_manager()

def load_model(checkpoints_dir, start_idx, end_idx, device):
    config, kwargs = AutoConfig.from_pretrained(
        args.ckpt_dir_hf_sep,
        return_unused_kwargs=True
    )
    #print('config: ', config)

    checkpoint_list = []
    checkpoints = sorted(Path(checkpoints_dir).glob("consolidated.*.pth"))
    assert len(checkpoints) > 0, f"no checkpoint files found in {checkpoints_dir}"

    checkpoint_idx = 0
    for checkpoint in checkpoints:
        ckpt_path = checkpoint
        #print(f'Loading checkpoint "{ckpt_path}"')

        checkpoint_list.append(torch.load(ckpt_path, map_location="cpu"))
        checkpoint_idx = checkpoint_idx + 1
        if checkpoint_idx > end_idx:
            break


    if device.type == 'cuda':
        torch.set_default_dtype(torch.float16)
        #torch.set_default_tensor_type(torch.cuda.HalfTensor)
    else:
        torch.set_default_tensor_type(torch.BFloat16Tensor)

    models = []
    for i in range(start_idx, end_idx + 1):
        #print('i: ', i)
        if i == 0:
            models.append(LlamaForCausalLM_emb(config))
            models[i].load_state_dict(checkpoint_list[i], strict=True)
            models[0].to(device)
        elif i == 33:
            models.append((LlamaForCausalLM_norm(config)))
            models[i].load_state_dict(checkpoint_list[i], strict=True)
            models[33].to(device)

        elif i == 34:
            models.append((LlamaForCausalLM_linear(config)))
            models[i].load_state_dict(checkpoint_list[i], strict=True)
            models[34].to(device)
        else:
            models.append(LlamaForCausalLM_layer_0(config))
            models[i].load_state_dict(checkpoint_list[i], strict=True)
            models[i].to(device)


    '''for i in range(0, len(models)):
        model = models[i]
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name, param.data)'''

    return models

def get_lm_head_idx(end_idx):

    lm_heads = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
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

def load_lm_head(checkpoints_dir, end_idx, device, cache_dir="llm_weights"):
    config, kwargs = AutoConfig.from_pretrained(
        args.ckpt_dir_hf,
        return_unused_kwargs=True
    )

    lm_head, lm_head_idx = get_lm_head_idx(end_idx)

    print('lm_head: ', lm_head)
    print('lm_head_idx: ', lm_head_idx)

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

def arrival_sampler(distribution="uniform", **kwargs):
    """Return a random delay based on given distribution"""
    if distribution == "uniform":
        return random.uniform(kwargs.get("low", 0.5), kwargs.get("high", 2.0))
    elif distribution == "exponential":
        return np.random.exponential(scale=kwargs.get("scale", 1.0))
    elif distribution == "poisson":
        # poisson returns integer, convert to float by adding jitter
        return np.random.poisson(lam=kwargs.get("lam", 1.0)) + random.uniform(0, 0.1)
    elif distribution == "normal":
        return max(0, random.gauss(kwargs.get("mu", 1.0), kwargs.get("sigma", 0.2)))
    else:
        raise ValueError("Unsupported distribution")

def data_producer(batch_size, seed, seqlen, bs, tokenizer, mode, distribution='uniform', dist_args={}):
    print('T1 start...')
    batch_count = 0
    is_first = True
    if mode == 1:   #batch arrival
        while True:
            if input_queue.qsize() == 0 and not is_first:
                while len(timestamp_manager.end_times) < batch_size:
                    time.sleep(0.0001)
                    break

            test_loader = get_wikitext2_testloader_full(tokenizer)
            testenc = test_loader.input_ids
            nsamples = testenc.numel() // seqlen

            for i in range(0, nsamples, bs):
                if i % 50 == 0:
                    print(f"sample {i}")

                # Calculate end index
                j = min(i + bs, nsamples)

                # Prepare inputs and move to device
                inputs = testenc[:, (i * seqlen):(j * seqlen)].to(device)
                inputs = inputs.reshape(j - i, seqlen)

                input_queue.put(inputs)

            is_first = False
            batch_count = batch_count + 1

            if batch_count == 10:
                return

    elif mode == 2: #stream arrival
        while True:
            if input_queue.qsize() == 0 and not is_first:
                while len(timestamp_manager.end_times) < batch_size:
                    time.sleep(0.0001)
                    break

            testenc = get_wikitext2_random_test_stream(batch_size, seed, seqlen, tokenizer)
            print('test loader: ', testenc)
            #testenc = test_loader.input_ids
            #nsamples = testenc.numel() // seqlen
            nsamples = len(testenc)

            for i in range(0, nsamples, bs):
                if i % 50 == 0:
                    print(f"sample {i}")

                # Calculate end index
                j = min(i + bs, nsamples)

                # Prepare inputs and move to device
                inputs = testenc[:, (i * seqlen):(j * seqlen)].to(device)
                inputs = inputs.reshape(j - i, seqlen)

                input_queue.put(inputs)

            is_first = False
            batch_count = batch_count + 1

            if batch_count == 10:
                return


    elif mode ==3:
        while True:
            if input_queue.qsize() == 0 and not is_first:
                while len(timestamp_manager.end_times) < batch_size:
                    time.sleep(0.0001)
                    break

            data = get_wikitext2_testloader(batch_size, seed, seqlen, tokenizer).input_ids
            index = 0
            while True:
                index = index + 1
                input_queue.put(data[index])

                delay = arrival_sampler(distribution, **dist_args)
                print(f"[Producer] New input arrived. Next in ~{delay:.2f}s.")
                time.sleep(delay)

                if index == n_sample:
                    break


            batch_count = batch_count + 1
            is_first = False

            if batch_count == 10:
                return



def task2_computation():
    print('T2 start...')
    for i in range(0, 10):
        while input_queue.empty():
            time.sleep(0.0001)

        while not input_queue.empty():
            print(input_queue.get())




if __name__ == '__main__':
    print('?????')
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    for key in config:
        for k, v in config[key].items():
            setattr(args, k, v)

    print('config type: ', args.config)

    device = torch.device("cuda")

    '''max_layers = args.max_layers
    start_idx = args.start_idx
    end_idx_buff = args.end_idx_buff
    head_idx = 2

    data_collector.statistic_period = 10

    models = load_model(args.ckpt_dir_hf_sep, start_idx, end_idx_buff, device)
    _, lm_models = load_lm_head(args.ckpt_dir_hf_sep, head_idx, device, cache_dir="llm_weights")'''
    print('hii')
    tokenizer = LlamaTokenizer.from_pretrained(args.ckpt_dir_hf, use_fast=False)

    n_sample = 10
    seed = random.seed(time.time())
    seqlen = 1024
    mode = 1
    bs = 1


    # Create and start threads
    thread1 = threading.Thread(target=data_producer, args=[n_sample, seed, seqlen, bs, tokenizer, mode], kwargs={
                                                                                            "distribution": "exponential",
                                                                                            "dist_args": {"scale": 0.8}
                                                                                            })
    thread2 = threading.Thread(target=task2_computation, args=[])
    #thread2 = threading.Thread(target=task2_computation,
    #                           args=[models, lm_models, start_idx, calculate_opt.end_idx, calculate_opt.end_idx_buff,
    #                                 head_idx, max_layers, device])
    #thread3 = threading.Thread(target=data_producer, args=[models, test_loader, bs, device])
    thread1.start()
    thread2.start()
    #thread3.start()

    # Wait for both threads to finish (optional)
    thread1.join()
    thread2.join()
    # thread3.join()