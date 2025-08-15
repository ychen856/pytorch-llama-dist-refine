import gc
import threading
import queue
import time
import random
from random import randrange
import torch
from natsort import natsorted
from queue import Queue
import numpy as np
import math
import os

import concurrent.futures
from pathlib import Path
import argparse
import yaml
import sys
from transformers import PreTrainedTokenizerFast, LlamaTokenizer, AutoModelForCausalLM, LlamaConfig, AutoConfig

from calculate_opt_adaptive import *
from lm_head_manager import LMHeadManager
from predictive_splitting_manager import PredictiveSplittingManager
from model_hf import LlamaForCausalLM_emb, LlamaForCausalLM_linear, LlamaForCausalLM_layer_0, LlamaForCausalLM_norm
from data import get_wikitext2_testloader, get_wikitext2_random_test_stream, get_wikitext2_testloader_full
from timestamp_manager import Timestamp_manager
from early_exit import early_exit_lm_head, early_exit_regression
#import http_sender
import http_sender_2 as http_sender
from logger import Logger

parser = argparse.ArgumentParser(
    description='Pytorch Imagenet Training')
parser.add_argument('--config', default='config_server.yaml')
parser.add_argument('--log', default='test.log')
parser.add_argument('--ppl', type=int, default=10)
parser.add_argument('--weight', type=float, default=0.0)
parser.add_argument('--mode', default='fixed')
args = parser.parse_args()
logger = Logger(filepath=args.log)
head_names = [1, 2, 3, 4, 6]
ppl_list = [10, 20, 30]
'''init_params = {
    (1, 10): {'a': 407,  'b': 2093, 'gamma': 1.0},
    (1, 20): {'a': 926,  'b': 1574, 'gamma': 1.0},
    (1, 30): {'a': 231,  'b': 2269, 'gamma': 1.0},
    (2, 10): {'a': 572,  'b': 1928, 'gamma': 1.0},
    (2, 20): {'a': 1075, 'b': 1425, 'gamma': 1.0},
    (2, 30): {'a': 1415, 'b': 1085, 'gamma': 1.0},
    (3, 10): {'a': 769,  'b': 1731, 'gamma': 1.0},
    (3, 20): {'a': 1422, 'b': 1078, 'gamma': 1.0},
    (3, 30): {'a': 1850, 'b': 650,  'gamma': 1.0},
    (4, 10): {'a': 788,  'b': 1712, 'gamma': 1.0},
    (4, 20): {'a': 1499, 'b': 1001, 'gamma': 1.0},
    (4, 30): {'a': 1973, 'b': 527,  'gamma': 1.0},
    (6, 10): {'a': 763,  'b': 1737, 'gamma': 1.0},
    (6, 20): {'a': 1659, 'b': 841,  'gamma': 1.0},
    (6, 30): {'a': 2266, 'b': 243,  'gamma': 1.0}
}'''
init_params = {
    (1, 10): {'a': 56,  'b': 277, 'gamma': 1.0},
    (1, 20): {'a': 126, 'b': 207, 'gamma': 1.0},
    (1, 30): {'a': 167, 'b': 166, 'gamma': 1.0},
    (2, 10): {'a': 69,  'b': 264, 'gamma': 1.0},
    (2, 20): {'a': 140, 'b': 193, 'gamma': 1.0},
    (2, 30): {'a': 188, 'b': 145, 'gamma': 1.0},
    (3, 10): {'a': 89,  'b': 244, 'gamma': 1.0},
    (3, 20): {'a': 194, 'b': 139, 'gamma': 1.0},
    (3, 30): {'a': 252, 'b': 81,  'gamma': 1.0},
    (4, 10): {'a': 99,  'b': 234, 'gamma': 1.0},
    (4, 20): {'a': 202, 'b': 131, 'gamma': 1.0},
    (4, 30): {'a': 268, 'b': 65,  'gamma': 1.0},
    (6, 10): {'a': 103, 'b': 230, 'gamma': 1.0},
    (6, 20): {'a': 227, 'b': 106, 'gamma': 1.0},
    (6, 30): {'a': 302, 'b': 31,  'gamma': 1.0}
}
lm_manager = LMHeadManager(head_names, ppl_list, init_params, logger)
shock_manager = PredictiveSplittingManager(lm_manager, logger, shock_alpha=1.5, window_size=5, shock_threshold=3)

sleep_time_per_layer = 0
performance_data_store = PerformanceDataStore(shock_manager, logger)
input_queue = Queue()
outgoing_queue = Queue()
timestamp_manager = Timestamp_manager(logger)
stop_event = threading.Event()

def layer_reallocation(type, start_idx, end_idx_buff, max_layers, models):
    if type == 1: #add buffer layers
        #print('increase buffer')
        config, kwargs = AutoConfig.from_pretrained(
            args.ckpt_dir_hf_sep,
            return_unused_kwargs=True
        )
        #print('config: ', config)

        checkpoint_list = []
        checkpoints = sorted(Path(args.ckpt_dir_hf_sep).glob("consolidated.*.pth"))
        assert len(checkpoints) > 0, f"no checkpoint files found in {args.ckpt_dir_hf_sep}"

        checkpoints = checkpoints[end_idx_buff + 1:]
        checkpoint_idx = end_idx_buff
        for checkpoint in checkpoints:
            ckpt_path = checkpoint

            checkpoint_list.append(torch.load(ckpt_path, map_location="cpu"))
            checkpoint_idx = checkpoint_idx + 1
            if checkpoint_idx >= max_layers:
                break
            if checkpoint_idx > end_idx_buff + 2:
                break

        start_idx = end_idx_buff + 1
        if end_idx_buff + 3 <= max_layers:
            end_idx_buff = end_idx_buff + 3
        else:
            end_idx_buff = max_layers


        if device.type == 'cuda':
            torch.set_default_tensor_type(torch.cuda.HalfTensor)
        else:
            torch.set_default_tensor_type(torch.BFloat16Tensor)


        for i in range(start_idx, end_idx_buff + 1):
            #print('i: ', i)
            try:
                if i == 0:
                    models.append(LlamaForCausalLM_emb(config))
                    models[i].load_state_dict(checkpoint_list[i - start_idx], strict=True)
                    models[0].to(device)
                elif i == 33:
                    models.append((LlamaForCausalLM_norm(config)))
                    models[i].load_state_dict(checkpoint_list[i - start_idx], strict=True)
                    models[33].to(device)

                elif i == 34:
                    models.append((LlamaForCausalLM_linear(config)))
                    models[i].load_state_dict(checkpoint_list[i - start_idx], strict=True)
                    models[34].to(device)
                else:
                    models.append(LlamaForCausalLM_layer_0(config))
                    models[i].load_state_dict(checkpoint_list[i - start_idx], strict=True)
                    models[i].to(device)
            except:
                end_idx_buff = i - 1
                break

    if type == 2: # drop layers
        #print('decrease buffer')
        models = models[:-1]
        end_idx_buff = end_idx_buff - 1
    if type == 4:   #reload the whole model
        load_model(args.ckpt_dir_hf_sep, 0, end_idx_buff, torch.device("cuda:0"))

    '''for i in range(0, len(models)):
                model = models[i]
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        print(name, param.data)'''

    return models, end_idx_buff
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
def load_lm_head(checkpoints_dir, end_idx, device, cache_dir="llm_weights"):
    config, kwargs = AutoConfig.from_pretrained(
        args.ckpt_dir_hf,
        return_unused_kwargs=True
    )

    lm_head, lm_head_idx = get_lm_head_idx(end_idx)

    print('lm_head: ', lm_head)
    print('lm_head_idx: ', lm_head_idx)
    logger.log(f"lm_head: {lm_head}")
    logger.log(f"lm_head_idx: {lm_head_idx}")

    checkpoint_list = []
    checkpoints = sorted(Path(checkpoints_dir).glob("lm_head.*.pth"))
    checkpoints = natsorted(checkpoints)
    logger.log(f'zzzzzz: {checkpoints}')

    assert len(checkpoints) > 0, f"no checkpoint files found in {checkpoints_dir}"


    for i in range(0, len(checkpoints)):
        if i == 0 or i == lm_head_idx:
            ckpt_path = checkpoints[i]
            print(f'Loading checkpoint "{ckpt_path}"')
            logger.log(f'Loading checkpoint "{ckpt_path}"')

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
def data_producer(total_batch_num, batch_size, seed, seqlen, bs, tokenizer, mode, device, distribution='uniform', dist_args={}):
    print('T1 start...')
    batch_count = 0
    is_first = True
    if mode == 1:   #batch arrival
        while True:
            if input_queue.qsize() == 0 and not is_first:
                while len(timestamp_manager.end_times) < batch_size:
                    time.sleep(0.0001)
                    break

                print('time: ', timestamp_manager)
                logger.log(f'time: {timestamp_manager}')
                timestamp_manager.get_time_diff_every_n_inputs(10)
                timestamp_manager.clearAll()

                print('batch count: ', batch_count)
                logger.log(f'batch count: {batch_count}')
                if batch_count > total_batch_num:
                    print('end...')
                    logger.log(f'end...')
                    return

            test_loader = get_wikitext2_testloader_full(tokenizer)
            testenc = test_loader.input_ids
            nsamples = testenc.numel() // seqlen

            for i in range(0, nsamples, bs):
                if i % 50 == 0:
                    print(f"sample {i}")
                    logger.log(f'sample {i}')

                # Calculate end index
                j = min(i + bs, nsamples)

                # Prepare inputs and move to device
                inputs = testenc[:, (i * seqlen):(j * seqlen)].to(device)
                inputs = inputs.reshape(j - i, seqlen)

                input_queue.put(inputs)

            is_first = False
            batch_count = batch_count + 1



    elif mode == 2:  # stream arrival
        testenc = None
        global sleep_time_per_layer
        #outgoing_queue.put(['server', 0])
        #outgoing_queue.put(['communication', 0])

        while True and not stop_event.is_set():
            if not is_first:

                while len(timestamp_manager.end_times) < batch_size:
                    time.sleep(0.0001)

                print('time: ', timestamp_manager)

                logger.log(f'time: {timestamp_manager}')

                timestamp_manager.get_time_diff_every_n_inputs(20)

                timestamp_manager.clearAll()

                time.sleep(10)
                gc.collect()

                print('batch count: ', batch_count)

                logger.log(f'batch count: {batch_count}')

                if batch_count > total_batch_num:
                    print('end...')
                    logger.log(f'end...')
                    time.sleep(10)
                    stop_event.set()

                    os._exit(0)

                    return

            '''if batch_count == 0:
                #sleep_time_per_layer = 0.3
                logger.log(f'??????????????????????')
                logger.log(f'??????????????????????')
                logger.log(f'??????????????????????')
                logger.log(f'??????????????????????')
                logger.log(f'??????????????????????')
                logger.log(f'??????????????????????')
                logger.log(f'??????????????????????')
                logger.log(f'??????????????????????')
                logger.log(f'??????????????????????')
                logger.log(f'??????????????????????')
                logger.log(f'??????????????????????')
                logger.log(f'??????????????????????')
                logger.log(f'??????????????????????')
                outgoing_queue.put(['communication', 1])'''

            if is_first:
                testenc = get_wikitext2_testloader(batch_size, seed, seqlen, tokenizer, device)
                logger.log(f'testenc: {testenc}')

            nsamples = len(testenc)

            print('test loader len: ', nsamples)

            for i in range(0, nsamples, bs):

                if i % 50 == 0:
                    print(f"sample {i}")

                input_queue.put(testenc[i].to(device))

            is_first = False

            batch_count = batch_count + 1



    elif mode == 3:
        print('mode: ', 3)
        while True:
            if input_queue.qsize() == 0 and not is_first:
                while len(timestamp_manager.end_times) < batch_size:
                    time.sleep(0.0001)
                    break

                print('time: ', timestamp_manager)
                timestamp_manager.get_time_diff_every_n_inputs(10)
                timestamp_manager.clearAll()

                print('batch count: ', batch_count)
                if batch_count > total_batch_num:
                    print('end...')
                    return

            testenc = get_wikitext2_testloader(batch_size, seed, seqlen, tokenizer, device)
            nsamples = len(testenc)

            index = 0
            while True:
                input_queue.put(testenc[index])

                delay = arrival_sampler(distribution, **dist_args)
                print(f"[Producer] New input arrived. Next in ~{delay:.2f}s.")
                time.sleep(delay)

                index = index + 1
                if index == nsamples:
                    break

            batch_count = batch_count + 1
            is_first = False

            '''print('batch count: ', batch_count)
            if batch_count == total_batch_num:
                return'''

def task1_data_sending(args):
    while 1 and not stop_event.is_set():
        timeout_count = 0

        #while outgoing_queue.qsize() < 3 and input_queue.qsize() > 0 and performance_data_store.steady_state:
        while outgoing_queue.qsize() < 3 and input_queue.qsize() > 0 and performance_data_store.steady_state:
        #while outgoing_queue.qsize() < 3 and input_queue.qsize() > 0:
            timeout_count = timeout_count + 1

            start_time = time.time()
            #print('outgoing queue size: ', outgoing_queue.qsize())

            if input_queue.qsize() > 0: #and calculate_opt.incoming_count + 2 >= calculate_opt.outgoint_count:
                idx = input_queue.qsize()
                timestamp_manager.start_times = (idx, start_time)

                output = input_queue.get()
                outgoing_queue.put([0, output, None, None, idx, 0])

                #packed_data = serialize_and_compress(0, [None, None, output], None, None, idx, 0)
                #outgoing_queue.put(packed_data)

                end_time = time.time()


                print('server idle!')
                logger.log(f'server idle!')
                logger.log(f'start idx: 0')
                logger.log(f'end idx: 0')
            else:
                logger.log(f'ELSE!')
                break


        data = outgoing_queue.get()
        performance_data_store.outgoing_count = performance_data_store.outgoing_count + 1
        #http_sender.send_data(args.server_ip, args.server_port, data, performance_data_store, timestamp_manager, logger)
        http_sender.send_data(args.gateway_ip, args.gateway_port, data, performance_data_store, timestamp_manager, logger)


def task1_data_sending_direct(args):
    while 1 and not stop_event.is_set():
        start_time = time.time()
        if input_queue.qsize() > 0:  # and calculate_opt.incoming_count + 2 >= calculate_opt.outgoint_count:
            logger.log(f'queue size: {input_queue.qsize()}')
            idx = input_queue.qsize()
            timestamp_manager.start_times = (idx, start_time)

            output = input_queue.get()
            outgoing_queue.put([0, output, None, None, idx, 0])

            # packed_data = serialize_and_compress(0, [None, None, output], None, None, idx, 0)
            # outgoing_queue.put(packed_data)

            end_time = time.time()

            print('server idle!')
            logger.log(f'server idle!')
            logger.log(f'start idx: 0')
            logger.log(f'end idx: 0')

            data = outgoing_queue.get()
            performance_data_store.outgoing_count = performance_data_store.outgoing_count + 1
            #http_sender.send_data(args.server_ip, args.server_port, data, performance_data_store, timestamp_manager, logger)
            http_sender.send_data(args.gateway_ip, args.gateway_port, data, performance_data_store, timestamp_manager,
                                  logger)




def task1_data_sending_multi(args):
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = []
        while 1 and not stop_event.is_set():
            timeout_count = 0

            logger.log(f'queue size t1: {outgoing_queue.qsize()}')
            prob = random.randint(1, 3)
            logger.log(f'probbb: {prob}')
            #while outgoing_queue.qsize() < 3 and input_queue.qsize() > 0 and performance_data_store.steady_state:
            while outgoing_queue.qsize() < 1 and input_queue.qsize() > 0 and performance_data_store.steady_state and prob % 3 >= 0:
            #while outgoing_queue.qsize() < 3 and input_queue.qsize() > 0:
                timeout_count = timeout_count + 1

                start_time = time.time()
                #print('outgoing queue size: ', outgoing_queue.qsize())

                if input_queue.qsize() > 0: #and calculate_opt.incoming_count + 2 >= calculate_opt.outgoint_count:
                    idx = input_queue.qsize()
                    timestamp_manager.start_times = (idx, start_time)

                    output = input_queue.get()
                    outgoing_queue.put([0, output, None, None, idx, 0])

                    #packed_data = serialize_and_compress(0, [None, None, output], None, None, idx, 0)
                    #outgoing_queue.put(packed_data)

                    end_time = time.time()


                    print('server idle!')
                    logger.log(f'server idle!')
                    logger.log(f'start idx: 0')
                    logger.log(f'end idx: 0')
                else:
                    logger.log(f'ELSE!')
                    break


            data = outgoing_queue.get()
            #performance_data_store.outgoing_count = performance_data_store.outgoing_count + 1
            #http_sender.send_data(args.gateway_ip, args.gateway_port, data, performance_data_store, timestamp_manager, logger)


            futures.append(
                executor.submit(http_sender.send_request, args.gateway_ip, args.gateway_port, data, performance_data_store, timestamp_manager, logger))
                #executor.submit(http_sender.send_request, args.server_ip, args.server_port, data,
                #                performance_data_store, timestamp_manager, logger))


        # 等所有任務完成
        concurrent.futures.wait(futures)


'''def task2_computation():
    print('T2 start...')
    for i in range(0, 10):
        while input_queue.empty():
            time.sleep(0.0001)

        while not input_queue.empty():
            print(input_queue.get())'''



def task2_computation(models, lm_models, start_idx, end_idx, end_idx_buff, head_idx, max_layers, batch_num, device):

    is_oom = False
    #prune_wanda_allocation(args, models, tokenizer, testenc[0], device=torch.device("cuda:0"))
    # Loop through each batch
    max_batch_num = batch_num
    batch_count = 0
    cycle_count = 0
    input_count = 0
    count = 0
    early_count = 0
    statistics_period = performance_data_store.statistic_period
    batch_size = 20
    # repeated 5->0, 10->1, 20->3
    global repeated
    #while not input_queue.empty():
    while(1 and not stop_event.is_set()):
        if count > max_batch_num * batch_size:
            break

        while input_queue.empty():
            time.sleep(0.0001)


        if input_count % batch_size == 0:
            early_count = 0


        is_early_exit = False
        count = count + 1
        #print('========================================')

        logger.log(f'queue size t2: {outgoing_queue.qsize()}')
        idx = input_queue.qsize()
        input = input_queue.get()

        if input_count % 50 == 0:
            print(f"sample {input_count}")


        start_time = time.time()
        timestamp_manager.start_times = (idx, start_time)


        print('start idx: ', 0)
        logger.log(f'start idx: 0')

        with torch.no_grad():
            # Forward pass through the model
            try:
                time.sleep(sleep_time_per_layer)
                out, ids, mask = models[0](input)
            except Exception as e:
                print(e)

            for k in range(1, end_idx + 1):
                try:
                    time.sleep(sleep_time_per_layer)
                    out, ids, mask = models[k](out.last_hidden_state, position_ids=ids, attention_mask=mask)
                    if k == head_idx:
                        try:
                            time.sleep(sleep_time_per_layer)
                            is_early_exit, lm_logits = early_exit_lm_head(lm_models, out, head_idx, args.ppl)
                        except Exception as e:
                            print('early oom!')
                            logger.log(f'early oom')
                            is_oom = True
                            is_early_exit = False

                            end_idx = k - 1

                        if is_early_exit:
                            print('end idx: ', k)
                            logger.log(f'end idx: {k}')
                            timestamp_manager.end_times = (idx, time.time())
                            lm_manager.update(k, args.ppl, True)
                            break

                        lm_manager.update(k, args.ppl, False)

                except Exception as e:
                    print('oom!!!')
                    logger.log(f'oom')
                    is_oom = True

                    end_idx = k - 1

                    #print('updated end idx: ', end_idx)
                    break


        end_time = time.time()

        if not is_early_exit:
            print('end idx: ', end_idx)
            logger.log(f'end idx: {end_idx}')

        print('is early: ', is_early_exit)
        logger.log(f'is early: {is_early_exit}')



        if is_early_exit:
            early_count = early_count + 1
            performance_data_store.add_client_info(datetime.now() + timedelta(milliseconds=50), end_idx, end_idx_buff, end_time - start_time, head_idx, True)

        if not is_early_exit:
            cycle_count = cycle_count + 1
            input_count = input_count + 1

            print('cycle count: ', cycle_count)
            print('input count: ', input_count)


            #outgoing_queue.put([end_idx + 1, pruned_feature_vector, ids, mask, idx, end_time - start_time])
            outgoing_queue.put([end_idx + 1, out, ids, mask, idx, end_time - start_time])

            print('outgoing queue PUT!')
            performance_data_store.add_client_info(datetime.now() + timedelta(milliseconds=50), end_idx, end_idx_buff, end_time - start_time, head_idx, False)

            if is_oom:
                end_idx = max(1, math.ceil(end_idx / 2))
                is_oom = False
            logger.log(f'cycle count: {cycle_count}')
            #print('statistic: ', statistics_period)
            if (input_count) % 4 == 0 and input_count < 24 and end_idx < max_layers and statistics_period <= 20:
                print('testing higher value(i<12)')
                logger.log(f'testing higher value(i<30)')
                performance_data_store.max_end_idx = end_idx
                end_idx = end_idx + 1
                logger.log(f'new end: {end_idx}')

            # if cycle_count > (statistics_period - 6) and input_count >= 12 and cycle_count % 2 == 0:
            # if cycle_count > (statistics_period - 12) and input_count >= 24 and cycle_count % 4 == 0:
            logger.log(f'testing statistic period: {statistics_period}')
            logger.log(f'testing cycle count: {cycle_count}')
            logger.log(f'testing s-c: {statistics_period - cycle_count}')
            if cycle_count > (statistics_period - 12) and input_count >= 24 and (
                    statistics_period - cycle_count) % 4 == 0:
                print('testing lower value (i>12)')
                logger.log(f'testing lower value (i>30)')
                end_idx = max(1, end_idx - 1)
                logger.log(f'new end: {end_idx}')

            # if cycle_count == (statistics_period - 6) and input_count >= 12 and end_idx < max_layers and cycle_count % 2 == 0:
            # if cycle_count == (statistics_period - 12) and input_count >= 24 and end_idx < max_layers and cycle_count % 4 == 0:
            if cycle_count == (statistics_period - 12) and input_count >= 24 and end_idx < max_layers and (
                    statistics_period - cycle_count) % 4 == 0:
                print('testing higher value (i>30)')
                logger.log(f'testing higher value (i>30)')
                performance_data_store.max_end_idx = end_idx
                end_idx = end_idx + 1
                logger.log(f'new end: {end_idx}')


        #if (input_count) % 10 == 0:
        print('record count: ', performance_data_store.new_record_count)
        if performance_data_store.new_record_count >= statistics_period:
            #print('statistic')
            #statistics_period = statistics_period + 5
            end_idx, end_idx_buff, statistics_period = calculate_opt(performance_data_store, args.ppl, lm_manager, args.mode, shock_manager, logger)
            outgoing_queue.put(['opt', end_idx + 1])

            print('opt end idx: ', end_idx)
            print('opt buff idx: ', end_idx_buff)
            print('opt statistics period: ', statistics_period)
            logger.log(f'opt end idx: {end_idx}')
            logger.log(f'opt buff idx: {end_idx_buff}')
            logger.log(f'opt statistics period: {statistics_period}')
            #outgoing_queue.put([end_idx + 1, None, None, None, None, None])
            '''packed_data = serialize_and_compress(end_idx + 1, [None, None, None], None, None, None, None)
            outgoing_queue.put(packed_data)'''

            cycle_count = 0
        elif performance_data_store.steady_state and shock_manager.is_trigger_override():
            end_idx = shock_manager.decide_k(args.ppl, end_idx)
            end_idx_buff = end_idx + 1
            shock_manager.reset_history()
            logger.log(f'NNNNNNNNNNNNNNNNNNNNNNNNNN')
            logger.log(f'NNNNNNNNNNNNNNNNNNNNNNNNNN')
            logger.log(f'NNNNNNNNNNNNNNNNNNNNNNNNNN')
            logger.log(f'NNNNNNNNNNNNNNNNNNNNNNNNNN')
            logger.log(f'NNNNNNNNNNNNNNNNNNNNNNNNNN')
            logger.log(f'NNNNNNNNNNNNNNNNNNNNNNNNNN')
            logger.log(f'new end: {end_idx}')

            outgoing_queue.put(['opt', end_idx + 1])

            performance_data_store._statisitc_period = max(10, math.floor((performance_data_store._statisitc_period * 2 / 3) / 2) * 2)
            performance_data_store.new_record_count = 0
            performance_data_store.data_storage.clear()
            logger.log(f'new period: {performance_data_store._statisitc_period}')
            cycle_count = 0



        '''# test...
        end_idx = 4
        end_idx_buff = 5
        # test...'''
        lm_head, _ = get_lm_head_idx(end_idx)
        if not lm_head == head_idx:
            head_idx, lm_models = load_lm_head(args.ckpt_dir_hf_sep, end_idx, device, cache_dir="llm_weights")

        #if end_idx_buff < end_idx and end_idx_buff + 3 <= max_layers:  #add buffer
        if end_idx_buff < end_idx and end_idx_buff < max_layers:
            models, end_idx_buff = layer_reallocation(1, start_idx, end_idx_buff, max_layers, models)
        while end_idx_buff > end_idx + 3:  #remove buffer
            models, end_idx_buff = layer_reallocation(2, start_idx, end_idx_buff, max_layers, models)

        gc.collect()
        torch.cuda.empty_cache()


    print('end T2...')




if __name__ == '__main__':
    print('?????')
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    for key in config:
        for k, v in config[key].items():
            setattr(args, k, v)

    print('config type: ', args.config)

    device = torch.device("cuda")


    logger.log(f'!!!?!!??!!?!?:  {lm_manager.get_all_exit_rates()}')

    max_layers = args.max_layers
    start_idx = args.start_idx
    end_idx = args.end_idx
    end_idx_buff = args.end_idx_buff
    head_idx = 2

    '''#test...
    end_idx = 4
    end_idx_buff = 5
    # test...'''

    performance_data_store.statistic_period = 20
    performance_data_store.end_idx = end_idx
    performance_data_store.end_idx_buff = end_idx_buff

    models = load_model(args.ckpt_dir_hf_sep, start_idx, end_idx_buff, device)
    _, lm_models = load_lm_head(args.ckpt_dir_hf_sep, head_idx, device, cache_dir="llm_weights")

    tokenizer = LlamaTokenizer.from_pretrained(args.ckpt_dir_hf, use_fast=False)

    n_sample = 10
    batch_num = 30
    #seed = random.seed(time.time())
    seed = 0
    seqlen = 1024
    mode = 2
    bs = 1


    # Create and start threads
    thread3 = threading.Thread(target=data_producer, args=[batch_num, n_sample, seed, seqlen, bs, tokenizer, mode, device], kwargs={
                                                                                            "distribution": "exponential",
                                                                                            "dist_args": {"scale": 0.8}
                                                                                            })
    thread1 = threading.Thread(target=task1_data_sending, args=[args])
    #thread1 = threading.Thread(target=task1_data_sending_direct, args=[args])
    #thread1 = threading.Thread(target=task1_data_sending_multi, args=[args])
    thread2 = threading.Thread(target=task2_computation,
                               args=[models, lm_models, start_idx, performance_data_store.end_idx, performance_data_store.end_idx_buff,
                                     head_idx, max_layers, batch_num, device])
    #thread3 = threading.Thread(target=data_producer, args=[models, test_loader, bs, device])
    thread1.start()
    thread2.start()
    thread3.start()

    # Wait for both threads to finish (optional)
    thread1.join()
    thread2.join()
    thread3.join()