# this code is used for seperating the weights into small pieces and store them into seperated .pt files. One time usage.
# receive original data and prune or w/o prune
import gc
import tracemalloc
import math
import threading

import torch
import time
from pathlib import Path
import argparse
import random


#import http_receiver
import http_receiver_2 as http_receiver
import http_sender_gateway
#import http_sender_gateway2 as http_sender_gateway

from safetensors.torch import save_file
from transformers import PreTrainedTokenizerFast, LlamaTokenizer, AutoModelForCausalLM, LlamaConfig, AutoConfig

from multiprocessing import set_start_method
import sys
from natsort import natsorted
import os

from lm_head_manager import LMHeadManager
from model_hf import LlamaForCausalLM, LlamaForCausalLM_emb, LlamaForCausalLM_layer_0, LlamaForCausalLM_norm, \
    LlamaForCausalLM_linear
import yaml
from queue import Queue

from calculate_edge_opt_adptive import *
#from calculate_opt import Calcualte_opt, find_row
from early_exit import early_exit_cpu, early_exit_cuda, early_exit_lm_head
from predictive_splitting_manager_2 import EdgeSplittingManagerPool
from timestamp_manager import Timestamp_manager
from threading import current_thread, Thread
from multiprocessing import current_process
from global_initial_estimator import GlobalInitialStageEstimator
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
head_names = [1, 2, 3, 4, 6, 8, 10]
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
    (6, 30): {'a': 119, 'b': 31,  'gamma': 1.0},
    (8, 10): {'a': 243, 'b': 230, 'gamma': 1.0},
    (8, 20): {'a': 322, 'b': 106, 'gamma': 1.0},
    (8, 30): {'a': 302, 'b': 31,  'gamma': 1.0},
    (10, 10): {'a': 135,  'b': 244, 'gamma': 1.0},
    (10, 20): {'a': 248, 'b': 139, 'gamma': 1.0},
    (10, 30): {'a': 321, 'b': 81,  'gamma': 1.0},

}
lm_manager = LMHeadManager(head_names, ppl_list, init_params, logger)
#shock_manager = PredictiveSplittingManager(lm_manager, logger, shock_alpha=1.5, window_size=5, shock_threshold=3)
edgeSplittingManagerPool = EdgeSplittingManagerPool(34, lm_manager, logger)
global_initial_estimator = GlobalInitialStageEstimator(lm_manager, logger, 34)
#incoming_queue = Queue()
outgoing_queue_forward = Queue()
performance_data_store = PerformanceDataStore(edgeSplittingManagerPool, global_initial_estimator, logger)
timestamp_manager = Timestamp_manager(logger)
nsamples = 0
temp = []



#def layer_reallocation(type, start_idx, end_idx_buff, max_layers, models):
def layer_reallocation(type, start_idx, end_idx, max_layers, models):
    end_idx_buff = min(end_idx + 1, max_layers)
    if type == 1:  # add buffer layers
        # print('increase buffer')
        config, kwargs = AutoConfig.from_pretrained(
            args.ckpt_dir_hf_sep,
            return_unused_kwargs=True
        )
        # print('config: ', config)

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
            # print('i: ', i)
            try:
                if i == 0:
                    models.append(LlamaForCausalLM_emb(config))
                    models[i].load_state_dict(checkpoint_list[i - start_idx], strict=True)
                    models[0].to(device)
                    models[i].eval()
                elif i == 33:
                    models.append((LlamaForCausalLM_norm(config)))
                    models[i].load_state_dict(checkpoint_list[i - start_idx], strict=True)
                    models[33].to(device)
                    models[i].eval()

                elif i == 34:
                    models.append((LlamaForCausalLM_linear(config)))
                    models[i].load_state_dict(checkpoint_list[i - start_idx], strict=True)
                    models[34].to(device)
                    models[i].eval()
                else:
                    models.append(LlamaForCausalLM_layer_0(config))
                    models[i].load_state_dict(checkpoint_list[i - start_idx], strict=True)
                    models[i].to(device)
                    models[i].eval()
            except:
                end_idx_buff = i - 1
                break

    if type == 2: # drop layers
        #print('decrease buffer')
        models = models[:-1]
        end_idx_buff = end_idx_buff - 1

    if type == 3:   #reallocate model
        #print('increase buffer')
        config, kwargs = AutoConfig.from_pretrained(
            args.ckpt_dir_hf,
            return_unused_kwargs=True
        )
        #print('config: ', config)

        checkpoint_list = []
        checkpoints = sorted(Path(args.ckpt_dir_hf_sep).glob("consolidated.*.pth"))
        checkpoints = natsorted(checkpoints)
        assert len(checkpoints) > 0, f"no checkpoint files found in {args.ckpt_dir_hf_sep}"

        start_idx_buff = max(0, start_idx - 2)
        #print('FFFFFFFFFFff: ', max_layers)
        checkpoints = checkpoints[start_idx_buff : end_idx_buff + 1]
        checkpoint_idx = start_idx_buff

        '''print('start idxzzzz: ', start_idx_buff)
        for layer in models:
            print('layer: ', layer)

        print('end idx buff: ', end_idx_buff)'''
        logger.log(f'start_idx_buff: {start_idx_buff}')
        logger.log(f'end_idx_buff: {end_idx_buff}')
        logger.log(f'max_layer: {max_layers}')
        logger.log(f'models length: {len(models)}')
        for checkpoint in checkpoints:
            print('checkpoint idx: ', checkpoint_idx)
            logger.log(f'checkpoint idx: {checkpoint_idx}')
            logger.log(f'checkpoint: {checkpoint}')
            if len(models) > checkpoint_idx and models[checkpoint_idx] is None:
                logger.log(f'A')
                ckpt_path = checkpoint
                checkpoint_list.append(torch.load(ckpt_path, map_location="cpu"))
            elif len(models) <= checkpoint_idx:
                logger.log(f'B')
                ckpt_path = checkpoint
                checkpoint_list.append(torch.load(ckpt_path, map_location="cpu"))

            '''if checkpoint_idx > end_idx_buff:
                logger.log(f'A')
                ckpt_path = checkpoint
                checkpoint_list.append(torch.load(ckpt_path, map_location="cpu"))
            #elif models[checkpoint_idx] is None:
            elif len(models) <= checkpoint_idx:
                logger.log(f'B')
                ckpt_path = checkpoint
                checkpoint_list.append(torch.load(ckpt_path, map_location="cpu"))'''

            checkpoint_idx = checkpoint_idx + 1


        #end_idx_buff = max_layers


        if device.type == 'cuda':
            torch.set_default_tensor_type(torch.cuda.HalfTensor)
        else:
            torch.set_default_tensor_type(torch.BFloat16Tensor)

        models = models[:end_idx_buff]

        logger.log(f'model length: {len(models)}')
        logger.log(f'start idx buff: {start_idx_buff}')
        print('model length: ', len(models))
        print('start idx buff: ', start_idx_buff)
        checkpoint_idx = 0
        for i in range(0, end_idx_buff + 1):
            if i < start_idx_buff and len(models) > i:
                logger.log(f'i: {i} -> None')
                models[i] = None
                continue
            elif i < start_idx_buff:
                logger.log(f'i: {i} -> None')
                models.append(None)
                continue
            #print('i: ', i)
            load_layer = False
            try:
                if i == 0:
                    if i >= len(models):
                        models.append(LlamaForCausalLM_emb(config))
                        load_layer = True
                    elif models[i] is None:
                        models[i] = LlamaForCausalLM_emb(config)
                        load_layer = True

                    if load_layer is True:
                        logger.log(f'i: {i} -> emb')
                        models[i].load_state_dict(checkpoint_list[checkpoint_idx], strict=True)
                        models[0].to(device)
                        models[i].eval()
                        checkpoint_idx = checkpoint_idx + 1
                elif i == 33:
                    if i >= len(models):
                        models.append((LlamaForCausalLM_norm(config)))
                        load_layer = True
                    elif models[i] is None:
                        models[i] = LlamaForCausalLM_norm(config)
                        load_layer = True

                    if load_layer is True:
                        logger.log(f'i: {i} -> 33')
                        models[i].load_state_dict(checkpoint_list[checkpoint_idx], strict=True)
                        models[33].to(device)
                        models[i].eval()
                        checkpoint_idx = checkpoint_idx + 1
                elif i == 34:
                    if i >= len(models):
                        models.append((LlamaForCausalLM_linear(config)))
                        load_layer = True
                    elif models[i] is None:
                        models[i] = LlamaForCausalLM_linear(config)
                        load_layer = True

                    if load_layer is True:
                        logger.log(f'i: {i} -> 34')
                        models[i].load_state_dict(checkpoint_list[checkpoint_idx], strict=True)
                        models[34].to(device)
                        models[i].eval()
                        checkpoint_idx = checkpoint_idx + 1
                else:
                    if i >= len(models):
                        models.append((LlamaForCausalLM_layer_0(config)))
                        load_layer = True
                    elif models[i] is None:
                        models[i] = LlamaForCausalLM_layer_0(config)
                        load_layer = True

                    if load_layer is True:
                        logger.log(f'i: {i} -> L')
                        models[i].load_state_dict(checkpoint_list[checkpoint_idx], strict=True)
                        models[i].to(device)
                        models[i].eval()
                        checkpoint_idx = checkpoint_idx + 1
            except:
                logger.log(f'expect!!')
                end_idx_buff = i - 1
                break
        #prune_wanda_allocation(args, models, tokenizer, device=torch.device("cuda:0"))
    if type == 4:   #reload the whole model
        load_model(args.ckpt_dir_hf_sep, 0, end_idx_buff, torch.device("cuda:0"))
    if type == 5:   #drop early layer
        for i in range(0, start_idx - 2):
            logger.log(f'i: {i}')
            models[i] = None

    '''for i in range(0, len(models)):
                model = models[i]
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        print(name, param.data)'''
    gc.collect()
    torch.cuda.empty_cache()

    '''logger.log(f'show model start...')
    for model in models:
        logger.log(f'model: {model}')
    logger.log(f'show model end...')'''
    return models, end_idx_buff


def load_model(checkpoints_dir, start_idx, end_idx, device):
    config, kwargs = AutoConfig.from_pretrained(
        args.ckpt_dir_hf,
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
        torch.set_default_tensor_type(torch.cuda.HalfTensor)
    else:
        torch.set_default_tensor_type(torch.BFloat16Tensor)

    models = []
    for i in range(0, start_idx):
        models.append(None)

    for i in range(start_idx, end_idx + 1):
        #print('check point list [i]: ', checkpoint_list[i])
        if i == 0:
            models.append(LlamaForCausalLM_emb(config))
            models[i].load_state_dict(checkpoint_list[i], strict=True)
            models[0].to(device)
            models[i].eval()
        elif i == 33:
            models.append((LlamaForCausalLM_norm(config)))
            models[i].load_state_dict(checkpoint_list[i], strict=True)
            models[33].to(device)
            models[i].eval()

        elif i == 34:
            models.append((LlamaForCausalLM_linear(config)))
            models[i].load_state_dict(checkpoint_list[i], strict=True)
            models[34].to(device)
            models[i].eval()
        else:
            models.append(LlamaForCausalLM_layer_0(config))
            models[i].load_state_dict(checkpoint_list[i], strict=True)
            models[i].to(device)
            models[i].eval()


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

    lm_head, lm_head_idx = utils.get_lm_head_idx(end_idx)

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
        torch.cuda.empty_cache()
        if i == 0:
            lm_models.append((LlamaForCausalLM_norm(config)))
            lm_models[i].load_state_dict(checkpoint_list[i], strict=True)
            lm_models[i].to(device)
            lm_models[i].eval()

        else:
            lm_models.append((LlamaForCausalLM_linear(config)))
            lm_models[i].load_state_dict(checkpoint_list[i], strict=True)
            lm_models[i].to(device)
            lm_models[i].eval()

    gc.collect()

    return lm_head, lm_models


def task1_data_receiving(args):
    pid = os.getpid()
    curr_thread = current_thread().name
    curr_process = current_process().name
    print(f'{pid} with thread {curr_thread}, with process: {curr_process} Started')
    print('T1 do nothing!')

    while 1:
        http_receiver.run(port=args.gateway_port)

def task1_data_sending(args):
    while 1:
        timeout_count = 0

        tracemalloc.start()
        # take before snapshot
        snapshot_before = tracemalloc.take_snapshot()

        start_comp_time = time.time()

        #print('zzz', calculate_opt.steady_state)
        #while outgoing_queue_forward.empty() and incoming_queue.qsize() > 0 and calculate_opt.steady_state:
        #while outgoing_queue.empty() and input_queue.qsize() > 0:
        #while outgoing_queue_forward.qsize() < 3 and incoming_queue.qsize() > 0 and performance_data_store.steady_state:
        logger.log(f'queue size t1: {http_receiver.incoming_queue.qsize()}')
        while http_receiver.incoming_queue.qsize() > 0 and performance_data_store.steady_state:
            logger.log(f'AAAAAAAAAAAAAAAA')
            timeout_count = timeout_count + 1

            start_time = time.time()
            #print('outgoing queue size: ', outgoing_queue.qsize())

            if http_receiver.incoming_queue.qsize() > 0: #and calculate_opt.incoming_count + 2 >= calculate_opt.outgoint_count:
                idx = http_receiver.incoming_queue.qsize()
                timestamp_manager.start_times = (idx, start_time)

                input = http_receiver.get_in_queue_data()

                if input[0] == 'gateway' or input[0] == 'communication' or input[0] == 'server' or input[0] == 'opt':
                    logger.log(f'I think this is where the error comes from...')
                    http_receiver.incoming_queue.put(input)
                    continue

                start_idx = input[0]
                out = input[1]
                ids = input[2]
                mask = input[3]
                idx = input[4]

                #if received origina data
                if start_idx == 0:
                    outgoing_queue_forward.put([0, out, None, None, idx, 0, 0])
                else:
                    outgoing_queue_forward.put([start_idx, out, ids, mask, idx, 0, start_idx])


                end_time = time.time()
                #print('client computation time: ', end_time - start_time)
                # calculate_opt.client_comp_statistics = (-1, end_idx_buff, end_time - start_time)
                print('server idle!')
                logger.log(f'server idle!')
            #else:
            #    break


        data = outgoing_queue_forward.get()
        #print('data: ', data)
        performance_data_store.outgoing_count = performance_data_store.outgoing_count + 1
        http_sender_gateway.send_data(args.server_ip, args.server_port, data, performance_data_store, timestamp_manager, logger)

        gc.collect()

        # take after snapshot
        snapshot_after = tracemalloc.take_snapshot()
        stats = snapshot_after.compare_to(snapshot_before, 'lineno')

        logger.log(f"[ Top memory usage T1 ]")
        for stat in stats[:10]:
            logger.log(f'{stat}')

def task2_computation(models, lm_models, start_idx, end_idx, early_idx_buff, end_idx_buff, max_layers, max_layer_amount, head_idx, tokenizer, device, is_dummy=True):
    pid = os.getpid()
    curr_thread = current_thread().name
    curr_process = current_process().name
    print(f'{pid} with thread {curr_thread}, with process: {curr_process} Started')
    print('T2 computaton...')
    cycle_count = 0
    input_count = 0
    layer_amount = 2
    start_idx_buff = start_idx
    opt_layer_amount = 2
    statistics_period = performance_data_store.statistic_period
    sleep_time_per_layer = 0
    is_exploring = True
    is_oom = False
    while(1):
        logger.log(f'XXXXXXXXXXXXXXXXXXXXXX')
        gc.collect()  # 手動觸發垃圾回收
        leaked_objs = gc.garbage  # 找出無法被釋放的物件
        logger.log(f"Leaked objects: {len(leaked_objs)}")
        for obj in leaked_objs:
            print(type(obj), repr(obj)[:200])
        logger.log(f'YYYYYYYYYYYYYYYYYYYYYY')
        tracemalloc.start()
        # take before snapshot
        snapshot_before = tracemalloc.take_snapshot()


        logger.log(f'queue size t2: {http_receiver.incoming_queue.qsize()}')
        #print('http sender outgoing queue size: ', outgoing_queue_forward.qsize())
        print('start time: ', time.time())
        input = http_receiver.get_in_queue_data()
        start_comp_time = time.time()
        #print('????: ', input)
        if input[0] == 'gateway':
            sleep_time_per_layer = input[1]
            print('sleep time: ', sleep_time_per_layer)
            http_receiver.set_outgoing_queue(['T'])
            continue
        if input[0] == 'communication':
            http_receiver.set_outgoing_queue(['T'])
            continue
        if input[0] == 'server':
            outgoing_queue_forward.put(['server', input[1]])
            continue
        if input[0] == 'opt' and not performance_data_store.steady_state:
            start_idx = input[1]
            logger.log(f'not steady opt start: {start_idx}')

            end_idx = global_initial_estimator.predict_best_m(args.ppl, input[1])
            #end_idx_buff = end_idx + 1
            logger.log(f'QQQQQQQQQQ end idx: {end_idx}')

            #max_layers = start_idx - 2 + max_layer_amount

            #models, end_idx_buff = layer_reallocation(3, start_idx, end_idx_buff, max_layers, models)
            models, end_idx_buff = layer_reallocation(3, start_idx, end_idx, max_layers, models)

            start_idx_buff = max(0, start_idx - 2)
            opt_layer_amount = end_idx - start_idx
            layer_amount = opt_layer_amount


            http_receiver.set_outgoing_queue(['T'])
            is_exploring = False
            continue
        elif input[0] == 'opt':
            start_idx = input[1]
            logger.log(f'steady opt start: {start_idx}')

            end_idx = global_initial_estimator.predict_best_m(args.ppl, input[1])
            #end_idx_buff = end_idx + 1
            logger.log(f'QQQQQQQQQQ end idx: {end_idx}')
            #max_layers = start_idx - 2 + max_layer_amount
            models, end_idx_buff = layer_reallocation(3, start_idx, end_idx, max_layers, models)
            # models, end_idx_buff = layer_reallocation(5, start_idx, end_idx_buff, max_layers, models)
            start_idx_buff = max(0, start_idx - 2)
            layer_amount = end_idx - start_idx

            http_receiver.set_outgoing_queue(['T'])
            #is_exploring = False
            continue
        elif performance_data_store.steady_state and not is_exploring:
            start_idx = input[0]
            logger.log(f'normal start: {start_idx}')
            result = performance_data_store.get_optimal_end_idx(start_idx)
            if result:
                logger.log(f'if result...')
                end_idx_temp, _ = result

                if is_oom:
                    end_idx = min(start_idx + layer_amount, end_idx_temp)
                logger.log(f'new end: {end_idx}')

                models, end_idx_buff = layer_reallocation(3, start_idx, end_idx, max_layers, models)
                start_idx_buff = max(0, start_idx - 2)
                layer_amount = end_idx - start_idx



        #if received original data
        start_idx = input[0]
        out = input[1]
        ids = input[2]
        mask = input[3]
        idx = input[4]
        is_early_exit = False

        logger.log(f'is exploring: {is_exploring}')

        #end recieved original data

        lm_head, _ = utils.get_lm_head_idx(end_idx)
        logger.log(f'new lm_head: {lm_head}')
        logger.log(f'old head_idx: {head_idx}')
        if not lm_head == head_idx:
            try:
                del lm_models, head_idx
            except:
                pass
            gc.collect()
            torch.cuda.empty_cache()
            head_idx, lm_models = load_lm_head(args.ckpt_dir_hf_sep, end_idx, device, cache_dir="llm_weights")


        is_oom = False
        logger.log(f'input...: {input}')


        print('start idx buff: ', start_idx_buff)
        print('end idx buff: ', end_idx_buff)
        print('start idx: ', start_idx)
        print('end idx: ', end_idx)
        logger.log(f'start_idx_buff: {start_idx_buff}')
        logger.log(f'end_idx_buff: {end_idx_buff}')
        logger.log(f'start_idx: {start_idx}')
        logger.log(f'end_idx: {end_idx}')

        #input = http_receiver.get_in_queue_data()
        #print('start compute time: ', time.time())
        start_time = time.time()

        # Forward pass through the model
        if start_idx < start_idx_buff or start_idx > end_idx:
            print('direct sent!')
            logger.log(f'direct sent!')

            #sending original data
            outgoing_queue_forward.put([start_idx, out, ids, mask, idx, 0, start_idx]) # forward the original input to the server
            '''start_idx_buff = max(0, start_idx - 2)
            end_idx = start_idx + 2
            layer_amount = 2'''
            continue


        with torch.no_grad():
            #if start_idx > 0 and start_idx <= max_layers and start_idx >= start_idx_buff:
            if start_idx >= start_idx_buff:
                #find opt
                # TODO
                logger.log(f'show model start...')
                for model in models:
                    logger.log(f'model: {model}')
                logger.log(f'show model end...')
                end_time = time.time()
                #print('0: ', end_time - start_time)
                for k in range(start_idx, end_idx + 1):
                    print('layer: ', k)
                    logger.log(f'k: {k}')
                    try:
                        time.sleep(sleep_time_per_layer)
                        if k == 0:
                            out, ids, mask = models[0](out)
                            continue

                        out, ids, mask = models[k](out.last_hidden_state, position_ids=ids, attention_mask=mask)
                        if k == head_idx:
                            try:
                                time.sleep(sleep_time_per_layer)
                                is_early_exit, lm_logits = early_exit_lm_head(lm_models, out, head_idx, args.ppl)
                                #print('is early: ', is_early_exit)
                            except Exception as e:
                                print('early oom!')
                                logger.log(f'early oom!')
                                is_oom = True
                                is_early_exit = False

                                end_idx = k

                            if is_early_exit:
                                timestamp_manager.end_times = (idx, time.time())
                                break

                    except Exception as e:
                        print('oom!!!')
                        logger.log(f'oom!!!')
                        logger.log(f'{e}')
                        is_oom = True

                        end_idx = k - 1

                        #print('updated end idx: ', end_idx)
                        break

            print('is early: ', is_early_exit)
            print('new ned: ', end_idx)
            logger.log(f'is early: {is_early_exit}')
            logger.log(f'new end: {end_idx}')

            if not is_early_exit and end_idx >= 33:
                start_time = time.time()
                lm_logits = models[33](out.last_hidden_state)

                if end_idx >=34:
                    start_time = time.time()
                    lm_logits = models[34](lm_logits)

        # take after snapshot
        snapshot_after = tracemalloc.take_snapshot()
        stats = snapshot_after.compare_to(snapshot_before, 'lineno')

        logger.log(f"[ Top memory usage T2]")
        for stat in stats[:10]:
            logger.log(f'{stat}')

        end_time = time.time()
        total_comp_time = time.time() - start_comp_time

        # print('out: ', out)
        print('end compute time: ', time.time())
        print('total computation time: ', total_comp_time)
        logger.log(f'total computation time: {total_comp_time}')


        '''if is_dummy:
            break'''
        '''if is_oom:
            end_idx = max(1, math.ceil((end_idx - start_idx) / 2 + start_idx))
            layer_amount = end_idx - start_idx
            end_idx_buff = end_idx + 1
            continue'''

        #finished process with early exit, return to the client
        if is_early_exit or end_idx >= 34:
            http_receiver.set_outgoing_queue([start_idx, total_comp_time, idx])
            performance_data_store.add_edge_server_info(datetime.now() + timedelta(milliseconds=50), start_idx, end_idx,
                                                        end_idx_buff, total_comp_time, head_idx, True)
        #finished the whole process, return to the client
        elif end_idx >= 34:
            http_receiver.set_outgoing_queue([start_idx, total_comp_time, idx])
            performance_data_store.add_edge_server_info(datetime.now() + timedelta(milliseconds=50), start_idx, end_idx,
                                                        end_idx_buff, total_comp_time, head_idx, False)
        #no layer was processed because of oom, direct sent!
        elif end_idx < 0:
            outgoing_queue_forward.put([0, out, None, None, idx, 0, 0])

            if is_oom:
                logger.log(f'oom A')
                end_idx = max(1, math.ceil((end_idx - start_idx) / 2 + start_idx))
                layer_amount = end_idx - start_idx
                #end_idx_buff = end_idx + 1
        elif end_idx < start_idx:
            outgoing_queue_forward.put([start_idx, out, ids, mask, idx, 0, start_idx])

            if is_oom:
                logger.log(f'oom B')
                end_idx = max(1, math.ceil((end_idx - start_idx) / 2 + start_idx))
                layer_amount = end_idx - start_idx
                #end_idx_buff = end_idx + 1
        #the process is executed normally
        elif not is_early_exit and end_idx < 34:
            '''if end_idx <= start_idx:
                outgoing_queue_forward.put([end_idx + 1, out, ids, mask, idx, total_comp_time, start_idx])
                performance_data_store.add_edge_server_info(datetime.now() + timedelta(milliseconds=50), end_idx,
                                                            end_idx, end_idx_buff, 0, head_idx, False)
                continue'''


            input_count = input_count + 1
            cycle_count = cycle_count + 1

            outgoing_queue_forward.put([end_idx + 1, out, ids, mask, idx, total_comp_time, start_idx])
            performance_data_store.add_edge_server_info(datetime.now() + timedelta(milliseconds=50), start_idx, end_idx, end_idx_buff, total_comp_time, head_idx, False)

            if is_oom:
                logger.log(f'oom C')
                end_idx = max(1, math.ceil((end_idx - start_idx) / 2 + start_idx))
                layer_amount = end_idx - start_idx
                #end_idx_buff = end_idx + 1
                #continue
                # is_oom = False
            else:
                existed_opt = performance_data_store.get_all_data_by_edge_server_start_index(start_idx)
                if len(existed_opt) == 0:
                    logger.log(f'no exist opt')
                    end_idx = start_idx + 1
                else:
                    logger.log(f'testing statistic period: {statistics_period}')
                    logger.log(f'testing cycle count: {cycle_count}')
                    logger.log(f'testing s-c: {statistics_period - cycle_count}')
                    if (input_count) % 4 == 0 and input_count < 24 and end_idx < max_layers and statistics_period <= 20:
                        print('testing higher value(i<12)')
                        logger.log(f'testing higher value(i<30)')

                        performance_data_store.max_layer_amount = layer_amount
                        layer_amount = layer_amount + 1
                        is_exploring = True

                    # if cycle_count > (statistics_period - 6) and input_count >= 12 and cycle_count % 2 == 0:
                    # if cycle_count > (statistics_period - 12) and input_count >= 24 and cycle_count % 4 == 0:
                    if cycle_count > (statistics_period - 12) and input_count >= 24 and (
                            statistics_period - cycle_count) % 4 == 0:
                        print('testing lower value (i>30)')
                        logger.log(f'testing lower value (i>30)')

                        layer_amount = max(1, layer_amount - 1)
                        is_exploring = True


                    # if cycle_count == (statistics_period - 6) and input_count >= 12 and end_idx < max_layers and cycle_count % 2 == 0:
                    # if cycle_count == (statistics_period - 12) and input_count >= 24 and end_idx < max_layers and cycle_count % 4 == 0:
                    if cycle_count == (statistics_period - 12) and input_count >= 24 and end_idx < max_layers and (
                            statistics_period - cycle_count) % 4 == 0:
                        print('testing higher value (i>30)')
                        logger.log(f'testing higher value (i>30)')

                        performance_data_store.max_layer_amount = layer_amount
                        layer_amount = layer_amount + 1
                        is_exploring = True


                    end_idx = start_idx + layer_amount

        #if (input_count) % 10 == 0:
        if performance_data_store.new_record_count >= statistics_period:
            #print(performance_data_store.get_all_data_by_type("edge_to_server"))
            end_idx, end_idx_buff, statistics_period = calculate_edge_server_opt(performance_data_store, args.ppl, lm_manager, args.mode, edgeSplittingManagerPool, logger, start_idx)
            print('end_idx: ', end_idx)
            print('end_idx_buff: ', end_idx_buff)
            print('statistics_period: ', statistics_period)
            logger.log(f'calculate opt!!!')
            logger.log(f'end_idx: {end_idx}')
            logger.log(f'end_idx_buff: {end_idx_buff}')
            logger.log(f'statistics_period: {statistics_period}')
            if not end_idx:
                end_idx = global_initial_estimator.predict_best_m(args.ppl, start_idx)
                #end_idx_buff = end_idx + 1

            opt_layer_amount = end_idx - start_idx
            layer_amount = opt_layer_amount
            #end_idx_buff = min(max_layers, end_idx_buff)
            #while new_buff_idx < end_idx_buff:
            #    models, end_idx_buff = layer_reallocation(2, start_idx, end_idx_buff, max_layers, models)

            lm_head, _ = get_lm_head_idx(end_idx)
            if not lm_head == head_idx:
                head_idx, lm_models = load_lm_head(args.ckpt_dir_hf_sep, end_idx, device, cache_dir="llm_weights")
            cycle_count = 0
            is_exploring = False
        '''elif performance_data_store.steady_state and edgeSplittingManagerPool.is_trigger_override():
            end_idx = edgeSplittingManagerPool.decide_m(start_idx, end_idx, args.ppl)
            end_idx_buff = end_idx + 2
            logger.log(f'NNNNNNNNNNNNNNNNNNNNNNNNNN')
            logger.log(f'NNNNNNNNNNNNNNNNNNNNNNNNNN')
            logger.log(f'NNNNNNNNNNNNNNNNNNNNNNNNNN')
            logger.log(f'NNNNNNNNNNNNNNNNNNNNNNNNNN')
            logger.log(f'NNNNNNNNNNNNNNNNNNNNNNNNNN')
            logger.log(f'NNNNNNNNNNNNNNNNNNNNNNNNNN')
            logger.log(f'new end: {end_idx}')

            performance_data_store._statisitc_period = max(10, math.floor((performance_data_store._statisitc_period * 2 / 3) / 2) * 2)
            performance_data_store.new_record_count = 0
            performance_data_store.data_storage.clear()
            performance_data_store.data_storage = {
            "client_to_server": collections.defaultdict(collections.deque),
            "edge_to_server": collections.defaultdict(collections.deque)
            }
            logger.log(f'new period: {performance_data_store._statisitc_period}')
            cycle_count = 0
            is_exploring = False'''

        #max_layers = start_idx + max_layer_amount

        #if end_idx_buff < end_idx and end_idx_buff + 3 <= max_layers:  #add buffer
        logger.log(f'####################')
        logger.log(f'## end_idx_buff: {end_idx_buff}')
        logger.log(f'## end_idx: {end_idx}')
        logger.log(f'## max_layers: {max_layers}')
        logger.log(f'## start_idx: {start_idx}')
        logger.log(f'## start_idx_buff: {start_idx_buff}')

        '''if (end_idx_buff < end_idx and end_idx_buff < max_layers) or start_idx < start_idx_buff:
            logger.log(f'load model E')
            models, end_idx_buff = layer_reallocation(3, start_idx, end_idx_buff, max_layers, models)
        if start_idx_buff < start_idx - 2:
            models, end_idx_buff = layer_reallocation(5, start_idx, end_idx_buff, max_layers, models)
        while end_idx_buff > end_idx + 2:  #remove end buffer
            logger.log(f'drop layers...')
            models, end_idx_buff = layer_reallocation(2, start_idx, end_idx_buff, max_layers, models)'''

        if (end_idx_buff < end_idx and end_idx_buff < max_layers) or start_idx < start_idx_buff:
            logger.log(f'load model E')
            models, end_idx_buff = layer_reallocation(3, start_idx, end_idx, max_layers, models)
        if start_idx_buff < start_idx - 2:
            models, end_idx_buff = layer_reallocation(5, start_idx, end_idx, max_layers, models)
        while end_idx_buff > end_idx + 2:  #remove end buffer
            logger.log(f'drop layers...')
            models, end_idx_buff = layer_reallocation(2, start_idx, end_idx, max_layers, models)

        #models, end_idx_buff = layer_reallocation(5, start_idx, end_idx_buff, max_layers, models)

        torch.cuda.empty_cache()


    performance_data_store.statistic_period = statistics_period



    #print('round time: ', time.time() - start_time_0)


if __name__ == '__main__':
    set_start_method('spawn')
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    for key in config:
        for k, v in config[key].items():
            setattr(args, k, v)

    print('config type: ', args.config)
    torch.manual_seed(0)


    performance_data_store.statistic_period = 20
    end_idx_buff = args.end_idx_buff
    early_idx_buff = args.early_idx_buff
    max_layer_amount = args.max_layer_amount
    max_layers = args.max_layers
    start_idx = args.start_idx
    end_idx = args.end_idx
    head_idx = args.head_idx

    #allow_cuda = False
    #device = 'cuda' if torch.cuda.is_available() and allow_cuda else 'cpu'
    device = torch.device("cuda")
    models = load_model(args.ckpt_dir_hf_sep, early_idx_buff, end_idx_buff, device)
    _, lm_models = load_lm_head(args.ckpt_dir_hf_sep, head_idx, device, cache_dir="llm_weights")
    tokenizer = LlamaTokenizer.from_pretrained(args.ckpt_dir_hf, use_fast=False)

    print("loading success")
    # Create and start threads


    start_time = time.time()
    thread1 = threading.Thread(target=task1_data_receiving, args=[args])
    thread2 = threading.Thread(target=task1_data_sending, args=[args])
    #(models, lm_models, start_idx, end_idx, early_idx_buff, end_idx_buff, max_layers, max_layer_amount, head_idx, tokenizer, device, is_dummy=True)
    thread3 = threading.Thread(target=task2_computation, args=[models, lm_models, start_idx, end_idx, early_idx_buff, end_idx_buff, max_layers, max_layer_amount, head_idx, tokenizer, device, False])

    thread1.start()
    thread2.start()
    thread3.start()

    # Wait for both threads to finish (optional)
    thread1.join()
    thread2.join()
    thread3.join()
    print('total_time: ', time.time() - start_time)
