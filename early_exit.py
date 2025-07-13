
import torch


def early_exit_cpu(models, out, ids, mask):
    threshold = 0.7
    temperature = 0.6


    '''print('ids shape: ', ids.shape)
    print('out shape: ', out.last_hidden_state.shape)
    print('maks shape: ', mask.shape)'''

    logits_norm = models[-2](out.last_hidden_state.detach().cpu())
    logits_linear = models[-1](logits_norm.detach().cpu())
    probs = torch.softmax(logits_linear / temperature, dim=-1)
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)

    probs = probs.squeeze(0)
    probs_sort = probs_sort.squeeze(0)
    probs_idx = probs_idx.squeeze(0)
    probs_sum = 0

    #print('probs sort: ', probs_sort)
    #prob_max_list = []
    pruned_data_list = []
    pruned_data_idx_list = []
    idx_diff = 0
    for i in range (0, len(probs)):
        #print('i: ', i)
        #prob_max_list.append(torch.max(probs_sort[i]).item())
        #probs_sum = probs_sum + torch.max(probs_sort[i]).item()
        if torch.max(probs_sort[i]).item() >= threshold:
            pruned_data_list.append(out.last_hidden_state)
            pruned_data_idx_list.append(i)

            #print('remove: ', i - idx_diff)
            ids = torch.cat((ids[:, :i - idx_diff], ids[:, i - idx_diff + 1:]), dim=1)
            out.last_hidden_state = torch.cat((out.last_hidden_state[:, :i - idx_diff, :], out.last_hidden_state[:, i - idx_diff + 1:, :]), dim=1)
            mask = torch.cat((mask[:, :, :i - idx_diff, :], mask[:, :, i - idx_diff + 1:, :]), dim=2)

            idx_diff = idx_diff + 1

    print('# rows droped: ', idx_diff)
    '''print('ids: ', ids)
    print('out: ', out.last_hidden_state)
    print('mask: ', mask)
    print('ids shape: ', ids.shape)
    print('out shape: ', out.last_hidden_state.shape)
    print('maks shape: ', mask.shape)'''

    #print('probs max: ', prob_max_list)
    #probs_avg = probs_sum / 1024
    #print('probs_avg: ', probs_avg)



    '''for prompt in out.last_hidden_state[0]:
        print('prompt: ', prompt)

        logits_norm = models[-2](out.last_hidden_state)
        logits_linear = models[-1](logits_norm)
        probs = torch.softmax(logits_linear[:, -1] / temperature, dim=-1)
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        
        print()
        print(torch.max(probs_sort).item())'''

    return out, ids, mask


def early_exit_cuda(models, out, ids, mask):
    threshold = 0.7
    temperature = 0.6

    '''print('ids shape: ', ids.shape)
    print('out shape: ', out.last_hidden_state.shape)
    print('maks shape: ', mask.shape)'''

    logits_norm = models[-2](out.last_hidden_state.detach())
    logits_linear = models[-1](logits_norm.detach())
    probs = torch.softmax(logits_linear / temperature, dim=-1)
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)

    probs = probs.squeeze(0)
    probs_sort = probs_sort.squeeze(0)
    probs_idx = probs_idx.squeeze(0)
    probs_sum = 0

    # print('probs sort: ', probs_sort)
    # prob_max_list = []
    pruned_data_list = []
    pruned_data_idx_list = []
    idx_diff = 0
    for i in range(0, len(probs)):
        # print('i: ', i)
        # prob_max_list.append(torch.max(probs_sort[i]).item())
        # probs_sum = probs_sum + torch.max(probs_sort[i]).item()
        if torch.max(probs_sort[i]).item() >= threshold:
            pruned_data_list.append(out.last_hidden_state)
            pruned_data_idx_list.append(i)

            # print('remove: ', i - idx_diff)
            ids = torch.cat((ids[:, :i - idx_diff], ids[:, i - idx_diff + 1:]), dim=1)
            out.last_hidden_state = torch.cat(
                (out.last_hidden_state[:, :i - idx_diff, :], out.last_hidden_state[:, i - idx_diff + 1:, :]), dim=1)
            #mask = torch.cat((mask[:, :, :i - idx_diff, :], mask[:, :, i - idx_diff + 1:, :]), dim=2)

            idx_diff = idx_diff + 1

    print('# rows droped: ', idx_diff)
    '''print('ids: ', ids)
    print('out: ', out.last_hidden_state)
    print('mask: ', mask)
    print('ids shape: ', ids.shape)
    print('out shape: ', out.last_hidden_state.shape)
    print('maks shape: ', mask.shape)'''

    # print('probs max: ', prob_max_list)
    # probs_avg = probs_sum / 1024
    # print('probs_avg: ', probs_avg)

    '''for prompt in out.last_hidden_state[0]:
        print('prompt: ', prompt)

        logits_norm = models[-2](out.last_hidden_state)
        logits_linear = models[-1](logits_norm)
        probs = torch.softmax(logits_linear[:, -1] / temperature, dim=-1)
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)

        print()
        print(torch.max(probs_sort).item())'''

    return out, ids, mask


def early_exit_cuda_ppl_test(models, out, ids, mask):
    threshold = 1
    delta = 0
    temperature = 0.6

    '''print('ids shape: ', ids.shape)
    print('out shape: ', out.last_hidden_state.shape)
    print('maks shape: ', mask.shape)'''

    logits_norm = models[-2](out.last_hidden_state.detach())
    logits_linear = models[-1](logits_norm.detach())
    probs = torch.softmax(logits_linear / temperature, dim=-1)
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)

    probs = probs.squeeze(0)
    probs_sort = probs_sort.squeeze(0)
    probs_idx = probs_idx.squeeze(0)
    probs_sum = 0

    # print('probs sort: ', probs_sort)
    # prob_max_list = []
    pruned_data_list = []
    pruned_data_idx_list = []
    idx_diff = 0
    early_count = 0


    for i in range(0, len(probs)):
        if torch.max(probs_sort[i]).item() >= threshold:
            early_count = early_count + 1
            pruned_data_list.append(out.last_hidden_state[0][i - idx_diff])
            pruned_data_idx_list.append(i - idx_diff)


            ids = torch.cat((ids[:, :i - idx_diff], ids[:, i - idx_diff + 1:]), dim=1)
            out.last_hidden_state = torch.cat(
                    (out.last_hidden_state[:, :i - idx_diff, :], out.last_hidden_state[:, i - idx_diff + 1:, :]), dim=1)
            #mask = torch.cat((mask[:, :, :i - idx_diff, :], mask[:, :, i - idx_diff + 1:, :]), dim=2)

            idx_diff = idx_diff + 1

        print('early count: ', early_count)
        print('# rows droped: ', idx_diff)

    '''print('ids: ', ids)
    print('out: ', out.last_hidden_state)
    print('mask: ', mask)
    print('ids shape: ', ids.shape)
    print('out shape: ', out.last_hidden_state.shape)
    print('maks shape: ', mask.shape)'''

    # print('probs max: ', prob_max_list)
    # probs_avg = probs_sum / 1024
    # print('probs_avg: ', probs_avg)

    '''for prompt in out.last_hidden_state[0]:
        print('prompt: ', prompt)

        logits_norm = models[-2](out.last_hidden_state)
        logits_linear = models[-1](logits_norm)
        probs = torch.softmax(logits_linear[:, -1] / temperature, dim=-1)
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)

        print()
        print(torch.max(probs_sort).item())'''

    return out, ids, mask, pruned_data_idx_list, pruned_data_list



def early_exit_lm_cuda_ppl_test(models, lm_models, out, ids, mask):
    threshold = 0.9
    delta = 0
    temperature = 0.6

    logits_norm = models[-2](out.last_hidden_state.detach())
    logits_linear = lm_models[0](logits_norm.detach())
    print('logit size: ', logits_linear.shape)
    probs = torch.softmax(logits_linear / temperature, dim=-1)
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)

    probs = probs.squeeze(0)
    probs_sort = probs_sort.squeeze(0)
    probs_idx = probs_idx.squeeze(0)
    probs_sum = 0

    # print('probs sort: ', probs_sort)
    # prob_max_list = []
    pruned_data_list = []
    pruned_data_idx_list = []
    idx_diff = 0
    early_count = 0


    for i in range(0, len(probs)):
        if torch.max(probs_sort[i]).item() >= threshold:
            probs_sum = probs_sum + torch.max(probs_sort[i]).item()
            early_count = early_count + 1
            '''pruned_data_list.append(out.last_hidden_state[0][i - idx_diff])
            pruned_data_idx_list.append(i - idx_diff)

            ids = torch.cat((ids[:, :i - idx_diff], ids[:, i - idx_diff + 1:]), dim=1)
            out.last_hidden_state = torch.cat(
                    (out.last_hidden_state[:, :i - idx_diff, :], out.last_hidden_state[:, i - idx_diff + 1:, :]), dim=1)
            mask = torch.cat((mask[:, :, :i - idx_diff, :], mask[:, :, i - idx_diff + 1:, :]), dim=2)

            idx_diff = idx_diff + 1'''

    print('avg prob: ', probs_sum / 1024)
    print('early count: ', early_count)
    print('# rows droped: ', idx_diff)

    '''print('ids: ', ids)
    print('out: ', out.last_hidden_state)
    print('mask: ', mask)
    print('ids shape: ', ids.shape)
    print('out shape: ', out.last_hidden_state.shape)
    print('maks shape: ', mask.shape)'''

    # print('probs max: ', prob_max_list)
    # probs_avg = probs_sum / 1024
    # print('probs_avg: ', probs_avg)

    '''for prompt in out.last_hidden_state[0]:
        print('prompt: ', prompt)

        logits_norm = models[-2](out.last_hidden_state)
        logits_linear = models[-1](logits_norm)
        probs = torch.softmax(logits_linear[:, -1] / temperature, dim=-1)
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)

        print()
        print(torch.max(probs_sort).item())'''

    #return out, ids, mask, pruned_data_idx_list, pruned_data_list
    return early_count, logits_linear



def early_exit_lm_head(lm_models, out, lm_head, ppl):
    threshold = 0.5
    temperature = 0.5

    logits_norm = lm_models[0](out.last_hidden_state.detach())
    logits_linear = lm_models[1](logits_norm.detach())

    probs = torch.softmax(logits_linear / temperature, dim=-1)
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)

    probs = probs.squeeze(0)
    probs_sort = probs_sort.squeeze(0)
    probs_sum = 0
    early_count = 0


    for i in range(0, len(probs)):
        if torch.max(probs_sort[i]).item() >= threshold:
            probs_sum = probs_sum + torch.max(probs_sort[i]).item()
            early_count = early_count + 1


    if ppl == 10:
        if lm_head == 10:
            early_rate = 0.74
        elif lm_head == 8:
            early_rate = 0.745
        elif lm_head == 6:
            early_rate = 0.73
        elif lm_head == 4:
            early_rate = 0.72
        elif lm_head == 2:
            early_rate = 0.69
        elif lm_head == 1:
            early_rate = 0.67
    elif ppl == 20:
        if lm_head == 10:
            early_rate = 0.71
        elif lm_head == 8:
            early_rate = 0.707
        elif lm_head == 6:
            early_rate = 0.7
        elif lm_head == 4:
            early_rate = 0.69
        elif lm_head == 2:
            early_rate = 0.67
        elif lm_head == 1:
            early_rate = 0.654
    elif ppl == 30:
        if lm_head == 10:
            early_rate = 0.67
        elif lm_head == 8:
            early_rate = 0.665
        elif lm_head == 6:
            early_rate = 0.667
        elif lm_head == 4:
            early_rate = 0.667
        elif lm_head == 2:
            early_rate = 0.66
        elif lm_head == 1:
            early_rate = 0.644



    '''early_rate = 0.91
    if lm_head == 10:
        early_rate = 0.8944140625
    elif lm_head == 8:
        early_rate = 0.8916015625
    elif lm_head == 6:
        early_rate = 0.886484375
    elif lm_head == 4:
        early_rate = 0.8799106607
        #early_rate = 0.8794106607
    elif lm_head == 2:
        early_rate = 0.8663311749
    elif lm_head == 1:
        early_rate = 0.8490639077'''

    #return True, logits_linear
    print('rate????: ', early_count / 1024)
    if early_count / 1024 > early_rate:
        return True, logits_linear
    else:
        return False, logits_linear


def early_exit_lm_head_shift(lm_models, out, lm_head, ppl):
    threshold = 0.5
    temperature = 0.5

    logits_norm = lm_models[0](out.last_hidden_state.detach())
    logits_linear = lm_models[1](logits_norm.detach())

    probs = torch.softmax(logits_linear / temperature, dim=-1)
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)

    probs = probs.squeeze(0)
    probs_sort = probs_sort.squeeze(0)
    probs_sum = 0
    early_count = 0


    for i in range(0, len(probs)):
        if torch.max(probs_sort[i]).item() >= threshold:
            probs_sum = probs_sum + torch.max(probs_sort[i]).item()
            early_count = early_count + 1


    if ppl == 10:
        if lm_head == 10:
            early_rate = 0.74
        elif lm_head == 8:
            early_rate = 0.745
        elif lm_head == 6:
            early_rate = 0.6689
        elif lm_head == 3:
            early_rate = 0.65506
        elif lm_head == 4:
            early_rate = 0.66406
        elif lm_head == 2:
            early_rate = 0.66113
        elif lm_head == 1:
            early_rate = 0.65567
    elif ppl == 20:
        if lm_head == 10:
            early_rate = 0.71
        elif lm_head == 8:
            early_rate = 0.707
        elif lm_head == 6:
            early_rate = 0.63476
        elif lm_head == 4:
            early_rate = 0.63769
        elif lm_head == 2:
            early_rate = 0.64355
        elif lm_head == 1:
            early_rate = 0.63476
    elif ppl == 30:
        if lm_head == 10:
            early_rate = 0.67
        elif lm_head == 8:
            early_rate = 0.665
        elif lm_head == 6:
            early_rate = 0.60376
        elif lm_head == 4:
            early_rate = 0.61718
        elif lm_head == 2:
            early_rate = 0.63281
        elif lm_head == 1:
            early_rate = 0.62597


    #return True, logits_linear
    print('rate????: ', early_count / 1024)
    if early_count / 1024 > early_rate:
        return True, logits_linear
    else:
        return False, logits_linear

def early_exit_regression(lm_models, out, lm_head, ppl, threshold=0.9):
    threshold = 0.5
    temperature = 0.5

    print('head: ', lm_head)

    logits_norm = lm_models[0](out.last_hidden_state.detach())
    logits_linear = lm_models[1](logits_norm.detach())

    probs = torch.softmax(logits_linear / temperature, dim=-1)

    print('confidence!!: ', probs.max(dim=-1).values.flatten())
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)

    probs = probs.squeeze(0)
    probs_sort = probs_sort.squeeze(0)
    probs_sum = 0
    early_count = 0

    for i in range(0, len(probs)):
        if torch.max(probs_sort[i]).item() >= threshold:
            probs_sum = probs_sum + torch.max(probs_sort[i]).item()
            early_count = early_count + 1

    print('early count: ', early_count)
    print('early rate: ', early_count / 1024)

    if ppl == 10:
        if lm_head == 10:
            early_rate = 0.74
        elif lm_head == 8:
            early_rate = 0.745
        elif lm_head == 6:
            early_rate = 0.73
        elif lm_head == 4:
            early_rate = 0.72
        elif lm_head == 2:
            early_rate = 0.69
        elif lm_head == 1:
            early_rate = 0.67


    '''early_rate = 0.91
    if lm_head == 10:
        early_rate = 0.67
    elif lm_head == 8:
        early_rate = 0.665
    elif lm_head == 6:
        early_rate = 0.667
    elif lm_head == 4:
        early_rate = 0.667
        # early_rate = 0.8794106607
    elif lm_head == 2:
        early_rate = 0.67
    elif lm_head == 1:
        early_rate = 0.644
        #early_rate = 0.8490639077'''


    # return True, logits_linear
    print('rate????: ', early_count / 1024)
    if early_count / 1024 > early_rate:
        return True, logits_linear
    else:
        return False, logits_linear