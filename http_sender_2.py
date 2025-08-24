import http.client
import pickle
import time
import concurrent.futures
import yaml
import gc
import threading
from queue import Queue
import lz4.frame
from datetime import datetime, timedelta


print_lock = threading.Lock()
returning_queue = Queue()

def get_queue_data():
    '''if len(returning_queue) > 0:
        return returning_queue[0]
    else:
        return []'''
    #while returning_queue.empty():
    #    time.sleep(0.5)
    data = []
    while not returning_queue.empty():
        data.append(returning_queue.get())

    return data


def pop_incoming_queue():
    returning_queue.get()


def send_request(server_ip, server_port, text, performance_data_store, timestamp_manager, logger):
    start_time = time.time()
    start_idx = text[0]

    newx = lz4.frame.compress(pickle.dumps(text))
    # newx = pickle.dumps(text)
    total_size = len(newx)
    print('communication size: ', total_size)


    conn = http.client.HTTPConnection(server_ip, server_port)
    #conn = http.client.HTTPSConnection(server_ip)
    conn.connect()

    conn.putrequest('POST', '/')
    conn.putheader('Content-Type', 'application/octet-stream')
    conn.putheader('Content-Length', str(len(newx)))

    conn.endheaders()
    conn.send(newx)

    resp = conn.getresponse()

    resp_data = resp.readlines()
    # print('TTTTTTTTTTTTTTTT:', resp_data)
    resp_str = b''

    for i in range(4, len(resp_data)):
        resp_str = resp_str + resp_data[i]
    end_time2 = time.time()
    rtt = end_time2 - start_time

    try:
        # resp_message = [start_idx, total_comp_time, idx]
        resp_message = pickle.loads(resp_str)

        resp_message = resp_message[0]
        resp_message.append(rtt)  # resp_message = [start_idx, total_comp_time, idx, rtt(total time)]
        print('server side resp: ', resp_message)
        logger.log(f'server side resp: {resp_message}')

        if not (resp_message[0] == -1 or resp_message[0] == 'T'):
            timestamp_manager.end_times = (resp_message[2], end_time2)

        if not (resp_message[0] == 0 or resp_message[0] == -1 or resp_message[0] == 'T'):
            print('data stored!')
            performance_data_store.incoming_count = performance_data_store.incoming_count + 1
            performance_data_store.add_server_info(datetime.now() + timedelta(milliseconds=50), resp_message[0], 34,
                                                       resp_message[1], resp_message[3] - resp_message[1])
        # returning_queue.put(resp_message)
    except:
        print('error')
    # print('return message: ', resp_message[0])
    # returning_queue.append(resp_message)

    # print('client receiving time: ', end_time2 - start_time2)
    with print_lock:
        print('http receiving: ', start_idx, rtt, flush=True)
        print('rrt: ', rtt, flush=True)
    gc.collect()


    conn.close()



'''# 這邊是模擬你要送出幾個請求的 payloads
payloads = [b"data1", b"data2", b"data3"]
server_ip = "127.0.0.1"
server_port = 8080
client_id = "Client-001"
session_id = "Session-ABC123"

with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
    futures = []
    for i, payload in enumerate(payloads):
        tag = f"chunk-{i}"
        futures.append(executor.submit(send_request, server_ip, server_port, payload, client_id, session_id, tag))

    # 等所有任務完成
    concurrent.futures.wait(futures)
'''