import time
import pickle
import lz4.frame
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from queue import Queue
import threading
import concurrent.futures
import uuid

# 控制最大同時 HTTP 處理請求數量
MAX_CONCURRENT_REQUESTS = 2
semaphore = threading.Semaphore(MAX_CONCURRENT_REQUESTS)

# 內部資料 queue
incoming_queue = Queue()
outgoing_map = {}
outgoing_map_lock = threading.Lock()

# background 處理 worker 數量（GPU 線程）
processing_pool = concurrent.futures.ThreadPoolExecutor(max_workers=3)

def get_in_queue_data():
    while incoming_queue.empty():
        time.sleep(0.001)
    return incoming_queue.get()

def get_out_queue_len():
    return len(outgoing_map)

def set_outgoing_result(request_id, result):
    with outgoing_map_lock:
        outgoing_map[request_id] = result

    print(f"[{time.strftime('%X')}] [SET RESULT] thread={threading.current_thread().name} request_id={request_id}, result={result}")

def wait_for_result(request_id):
    while True:
        with outgoing_map_lock:
            if request_id in outgoing_map:
                return outgoing_map.pop(request_id)
        time.sleep(0.001)

class S(BaseHTTPRequestHandler):
    sleep_time = 0

    def _set_headers(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/octet-stream')
        self.end_headers()

    def do_POST(self):
        print('NaNI???????????')
        with semaphore:
            try:
                content_length = int(self.headers['Content-Length'])

                post_data = self.rfile.read(content_length)
                decompress_data = lz4.frame.decompress(post_data)
                del post_data
                decrypt_data = pickle.loads(decompress_data)
                del decompress_data

                # 新增唯一 request_id（例如 input 的 idx）
                request_id = uuid.uuid4().hex  # 假設格式為 [..., idx, ...]

                # 把資料送到 incoming_queue 供主線程處理
                incoming_queue.put((request_id, decrypt_data))

                # 等待背景計算完成的結果
                result = wait_for_result(request_id)

                time.sleep(S.sleep_time)
                self._set_headers()
                self.send_response(200)
                self.end_headers()

                newx = pickle.dumps([result, 'Data received successfully!'])
                self.wfile.write(newx)

            except Exception as e:
                print(f"[ERROR] POST failed: {e}")
                self.send_response(500)
                self.end_headers()

def run(server_ip='', port=80):
    server_address = (server_ip, port)
    httpd = ThreadingHTTPServer(server_address, S)
    print(f'Starting threaded HTTP server on {server_ip}:{port}...')
    httpd.serve_forever()
