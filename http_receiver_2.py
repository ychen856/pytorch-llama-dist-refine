import time
import pickle
import lz4.frame
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from queue import Queue
import threading

MAX_CONCURRENT_REQUESTS = 2
semaphore = threading.Semaphore(MAX_CONCURRENT_REQUESTS)
incoming_queue = Queue()
outgoing_queue = Queue()

def get_in_queue_data():
    while incoming_queue.empty():
        time.sleep(0.001)
    print('http receiver incoming queue size: ', incoming_queue.qsize())
    return incoming_queue.get()

def set_outgoing_queue(outputs):
    outgoing_queue.put(outputs)

def get_out_queue_data():
    while outgoing_queue.empty():
        time.sleep(0.005)
    print('http receiver returning queue size: ', outgoing_queue.qsize())
    return outgoing_queue.get()

class S(BaseHTTPRequestHandler):
    sleep_time = 0

    def _set_headers(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()

    def do_POST(self):
        start_time = time.time()

        with semaphore:
            try:

                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)

                decompress_data = lz4.frame.decompress(post_data)
                del post_data
                decrypt_data = pickle.loads(decompress_data)
                del decompress_data

                incoming_queue.put(decrypt_data)
                print(f'[{time.strftime("%X")}] Queue size after put: {incoming_queue.qsize()}')

                time.sleep(S.sleep_time)

                self.send_response(200)
                self.end_headers()

                output_message = outgoing_queue.get()
                print(f'[{time.strftime("%X")}] http returning: ', output_message)

                newx = pickle.dumps([output_message, 'Data received successfully!'])
                self.wfile.write(newx)

            except Exception as e:
                print(f"[ERROR] Exception in POST: {e}")
                self.send_response(500)
                self.end_headers()

def run(server_ip='', port=80):
    server_address = (server_ip, port)
    httpd = ThreadingHTTPServer(server_address, S)
    print(f'Starting threaded HTTP server on {server_ip}:{port}...')
    httpd.serve_forever()
