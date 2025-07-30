#!/usr/bin/env python3
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
import pickle
import argparse
import yaml
import http_sender
from queue import Queue
import multiprocessing
import lz4.frame

#incoming_queue = []
#outgoing_queue = []

incoming_queue = Queue()
outgoing_queue = Queue()

def get_in_queue_data():
    '''if len(incoming_queue) > 0:
        return incoming_queue[0]
    else:
        return []'''
    while incoming_queue.empty():
        time.sleep(0.001)

    print('http receiver incoming queue size: ', incoming_queue.qsize())
    return incoming_queue.get()

def get_in_queue_len():
    return incoming_queue.qsize()

def get_out_queue_data():
    '''if len(incoming_queue) > 0:
        return incoming_queue[0]
    else:
        return []'''
    while outgoing_queue.empty():
        time.sleep(0.005)

    print('http receiver returning queue size: ', outgoing_queue.qsize())
    return outgoing_queue.get()

def set_outgoing_queue(outputs):
    #outgoing_queue.append(outputs)
    outgoing_queue.put(outputs)
def pop_incoming_queue():
    incoming_queue.pop(0)


class S(BaseHTTPRequestHandler):
    sleep_time = 0
    def _set_headers(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()

    def do_GET(self):
        self._set_headers()

    def do_HEAD(self):
        self._set_headers()

    def do_POST(self):
        #print('receive POST:')
        start_time = time.time()
        content_length = int(self.headers['Content-Length'])

        post_data = self.rfile.read(content_length)
        decompress_data = lz4.frame.decompress(post_data)
        self._set_headers()
        decrypt_data = pickle.loads(decompress_data)


        '''post_data = self.rfile.read(content_length)
        self._set_headers()

        decrypt_data = pickle.loads(post_data)'''

        #if decrypt_data[0] == 'communication':
            #S.sleep_time = decrypt_data[1]
            #set_outgoing_queue(['T'])


        incoming_queue.put(decrypt_data)
        end_time = time.time()
        #print('server receiving time: ', end_time - start_time)

        time.sleep(S.sleep_time)

        self.send_response(200)
        self.end_headers()

        output_message = outgoing_queue.get()
        print('http returning: ', output_message)

        newx = pickle.dumps([output_message, 'Data received successfully!'])
        self.wfile.write(newx)

        #self.return_message()


def run(server_class=HTTPServer, handler_class=S, server_ip='', port=80):
    #server_address = ('10.7.48.14', port)
    server_address = (server_ip, port)
    httpd = server_class(server_address, handler_class)
    print('Starting httpd...')
    httpd.serve_forever()


if __name__ == "__main__":
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    for key in config:
        for k, v in config[key].items():
            setattr(args, k, v)

    print('port: ', args.server_port)
    run(port=args.server_port)


'''if __name__ == "__main__":
    from sys import argv

    if len(argv) == 2:
        run(port=int(argv[1]))
    else:
        run()'''