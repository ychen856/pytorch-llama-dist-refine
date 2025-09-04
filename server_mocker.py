import time
import http.server
import ssl
import sys
import os
import lz4.frame
import pickle
from http.server import BaseHTTPRequestHandler, HTTPServer

# 設定伺服器位址和連接埠
# 這是 Mocker 服務實際運行的位址和埠，它會從您的服務層接收請求
SERVER_HOST = ''
SERVER_PORT = 80  # 更改為一個常見的 HTTP 埠


class S(BaseHTTPRequestHandler):
    """
    處理來自服務層轉發過來的 POST 請求。
    """

    def do_POST(self):
        try:
            # 取得請求的內容長度
            content_length = int(self.headers.get('Content-Length', 0))

            # 讀取請求體
            post_data = self.rfile.read(content_length)

            # 嘗試解壓縮和反序列化，以確保數據是完整的
            try:
                decompress_data = lz4.frame.decompress(post_data)
                decrypt_data = pickle.loads(decompress_data)
                print(f"Successfully processed {len(post_data)} bytes.")
            except Exception as e:
                print(f"Error processing received data: {e}")

            # 發送回應
            self.send_response(200)
            self.send_header('Content-Type', 'application/octet-stream')
            self.end_headers()

            # 回傳一個簡單的確認訊息
            response_data = pickle.dumps(['Data received successfully!'])
            self.wfile.write(response_data)

        except Exception as e:
            print(f"An error occurred: {e}")
            self.send_response(500)
            self.end_headers()


def run_server(server_class=HTTPServer, handler_class=S, server_ip=SERVER_HOST, port=SERVER_PORT):
    server_address = (server_ip, port)
    # 使用 BaseHTTPRequestHandler 和 HTTP 伺服器
    httpd = http.server.HTTPServer(server_address, handler_class)

    print(f"Starting HTTP server on {server_ip}:{port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer shutting down.")
        httpd.socket.close()


if __name__ == '__main__':
    run_server()