import time
import http.server
import ssl
import sys
import os
import lz4.frame
import pickle
from http.server import BaseHTTPRequestHandler, HTTPServer

# 設定伺服器位址和連接埠
SERVER_HOST = '0.0.0.0'
SERVER_PORT = 443  # 使用標準的 HTTPS 埠

# 設定 SSL 憑證檔案路徑
# 請將這裡的路徑替換成您自己的憑證檔案
# 這是為了讓伺服器能啟用 HTTPS
CERT_FILE = 'server.pem'


class S(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            # 取得請求的內容長度
            content_length = int(self.headers.get('Content-Length', 0))

            # 讀取請求體
            post_data = self.rfile.read(content_length)

            # 嘗試解壓縮和反序列化
            try:
                decompress_data = lz4.frame.decompress(post_data)
                _ = pickle.loads(decompress_data)
                print(f"Successfully processed {content_length} bytes.")
            except Exception as e:
                print(f"Error processing received data: {e}")

            # 發送回應
            self.send_response(200)
            self.send_header('Content-Type', 'application/octet-stream')
            self.end_headers()

            # 回傳一個簡單的回應
            response_data = pickle.dumps('Data received successfully!')
            self.wfile.write(response_data)

        except Exception as e:
            print(f"An error occurred: {e}")
            self.send_response(500)
            self.end_headers()


def run_server(server_class=HTTPServer, handler_class=S, server_ip=SERVER_HOST, port=SERVER_PORT):
    # 檢查 SSL 憑證檔案是否存在
    if not os.path.exists(CERT_FILE):
        print(f"Error: SSL certificate file '{CERT_FILE}' not found.")
        print("Please create one using OpenSSL, for example:")
        print("openssl req -x509 -newkey rsa:4096 -keyout server.pem -out server.pem -sha256 -days 365 -nodes")
        sys.exit(1)

    server_address = (server_ip, port)
    httpd = server_class(server_address, handler_class)

    # 將伺服器包裝在 SSL 層中
    httpd.socket = ssl.wrap_socket(
        httpd.socket,
        server_side=True,
        certfile=CERT_FILE,
        ssl_version=ssl.PROTOCOL_TLS
    )

    print(f"Starting HTTPS server on {server_ip}:{port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer shutting down.")
        httpd.socket.close()


if __name__ == '__main__':
    run_server()