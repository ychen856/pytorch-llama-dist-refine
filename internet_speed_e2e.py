import time
import datetime
import http.client
import csv
import pickle
import lz4.frame
import torch

# 這裡需要替換成你的實際 API 端點
API_ENDPOINT = "https://test-server-service.nrp-nautilus.io/"
FEATURE_VECTOR_FILE_1 = "vector_out.pt"
FEATURE_VECTOR_FILE_2 = "vector_ids.pt"
FEATURE_VECTOR_FILE_3 = "vector_mask.pt"


def read_feature_vectors(file_path):
    """
    從 .pt 檔案中讀取特徵向量。
    """
    try:
        vectors = torch.load(file_path)
        if isinstance(vectors, torch.Tensor):
            return vectors.tolist()
        else:
            print(f"The loaded .pt file '{file_path}' is not a tensor.")
            return []
    except FileNotFoundError:
        print(f"Error: {file_path} not found.")
        return []
    except Exception as e:
        print(f"Error loading .pt file: {e}")
        return []


def measure_latency(api_endpoint, payload):
    """
    模擬發送特徵向量到指定的 API 端點，並返回網路延遲。
    """
    try:
        # 準備要發送的資料
        newx = lz4.frame.compress(pickle.dumps(payload))
        total_size = len(newx)
        print(f'Communication size: {total_size} bytes')

        # 解析 URL 以建立連線
        url_parts = api_endpoint.split('://')
        protocol = url_parts[0]
        host = url_parts[1].split('/')[0]

        if protocol == 'https':
            conn = http.client.HTTPSConnection(host)
        else:
            conn = http.client.HTTPConnection(host)

        # 計時開始：在發送請求之前
        start_time = time.time()

        conn.request('POST', '/', body=newx, headers={
            'Content-Type': 'application/octet-stream',
            'Content-Length': str(total_size)
        })

        resp = conn.getresponse()

        # 計時結束：在收到回應之後
        end_time = time.time()

        # 關閉連線
        conn.close()

        if resp.status == 200:
            latency = (end_time - start_time) * 1000  # 轉換為毫秒
            return latency
        else:
            print(f"Received non-200 status code: {resp.status}")
            return "N/A"
    except Exception as e:
        print(f"Error during request: {e}")
        return "N/A"


def run_test(duration_hours=24, interval_seconds=3600):
    """
    定期運行測試並記錄結果。
    """
    feature_vectors_1 = read_feature_vectors(FEATURE_VECTOR_FILE_1)
    feature_vectors_2 = read_feature_vectors(FEATURE_VECTOR_FILE_2)
    feature_vectors_3 = read_feature_vectors(FEATURE_VECTOR_FILE_3)

    if not feature_vectors_1 or not feature_vectors_2 or not feature_vectors_3:
        print("Required feature vectors not found. Exiting.")
        return

    end_time = time.time() + duration_hours * 3600
    file_name = f"latency_stats_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    with open(file_name, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Latency (ms)"])

        print(f"開始網路延遲測試，結果將儲存到 {file_name}")

        # 取得最小的向量數來確保循環不會出錯
        min_len = min(len(feature_vectors_1), len(feature_vectors_2), len(feature_vectors_3))
        vector_index = 0

        while time.time() < end_time:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"\n[{timestamp}] Measuring latency...")

            # 組合特徵向量為一個元組
            test_vector = (
                feature_vectors_1[vector_index],
                feature_vectors_2[vector_index],
                feature_vectors_3[vector_index]
            )

            latency = measure_latency(API_ENDPOINT, test_vector)

            writer.writerow([timestamp, latency])
            print(f"[{timestamp}] Latency: {latency} ms")

            # 循環使用特徵向量
            vector_index = (vector_index + 1) % min_len

            if time.time() < end_time:
                time.sleep(interval_seconds)


if __name__ == "__main__":
    run_test(duration_hours=24, interval_seconds=3600)