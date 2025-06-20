import time
import random


class ServerClientDataCollector:
    """
    用於收集客戶端和邊緣伺服器資訊，並進行合併的類別。
    """

    def __init__(self):
        """
        初始化資料收集器。
        """
        self.client_records = []
        self.edge_server_records = []
        self.merged_data = []
        self.unmatched_clients = []
        self.unmatched_edge_servers = []

    def get_client_info(self):
        """
        模擬收集客戶端資訊並添加到 client_records 列表中。
        """
        print("Collecting client information...")
        client_start_index = random.randint(0, 100)
        client_end_index = random.randint(client_start_index, client_start_index + 50)
        client_buffer_index = random.randint(0, 10)

        # 模擬計算時間
        computation_start_time = time.perf_counter()
        time.sleep(random.uniform(0.01, 0.1))  # 模擬一些計算延遲
        client_computation_time = (time.perf_counter() - computation_start_time) * 1000  # 轉換為毫秒

        is_early_exit = random.choice([True, False])
        early_exit_index = client_end_index if is_early_exit else -1  # 如果沒有提前退出，則設置為-1

        # 模擬客戶端到邊緣伺服器的傳輸時間
        client_to_edge_server_computation_time = random.uniform(5, 50)  # 毫秒

        client_info = {
            "client_start_index": client_start_index,
            "client_end_index": client_end_index,
            "client_buffer_index": client_buffer_index,
            "client_computation_time_ms": round(client_computation_time, 2),
            "early_exit_index": early_exit_index,
            "is_early_exit": is_early_exit,
            "client_to_edge_server_computation_time_ms": round(client_to_edge_server_computation_time, 2),
        }
        self.client_records.append(client_info)
        print("Client information collected and added to records.")
        return client_info

    def get_edge_server_info(self, start_index=None):
        """
        模擬收集邊緣伺服器資訊並添加到 edge_server_records 列表中。
        如果提供了 start_index，則 edge_server_start_index 會從該值開始。
        """
        print("Collecting edge server information...")
        if start_index is not None:
            edge_server_start_index = start_index
        else:
            edge_server_start_index = random.randint(0, 100)

        edge_server_end_index = random.randint(edge_server_start_index, edge_server_start_index + 50)

        # 模擬計算時間
        computation_start_time = time