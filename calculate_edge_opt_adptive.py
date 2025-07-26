import collections
import math
from datetime import datetime, timedelta

import utils
from exit_weight_manager import ExitWeightManager

class PerformanceDataStore:
    def __init__(self, shock_manager, global_initial_estimator, logger, max_records_per_type=0, statisitc_period=10):
        """
        Initializes the CommunicationDataStore.

        Args:
            max_records_per_type (int): The maximum number of records to store
                                        for each unique (start_index, end_index) pair.
        """
        # Main storage for complete records (client + server data), keyed by (start_index, end_index)
        self.data_storage = {
            "client_to_server": collections.defaultdict(collections.deque),
            "edge_to_server": collections.defaultdict(collections.deque)
        }
        self.max_records_per_type = max_records_per_type

        # Temporary storage for pending client data.
        # Key: client_end_index (since this is what we have)
        # Value: List of client records with that end_index, as multiple client events might share an end_index
        self.pending_client_data = collections.defaultdict(collections.deque)

        self.pending_edge_server_data = collections.defaultdict(collections.deque)  # Key: edge_server_end_index


        # Temporary storage for pending server data.
        # Key: server_start_index (since this is what we have)
        # Value: List of server records with that start_index
        self.pending_server_data = collections.defaultdict(collections.deque)
        self.optimal_latency_history = math.inf


        self._start_idx = 0
        self._end_idx = 0
        self._end_idx_buff = 0
        self._max_end_idx = 0
        self._max_layer_amount = 0
        self._statisitc_period = statisitc_period
        self._outgoing_count = 0
        self._incoming_count = 0
        self._new_record_count = 0
        self._steady_state = False
        self.shock_manager = shock_manager
        self.global_initial_estimator = global_initial_estimator

        self.optimal_latency_map = {}
        self.logger = logger


    @property
    def start_idx(self):
        return self._start_idx

    @start_idx.setter
    def start_idx(self, value):
        self._start_idx = value

    @property
    def end_idx(self):
        return self._end_idx

    @end_idx.setter
    def end_idx(self, value):
        self._end_idx = value

    @property
    def end_idx_buff(self):
        return self._end_idx_buff

    @end_idx_buff.setter
    def end_idx_buff(self, value):
        self._end_idx_buff = value

    @property
    def max_end_idx(self):
        return self._max_end_idx

    @max_end_idx.setter
    def max_end_idx(self, end_idx):
        self._max_end_idx = max(self._max_end_idx, end_idx)

    @property
    def max_layer_amount(self):
        return self._max_layer_amount

    @max_layer_amount.setter
    def max_layer_amount(self, value):
        self._max_layer_amount = max(self._max_layer_amount, value)

    @property
    def statistic_period(self):
        return self._statisitc_period

    @statistic_period.setter
    def statistic_period(self, value):
        self._statisitc_period = value

    @property
    def outgoing_count(self):
        return self._outgoing_count

    @outgoing_count.setter
    def outgoing_count(self, value):
        self._outgoing_count = value

    @property
    def incoming_count(self):
        return self._incoming_count

    @incoming_count.setter
    def incoming_count(self, value):
        self._incoming_count = value

    @property
    def new_record_count(self):
        return self._new_record_count

    @new_record_count.setter
    def new_record_count(self, value):
        self._new_record_count = value

    @property
    def steady_state(self):
        return self._steady_state

    @steady_state.setter
    def steady_state(self, value):
        self._steady_state = value

    def _add_complete_record_to_main_storage(self, record_type, complete_record):
        """
        Helper to manage adding a complete record to the main data_storage,
        respecting the max_records_per_type limit for its key.

        Args:
            record_type (str): "client_to_server" or "edge_to_server".
            complete_record (dict): The combined record.
        """
        if record_type == "client_to_server":
            key = (complete_record["server_start_index"], complete_record["server_end_index"])
        elif record_type == "edge_to_server":
            key = (complete_record["edge_server_start_index"], complete_record["edge_server_end_index"],
                   complete_record["server_start_index"])
        else:
            raise ValueError(f"Unknown record_type: {record_type}")

        '''target_deque = self.data_storage[record_type][key]
        if len(target_deque) >= self.max_records_per_type:
            target_deque.popleft()  # Remove the oldest if max size reached
        target_deque.append(complete_record)'''

        if len(self.data_storage[record_type][key]) < self.max_records_per_type + self.statistic_period:
            self.data_storage[record_type][key].append(complete_record)
        else:
            self.data_storage[record_type][key].popleft()  # Remove the oldest
            self.data_storage[record_type][key].append(complete_record)

        if not complete_record['is_early_exit']:
            self.new_record_count = self.new_record_count + 1
            self.shock_manager.record_latency_and_check_shock(complete_record["edge_server_start_index"],
                                                              complete_record["edge_server_end_index"] - complete_record["edge_server_start_index"] + 1,
                                                                complete_record["edge_server_computation_time"],
                                                                complete_record["communication_time_edge_to_server"],
                                                                complete_record["server_computation_time"])
            self.global_initial_estimator.add_sample(complete_record["edge_server_start_index"],
                                                     complete_record["edge_server_end_index"] - complete_record["edge_server_start_index"] + 1,
                                                     complete_record["edge_server_computation_time"],
                                                     complete_record["communication_time_edge_to_server"],
                                                     complete_record["server_computation_time"])

        # print(f"[{datetime.now()}] Combined record added to main storage for type '{record_type}' with key {key}. Current count: {len(target_deque)}")

    def add_client_info(self, timestamp, client_end_index, end_index_buffer, client_computation_time, early_exit_index,
                        is_early_exit):
        """
        Stores client-side information and attempts to form complete records
        by matching with pending server data based on the linking rule
        (server_start_idx == client_end_index + 1) and oldest timestamp preference.

        Args:
            timestamp (datetime): The timestamp when the data was recorded on the client.
            client_end_index (int): The end index collected by the client.
            end_index_buffer (any): The end index buffer data from client.
            client_computation_time (float): The client's computation time.
        """
        client_data = {
            "timestamp": timestamp,
            "client_end_index": client_end_index,
            "end_index_buffer": end_index_buffer,
            "client_computation_time": client_computation_time,
            "early_exit_index": early_exit_index,
            "is_early_exit": is_early_exit,
        }
        # print(f"[{datetime.now()}] Client: Received data (client_end_idx={client_end_index}, timestamp={timestamp}). Checking for server match...")

        server_match_start_index_key = client_end_index + 1

        matched_server_record = None
        if not is_early_exit and server_match_start_index_key in self.pending_server_data and self.pending_server_data[
            server_match_start_index_key]:
            oldest_timestamp = datetime.max
            oldest_server_rec_to_match = None

            # Find the oldest matching server record
            for server_rec in self.pending_server_data[server_match_start_index_key]:
                if server_rec["timestamp"] < oldest_timestamp:
                    oldest_timestamp = server_rec["timestamp"]
                    oldest_server_rec_to_match = server_rec

            if oldest_server_rec_to_match:
                matched_server_record = oldest_server_rec_to_match
                # Now remove the specific matched record from the deque
                # Create a new deque containing all elements EXCEPT the matched_server_record
                new_deque = collections.deque(
                    [rec for rec in self.pending_server_data[server_match_start_index_key]
                     if rec is not matched_server_record]  # Use 'is' for identity comparison
                )
                self.pending_server_data[server_match_start_index_key] = new_deque

        if is_early_exit:
            complete_record = {
                "timestamp": client_data["timestamp"],
                "start_index": client_data["client_end_index"] + 1,
                "end_index": 34,
                "client_end_index": client_data["client_end_index"],
                "end_index_buffer": client_data["end_index_buffer"],
                "is_early_exit": is_early_exit,
                "early_exit_index": early_exit_index,
                "client_computation_time": client_data["client_computation_time"],
                "server_computation_time": 0,
                "communication_time": 0,
            }

            self._add_complete_record_to_main_storage(complete_record)
        elif matched_server_record:
            complete_record = {
                "timestamp": client_data["timestamp"],
                "start_index": matched_server_record["start_index"],
                "end_index": matched_server_record["end_index"],
                "client_end_index": client_data["client_end_index"],
                "end_index_buffer": client_data["end_index_buffer"],
                "is_early_exit": is_early_exit,
                "early_exit_index": early_exit_index,
                "client_computation_time": client_data["client_computation_time"],
                "server_computation_time": matched_server_record["server_computation_time"],
                "communication_time": matched_server_record["communication_time"],
            }
            self._add_complete_record_to_main_storage(complete_record)

        else:
            self.pending_client_data[client_end_index].append(client_data)
            # print(f"[{datetime.now()}] Client: No server match for client_end_idx={client_end_index}. Stored as pending.")

    def add_edge_server_info(self, timestamp, edge_server_start_index, edge_server_end_index, edge_server_buffer_index,
                             edge_server_computation_time, early_exit_index, is_early_exit):
        """
        Stores edge server-side information and attempts to form complete records
        by matching with pending server data based on the linking rule
        (server_start_idx == edge_server_end_index + 1) and oldest timestamp preference.

        Args:
            timestamp (datetime): The timestamp when the data was recorded on the edge server.
            edge_server_start_index (int): The starting index from edge server.
            edge_server_end_index (int): The ending index from edge server.
            edge_server_buffer_index (any): The buffer index data from edge server.
            edge_server_computation_time (float): The edge server's computation time.
            communication_time_edge_to_server (float): Communication time between edge server and central server.
        """
        edge_data = {
            "timestamp": timestamp,
            "edge_server_start_index": edge_server_start_index,
            "edge_server_end_index": edge_server_end_index,
            "edge_server_buffer_index": edge_server_buffer_index,
            "edge_server_computation_time": edge_server_computation_time,
            "early_exit_index": early_exit_index,
            "is_early_exit": is_early_exit,
        }
        # print(f"[{datetime.now()}] Edge Server: Received data (ES_start={edge_server_start_index}, ES_end={edge_server_end_index}, timestamp={timestamp}). Checking for central server match...")

        server_match_start_index_key = edge_server_end_index + 1
        matched_server_record = None
        if not is_early_exit and server_match_start_index_key in self.pending_server_data and self.pending_server_data[server_match_start_index_key]:
            oldest_timestamp = datetime.max
            oldest_server_rec_to_match = None

            # Find the oldest matching server record
            for server_rec in self.pending_server_data[server_match_start_index_key]:
                if server_rec["timestamp"] < oldest_timestamp:
                    oldest_timestamp = server_rec["timestamp"]
                    oldest_server_rec_to_match = server_rec

            if oldest_server_rec_to_match:
                matched_server_record = oldest_server_rec_to_match
                # Now remove the specific matched record from the deque
                # Create a new deque containing all elements EXCEPT the matched_server_record
                new_deque = collections.deque(
                    [rec for rec in self.pending_server_data[server_match_start_index_key]
                     if rec is not matched_server_record]  # Use 'is' for identity comparison
                )
                self.pending_server_data[server_match_start_index_key] = new_deque

            if is_early_exit:
                complete_record = {
                    "timestamp": edge_data["timestamp"],
                    "edge_server_start_index": edge_data["edge_server_start_index"],
                    "edge_server_end_index": edge_data["edge_server_end_index"],
                    "edge_server_buffer_index": edge_data["edge_server_buffer_index"],
                    "is_early_exit": is_early_exit,
                    "early_exit_index": early_exit_index,
                    "edge_server_computation_time": edge_data["edge_server_computation_time"],
                    "server_start_index": edge_data["edge_server_end_index"] + 1,
                    "server_end_index": 34,
                    "server_computation_time": 0,
                    "communication_time_edge_to_server": 0,
                }

                self._add_complete_record_to_main_storage("edge_to_server", complete_record)
            elif matched_server_record:
                complete_record = {
                    "timestamp": edge_data["timestamp"],
                    "edge_server_start_index": edge_data["edge_server_start_index"],
                    "edge_server_end_index": edge_data["edge_server_end_index"],
                    "edge_server_buffer_index": edge_data["edge_server_buffer_index"],
                    "is_early_exit": is_early_exit,
                    "early_exit_index": early_exit_index,
                    "edge_server_computation_time": edge_data["edge_server_computation_time"],
                    "server_start_index": matched_server_record["server_start_index"],
                    "server_end_index": matched_server_record["server_end_index"],
                    "server_computation_time": matched_server_record["server_computation_time"],
                    "communication_time_edge_to_server": matched_server_record["communication_time_to_server"],
                }

                self._add_complete_record_to_main_storage("edge_to_server", complete_record)
            else:
                # Server data will try to find edge data when it arrives, so just store edge data for now.
                # The add_server_info function will handle the matching.
                self.pending_edge_server_data[edge_server_end_index].append(edge_data)
                # print(f"[{datetime.now()}] Edge Server: Stored as pending for edge_server_end_index={edge_server_end_index}.")

    def add_server_info(self, timestamp, server_start_index, server_end_index, server_computation_time,
                        communication_time_to_server):
        """
        Stores central server-side information and attempts to form complete records
        by matching with pending edge server data (priority) or client data.

        Args:
            timestamp (datetime): The timestamp when the data was recorded on the server.
            server_start_index (int): The starting index from server.
            server_end_index (int): The ending index from server.
            server_computation_time (float): The server's computation time.
            communication_time_to_server (float): The communication time to the server (from client or edge).
                                                   This parameter is generic now.
        """
        server_data = {
            "timestamp": timestamp,
            "server_start_index": server_start_index,
            "server_end_index": server_end_index,
            "server_computation_time": server_computation_time,
            "communication_time_to_server": communication_time_to_server,  # Generic name
        }
        # print(f"[{datetime.now()}] Server: Received data (S_start={server_start_index}, S_end={server_end_index}, timestamp={timestamp}). Checking for match...")

        # --- Priority 1: Try to match with pending Edge Server data ---
        edge_match_key = server_start_index - 1  # This is the edge_server_end_index we're looking for

        matched_edge_record = None
        if edge_match_key in self.pending_edge_server_data and self.pending_edge_server_data[edge_match_key]:
            oldest_timestamp = datetime.max
            oldest_edge_rec_to_match = None

            for edge_rec in self.pending_edge_server_data[edge_match_key]:
                if edge_rec["timestamp"] < oldest_timestamp:
                    oldest_timestamp = edge_rec["timestamp"]
                    oldest_edge_rec_to_match = edge_rec

            if oldest_edge_rec_to_match:
                matched_edge_record = oldest_edge_rec_to_match
                # Remove the specific matched record
                new_deque = collections.deque(
                    [rec for rec in self.pending_edge_server_data[edge_match_key]
                     if rec is not matched_edge_record]
                )
                self.pending_edge_server_data[edge_match_key] = new_deque

        if matched_edge_record:
            complete_record = {
                "timestamp": matched_edge_record["timestamp"],  # Use edge's timestamp as event start
                "edge_server_start_index": matched_edge_record["edge_server_start_index"],
                "edge_server_end_index": matched_edge_record["edge_server_end_index"],
                "edge_server_buffer_index": matched_edge_record["edge_server_buffer_index"],
                "is_early_exit": matched_edge_record["is_early_exit"],
                "early_exit_index": matched_edge_record["early_exit_index"],
                "edge_server_computation_time": matched_edge_record["edge_server_computation_time"],
                "server_start_index": server_data["server_start_index"],
                "server_end_index": server_data["server_end_index"],
                "server_computation_time": server_data["server_computation_time"],
                "communication_time_edge_to_server": server_data["communication_time_to_server"],
                # Note: communication_time_to_server from add_server_info is NOT used here as edge provides its own comm time.
                # If this comm_time_to_server was general network latency, it would be added.
                # Assuming communication_time_edge_to_server is the one relevant for this path.
            }
            self._add_complete_record_to_main_storage("edge_to_server", complete_record)
            return  # Match found and processed, exit

        '''# --- Priority 2: If no Edge Server match, try to match with pending Client data ---
        client_match_key = server_start_index - 1  # This is the client_end_index we're looking for

        matched_client_record = None
        if client_match_key in self.pending_client_data and self.pending_client_data[client_match_key]:
            oldest_timestamp = datetime.max
            oldest_client_rec_to_match = None

            for client_rec in self.pending_client_data[client_match_key]:
                if client_rec["timestamp"] < oldest_timestamp:
                    oldest_timestamp = client_rec["timestamp"]
                    oldest_client_rec_to_match = client_rec

            if oldest_client_rec_to_match:
                matched_client_record = oldest_client_rec_to_match
                # Remove the specific matched record
                new_deque = collections.deque(
                    [rec for rec in self.pending_client_data[client_match_key]
                     if rec is not matched_client_record]
                )
                self.pending_client_data[client_match_key] = new_deque

        if matched_client_record:
            complete_record = {
                "timestamp": matched_client_record["timestamp"],  # Use client's timestamp as event start
                "server_start_index": server_data["server_start_index"],
                "server_end_index": server_data["server_end_index"],
                "client_end_index": matched_client_record["client_end_index"],
                "end_index_buffer": matched_client_record["end_index_buffer"],
                "client_computation_time": matched_client_record["client_computation_time"],
                "server_computation_time": server_data["server_computation_time"],
                "communication_time_client_to_server": server_data["communication_time_to_server"],
                # Use the generic comm time here
            }
            self._add_complete_record_to_main_storage("client_to_server", complete_record)
            return  # Match found and processed, exit'''

        # --- If no match (neither edge nor client), store server data in pending ---
        self.pending_server_data[server_start_index].append(server_data)
        # print(f"[{datetime.now()}] Server: No edge/client match for server_start_idx={server_start_index}. Stored as pending.")

    def get_data(self, record_type, start_index_tuple):
        """
        Retrieves all stored complete data for a specific record_type and its key.
        Args:
            record_type (str): "client_to_server" or "edge_to_server".
            start_index_tuple (tuple): Key for the specific record set.
                                       For client_to_server: (server_start_index, server_end_index)
                                       For edge_to_server: (edge_server_start_index, edge_server_end_index, server_start_index)
        """
        return list(self.data_storage.get(record_type, {}).get(start_index_tuple, collections.deque()))

    def get_all_data_by_type(self, record_type):
        """
        Retrieves all complete data for a specific record_type, grouped by its key.
        """
        return {k: list(v) for k, v in self.data_storage.get(record_type, {}).items()}

    def get_all_data_by_edge_server_start_index(self, edge_server_start_idx: int):
        """
        Retrieves all complete 'edge_to_server' records that have the specified
        edge_server_start_index.

        Args:
            edge_server_start_idx (int): The specific edge server start index to filter by.

        Returns:
            dict: A dictionary where keys are the full (edge_server_start_index, edge_server_end_index, server_start_index)
                  tuples and values are lists of records, filtered by the input edge_server_start_idx.
        """
        filtered_data = {}
        all_edge_to_server_data = self.get_all_data_by_type("edge_to_server")

        for key_tuple, records_deque in all_edge_to_server_data.items():
            current_es_start_idx = key_tuple[0]  # The first element of the key is edge_server_start_index
            if current_es_start_idx == edge_server_start_idx:
                filtered_data[key_tuple] = list(records_deque)  # Convert deque to list for return

        return filtered_data

    # --- Helper methods for getting counts of pending/stored data ---
    def get_num_pending_client_items(self):
        return sum(len(deque) for deque in self.pending_client_data.values())

    def get_num_pending_edge_server_items(self):
        return sum(len(deque) for deque in self.pending_edge_server_data.values())

    def get_num_pending_server_items(self):
        return sum(len(deque) for deque in self.pending_server_data.values())

    def get_total_stored_records(self):
        total_records = 0
        for record_type_dict in self.data_storage.values():
            for deque_of_records in record_type_dict.values():
                total_records += len(deque_of_records)
        return total_records

    def get_pending_data_status(self):
        status = {
            "pending_client": {k: len(v) for k, v in self.pending_client_data.items()},
            "pending_edge_server": {k: len(v) for k, v in self.pending_edge_server_data.items()},
            "pending_server": {k: len(v) for k, v in self.pending_server_data.items()},
            "total_pending_client": self.get_num_pending_client_items(),
            "total_pending_edge_server": self.get_num_pending_edge_server_items(),
            "total_pending_server": self.get_num_pending_server_items(),
        }
        return status

    def get_optimal_end_idx(self, start_idx):
        """
        回傳該 start_idx 對應的最佳 end_idx 與 latency。
        若該 start_idx 尚未記錄，則回傳 None。
        """
        return self.optimal_latency_map.get(start_idx, None)

    def set_optimal_set(self, start_idx, end_idx, latency):
        """
        若該 start_idx 尚未記錄，或 latency 優於目前記錄的值，則更新。
        否則不更新。
        """
        current = self.optimal_latency_map.get(start_idx)
        if current is None or latency < current[1]:
            self.optimal_latency_map[start_idx] = (end_idx, latency)
            return True  # 表示有更新
        return False  # 表示未更新


def calculate_edge_server_opt(data_store: PerformanceDataStore, ppl, lm_manager, mode, shock_manager, logger,
                              edge_server_start_idx: int):
    """
    For a given `edge_server_start_idx`, finds the `edge_server_end_index` that results
    in the minimal weighted average latency for the Edge-to-Server segment.
    The latency is the sum of edge server computation time, server computation time,
    and communication time between edge server and server. Before weighting, the single
    data row (record) with the highest total latency within each path segment
    is removed. The 'k_oldest_weighted' records get a smaller weight (0.3),
    and newer records get a larger weight (0.7).

    Args:
        data_store (CommunicationDataStore): An instance of the CommunicationDataStore.
        edge_server_start_idx (int): The specific edge server start index to analyze.
        k_oldest_weighted (int): The number of oldest records in each path segment
                                 to apply the smaller weight to.

    Returns:
        tuple or None: A tuple (
            edge_server_start_idx,
            optimal_edge_server_end_index,
            minimal_total_weighted_latency_for_that_end_index,
            is_converging: bool,
            latency_diff: float or None
        ) if valid data is available for the specified start index, otherwise None.
    """
    exit_rate_manager = ExitWeightManager(mode=mode)

    start_idx_list = []
    for key_tuple, records_list in data_store.get_all_data_by_type('edge_to_server').items():
        logger.log(f'(key_tuple, record_list):({key_tuple}, {records_list})')
        temp_start_idx, _, _ = key_tuple
        start_idx_list.append(temp_start_idx)

    for temp_edge_server_start_idx in start_idx_list:
        relevant_data_by_full_key = data_store.get_all_data_by_edge_server_start_index(temp_edge_server_start_idx)
        if not relevant_data_by_full_key:
            return None

        # This will now store (edge_server_start_index, edge_server_end_index) -> list of (total_latency, timestamp)
        path_latencies_with_timestamps = collections.defaultdict(list)

        WEIGHT_OLD = 0
        WEIGHT_NEW = 1

        for key_tuple, records_list in relevant_data_by_full_key.items():
            current_es_start_idx, current_es_end_idx, server_start_idx = key_tuple
            logger.log(f'record (start, end): ({current_es_start_idx}, {current_es_end_idx})')

            for record in records_list:
                if (record.get("edge_server_computation_time") is not None and
                        record.get("server_computation_time") is not None and
                        record.get("communication_time_edge_to_server") is not None and
                        not record.get("is_early_exit")):
                    segment_latency = (record["edge_server_computation_time"] +
                                   record["server_computation_time"] +
                                   record["communication_time_edge_to_server"])

                    path_latencies_with_timestamps[(current_es_start_idx, current_es_end_idx)].append(
                        (segment_latency, record["timestamp"], record["edge_server_computation_time"],
                        record["server_computation_time"], record["communication_time_edge_to_server"], False))
                elif (record.get("edge_server_computation_time") is not None and
                    record.get("server_computation_time") is not None and
                    record.get("communication_time_edge_to_server") is not None and
                    record.get("is_early_exit")):
                    segment_latency = (record["edge_server_computation_time"] +
                                   record["server_computation_time"] +
                                   record["communication_time_edge_to_server"])

                    path_latencies_with_timestamps[(current_es_start_idx, current_es_end_idx)].append(
                        (segment_latency, record["timestamp"], record["edge_server_computation_time"],
                        record["server_computation_time"], record["communication_time_edge_to_server"], True))

        if not path_latencies_with_timestamps:
            return None

        min_weighted_avg_latency = float('inf')
        optimal_es_end_idx = None

        # Iterate through each unique path (edge_server_start_index, edge_server_end_index)
        for (es_start, es_end), latencies_with_timestamps in path_latencies_with_timestamps.items():
            if not latencies_with_timestamps:
                continue

            print('(start, end): ', (es_start, es_end))
            # Sort by timestamp to ensure oldest are truly first for weighting
            latencies_with_timestamps.sort(key=lambda x: x[1])

            # --- NEW LOGIC: Remove the data row with the most latency ---
            if len(latencies_with_timestamps) > 2:  # Need at least two records to remove one and still have data
                # max_latency_entry = max(latencies_with_timestamps, key=lambda x: x[0])
                # latencies_with_timestamps.remove(max_latency_entry)
                for flag in [True, False]:
                    # Get candidates that match the flag
                    flagged_entries = [entry for entry in latencies_with_timestamps if entry[5] == flag]
                    if len(flagged_entries) > 0:
                        max_entry = max(flagged_entries, key=lambda x: x[0])  # Find max segment_latency
                        latencies_with_timestamps.remove(max_entry)
                # print(f"DEBUG ES_OPT: For path {(es_start, es_end)}, removed record with latency {max_latency_entry[0]:.4f}")
            # --- END NEW LOGIC ---

            weighted_sum_for_path = 0.0
            total_weight_for_path = 0.0
            head_name, _ = utils.get_lm_head_idx(es_end)
            early_rate = lm_manager.predict_exit_rate(head_name, ppl)
            logger.log(f'exit rate: {early_rate}')

            latency_edge_server = 0.0
            latency_comm = 0.0
            latency_server = 0.0
            client_count = 0.0
            comm_count = 0.0
            server_count = 0.0
            for i, (latency, _, edge_comp_time, server_comp_time, comm_time, is_early) in enumerate(
                    latencies_with_timestamps):
                print('record(latency, timestamp, edge_comp_time, server_comp_time, comm_time, is_early): ',
                    (latency, _, edge_comp_time, server_comp_time, comm_time, is_early))
                if not is_early:
                    latency_edge_server += edge_comp_time
                    latency_comm += comm_time
                    latency_server += server_comp_time
                    client_count = client_count + 1
                    comm_count = comm_count + 1
                    server_count = server_count + 1
                else:
                    latency_edge_server += edge_comp_time
                    client_count = client_count + 1

            # shock_manager.set_avg_latency(es_end - es_start + 1, latency_edge_server / (client_count + 1e-6),
            #                              latency_comm / (comm_count + 1e-6), latency_server / (server_count + 1e-6))
            WEIGHT_EARLY = exit_rate_manager.compute_weight(logger, early_rate, latency_edge_server / (client_count + 1e-6),
                                                        latency_comm / (comm_count + 1e-6),
                                                        latency_server / (server_count + 1e-6))
            logger.log(f'early weight: {WEIGHT_EARLY}')

            # Apply weighted average
            for i, (latency, _, edge_comp_time, server_comp_time, comm_time, is_early) in enumerate(
                    latencies_with_timestamps):
                if i < data_store.max_records_per_type:
                    if is_early:
                        weighted_sum_for_path += latency * WEIGHT_OLD * WEIGHT_EARLY
                        total_weight_for_path += WEIGHT_OLD
                    else:
                        weighted_sum_for_path += latency * WEIGHT_OLD
                        total_weight_for_path += WEIGHT_OLD
                else:
                    if is_early:
                        weighted_sum_for_path += latency * WEIGHT_NEW * WEIGHT_EARLY
                        total_weight_for_path += WEIGHT_NEW * WEIGHT_EARLY
                    else:
                        weighted_sum_for_path += latency * WEIGHT_NEW
                        total_weight_for_path += WEIGHT_NEW

            current_weighted_avg_latency = 0.0
            if total_weight_for_path > 0:
                current_weighted_avg_latency = weighted_sum_for_path / total_weight_for_path
            else:  # No valid records or weights applied after removal
                continue

            data_store.set_optimal_set(es_start, es_end, current_weighted_avg_latency)

        '''# Find the path with the minimum weighted average latency
        if current_weighted_avg_latency < min_weighted_avg_latency:
            min_weighted_avg_latency = current_weighted_avg_latency
            optimal_es_end_idx = es_end'''

    '''if optimal_es_end_idx is None:
        return None'''

    logger.log(f'opt map: {data_store.get_optimal_end_idx(edge_server_start_idx)}')
    result = data_store.get_optimal_end_idx(edge_server_start_idx)
    if result:
        optimal_es_end_idx, min_weighted_avg_latency = result
    else:
        return None, None, data_store._statisitc_period

    if data_store.optimal_latency_history * 1.1 < min_weighted_avg_latency:
        data_store._statisitc_period = max(10, math.floor(data_store._statisitc_period * 2 / 3))
        # self._statisitc_period = max(10, self._statisitc_period - 4)
    elif data_store.optimal_latency_history * 1.1 > min_weighted_avg_latency:
        data_store._statisitc_period = min(100, data_store._statisitc_period + 6)

    data_store.optimal_latency_history = min_weighted_avg_latency

    if data_store._statisitc_period > 20:
        data_store._steady_state = True
        data_store.data_storage.clear()
        data_store.data_storage = {
            "client_to_server": collections.defaultdict(collections.deque),
            "edge_to_server": collections.defaultdict(collections.deque)
        }

    data_store._new_record_count = 0
    data_store.max_records_per_type = 0
    shock_manager.reset_history()

    logger.log(f'optimal map: {data_store.optimal_latency_map}')

    return optimal_es_end_idx, optimal_es_end_idx + 2, data_store._statisitc_period


def calculate_edge_server_opt3(data_store: PerformanceDataStore, ppl, lm_manager, mode, shock_manager, logger, edge_server_start_idx: int):
    """
    For a given `edge_server_start_idx`, finds the `edge_server_end_index` that results
    in the minimal weighted average latency for the Edge-to-Server segment.
    The latency is the sum of edge server computation time, server computation time,
    and communication time between edge server and server. Before weighting, the single
    data row (record) with the highest total latency within each path segment
    is removed. The 'k_oldest_weighted' records get a smaller weight (0.3),
    and newer records get a larger weight (0.7).

    Args:
        data_store (CommunicationDataStore): An instance of the CommunicationDataStore.
        edge_server_start_idx (int): The specific edge server start index to analyze.
        k_oldest_weighted (int): The number of oldest records in each path segment
                                 to apply the smaller weight to.

    Returns:
        tuple or None: A tuple (
            edge_server_start_idx,
            optimal_edge_server_end_index,
            minimal_total_weighted_latency_for_that_end_index,
            is_converging: bool,
            latency_diff: float or None
        ) if valid data is available for the specified start index, otherwise None.
    """
    exit_rate_manager = ExitWeightManager(mode=mode)

    start_idx_list = []
    for key_tuple, records_list in data_store.get_all_data_by_type('edge_to_server').items():
        logger.log(f'(key_tuple, record_list):({key_tuple}, {records_list})')
        temp_start_idx, _, _ = key_tuple
        start_idx_list.append(temp_start_idx)




    relevant_data_by_full_key = data_store.get_all_data_by_edge_server_start_index(edge_server_start_idx)
    if not relevant_data_by_full_key:
        return None

    # This will now store (edge_server_start_index, edge_server_end_index) -> list of (total_latency, timestamp)
    path_latencies_with_timestamps = collections.defaultdict(list)

    WEIGHT_OLD = 0
    WEIGHT_NEW = 1



    for key_tuple, records_list in relevant_data_by_full_key.items():
        current_es_start_idx, current_es_end_idx, server_start_idx = key_tuple
        logger.log(f'record (start, end): ({current_es_start_idx}, {current_es_end_idx})')

        for record in records_list:
            if (record.get("edge_server_computation_time") is not None and
                record.get("server_computation_time") is not None and
                record.get("communication_time_edge_to_server") is not None and
                not record.get("is_early_exit")):
                segment_latency = (record["edge_server_computation_time"] +
                                   record["server_computation_time"] +
                                   record["communication_time_edge_to_server"])

                path_latencies_with_timestamps[(current_es_start_idx, current_es_end_idx)].append(
                    (segment_latency, record["timestamp"], record["edge_server_computation_time"], record["server_computation_time"], record["communication_time_edge_to_server"], False))
            elif (record.get("edge_server_computation_time") is not None and
                record.get("server_computation_time") is not None and
                record.get("communication_time_edge_to_server") is not None and
                record.get("is_early_exit")):
                segment_latency = (record["edge_server_computation_time"] +
                                   record["server_computation_time"] +
                                   record["communication_time_edge_to_server"])

                path_latencies_with_timestamps[(current_es_start_idx, current_es_end_idx)].append(
                    (segment_latency, record["timestamp"], record["edge_server_computation_time"], record["server_computation_time"], record["communication_time_edge_to_server"], True))

    if not path_latencies_with_timestamps:
        return None

    min_weighted_avg_latency = float('inf')
    optimal_es_end_idx = None

    # Iterate through each unique path (edge_server_start_index, edge_server_end_index)
    for (es_start, es_end), latencies_with_timestamps in path_latencies_with_timestamps.items():
        if not latencies_with_timestamps:
            continue

        print('(start, end): ', (es_start, es_end))
        # Sort by timestamp to ensure oldest are truly first for weighting
        latencies_with_timestamps.sort(key=lambda x: x[1])

        # --- NEW LOGIC: Remove the data row with the most latency ---
        if len(latencies_with_timestamps) > 2:  # Need at least two records to remove one and still have data
            #max_latency_entry = max(latencies_with_timestamps, key=lambda x: x[0])
            #latencies_with_timestamps.remove(max_latency_entry)
            for flag in [True, False]:
                # Get candidates that match the flag
                flagged_entries = [entry for entry in latencies_with_timestamps if entry[5] == flag]
                if len(flagged_entries) > 0:
                    max_entry = max(flagged_entries, key=lambda x: x[0])  # Find max segment_latency
                    latencies_with_timestamps.remove(max_entry)
            # print(f"DEBUG ES_OPT: For path {(es_start, es_end)}, removed record with latency {max_latency_entry[0]:.4f}")
        # --- END NEW LOGIC ---

        weighted_sum_for_path = 0.0
        total_weight_for_path = 0.0
        head_name, _ = utils.get_lm_head_idx(es_end)
        early_rate = lm_manager.predict_exit_rate(head_name, ppl)
        logger.log(f'exit rate: {early_rate}')

        latency_edge_server = 0.0
        latency_comm = 0.0
        latency_server = 0.0
        client_count = 0.0
        comm_count = 0.0
        server_count = 0.0
        for i, (latency, _, edge_comp_time, server_comp_time, comm_time, is_early) in enumerate(latencies_with_timestamps):
            print('record(latency, timestamp, edge_comp_time, server_comp_time, comm_time, is_early): ', (latency, _, edge_comp_time, server_comp_time, comm_time, is_early))
            if not is_early:
                latency_edge_server += edge_comp_time
                latency_comm += comm_time
                latency_server += server_comp_time
                client_count = client_count + 1
                comm_count = comm_count + 1
                server_count = server_count + 1
            else:
                latency_edge_server += edge_comp_time
                client_count = client_count + 1



        #shock_manager.set_avg_latency(es_end - es_start + 1, latency_edge_server / (client_count + 1e-6),
        #                              latency_comm / (comm_count + 1e-6), latency_server / (server_count + 1e-6))
        WEIGHT_EARLY = exit_rate_manager.compute_weight(logger, early_rate, latency_edge_server / (client_count + 1e-6),
                                                        latency_comm / (comm_count + 1e-6),
                                                        latency_server / (server_count + 1e-6))
        logger.log(f'early weight: {WEIGHT_EARLY}')

        # Apply weighted average
        for i, (latency, _, edge_comp_time, server_comp_time, comm_time, is_early) in enumerate(latencies_with_timestamps):
            if i < data_store.max_records_per_type:
                if is_early:
                    weighted_sum_for_path += latency * WEIGHT_OLD * WEIGHT_EARLY
                    total_weight_for_path += WEIGHT_OLD
                else:
                    weighted_sum_for_path += latency * WEIGHT_OLD
                    total_weight_for_path += WEIGHT_OLD
            else:
                if is_early:
                    weighted_sum_for_path += latency * WEIGHT_NEW * WEIGHT_EARLY
                    total_weight_for_path += WEIGHT_NEW * WEIGHT_EARLY
                else:
                    weighted_sum_for_path += latency * WEIGHT_NEW
                    total_weight_for_path += WEIGHT_NEW

        current_weighted_avg_latency = 0.0
        if total_weight_for_path > 0:
            current_weighted_avg_latency = weighted_sum_for_path / total_weight_for_path
        else:  # No valid records or weights applied after removal
            continue


        data_store.set_optimal_set(es_start, es_end, current_weighted_avg_latency)
        # Find the path with the minimum weighted average latency
        if current_weighted_avg_latency < min_weighted_avg_latency:
            min_weighted_avg_latency = current_weighted_avg_latency
            optimal_es_end_idx = es_end





    if optimal_es_end_idx is None:
        return None

    if data_store.optimal_latency_history * 1.1 < min_weighted_avg_latency:
        data_store._statisitc_period = max(10, math.floor(data_store._statisitc_period * 2 / 3))
        # self._statisitc_period = max(10, self._statisitc_period - 4)
    elif data_store.optimal_latency_history * 1.1 > min_weighted_avg_latency:
        data_store._statisitc_period = min(100, data_store._statisitc_period + 6)


    data_store.optimal_latency_history = min_weighted_avg_latency

    if data_store._statisitc_period > 20:
        data_store._steady_state = True
        data_store.data_storage.clear()
        data_store.data_storage = {
            "client_to_server": collections.defaultdict(collections.deque),
            "edge_to_server": collections.defaultdict(collections.deque)
        }

    data_store._new_record_count = 0
    data_store.max_records_per_type = 0
    shock_manager.reset_history()

    logger.log(f'optimal map: {data_store.optimal_latency_map}')

    return optimal_es_end_idx, optimal_es_end_idx + 2,  data_store._statisitc_period



def calculate_edge_server_opt2(data_store: PerformanceDataStore, edge_server_start_idx: int):
    """
    For a given `edge_server_start_idx`, finds the `edge_server_end_index` that results
    in the minimal average latency for the Edge-to-Server segment.
    The latency is the sum of edge server computation time, server computation time,
    and communication time between edge server and server.

    Args:
        data_store (CommunicationDataStore): An instance of the CommunicationDataStore.
        edge_server_start_idx (int): The specific edge server start index to analyze.

    Returns:
        tuple or None: A tuple (
            edge_server_start_idx,
            optimal_edge_server_end_index,
            minimal_total_latency_for_that_end_index,
            is_converging: bool,
            latency_diff: float or None
        ) if valid data is available for the specified start index, otherwise None.
    """
    all_edge_to_server_data = data_store.get_all_data_by_type("edge_to_server")
    if not all_edge_to_server_data:
        return None

    # Group records by (edge_server_start_index, edge_server_end_index)
    # This will hold lists of individual latency measurements for each specific path
    path_latencies = collections.defaultdict(list)

    for key_tuple, records_deque in all_edge_to_server_data.items():
        current_es_start_idx, current_es_end_idx, server_start_idx = key_tuple

        if current_es_start_idx == edge_server_start_idx:
            for record in records_deque:
                if (record.get("edge_server_computation_time") is not None and
                        record.get("server_computation_time") is not None and
                        record.get("communication_time_edge_to_server") is not None):
                    segment_latency = (record["edge_server_computation_time"] +
                                       record["server_computation_time"] +
                                       record["communication_time_edge_to_server"])
                    # Store latency for the specific (edge_server_start_index, edge_server_end_index) path
                    path_latencies[(current_es_start_idx, current_es_end_idx)].append(segment_latency)

    if not path_latencies:
        return None  # No relevant data for this edge_server_start_idx


    min_avg_latency = float('inf')
    optimal_es_end_idx = None  # This will store the edge_server_end_index that is optimal

    # Calculate average latency for each (edge_server_start_index, edge_server_end_index) path
    # and find the overall minimum
    for (es_start, es_end), latencies in path_latencies.items():
        if latencies:
            current_avg_latency = sum(latencies) / len(latencies)
            if current_avg_latency < min_avg_latency:
                min_avg_latency = current_avg_latency
                optimal_es_end_idx = es_end  # Store the edge_server_end_index

    if optimal_es_end_idx is None:  # No valid average could be calculated
        return None

    if data_store.optimal_latency_history * 1.1 < min_avg_latency:
        data_store._statisitc_period = max(10, math.floor(data_store._statisitc_period * 2 / 3))
        # self._statisitc_period = max(10, self._statisitc_period - 4)
    elif data_store.optimal_latency_history * 1.1 > min_avg_latency:
        data_store._statisitc_period = min(100, data_store._statisitc_period + 6)


    data_store.optimal_latency_history = min_avg_latency

    if data_store._statisitc_period > 20:
        data_store._steady_state = True

    data_store._new_record_count = 0

    return optimal_es_end_idx, optimal_es_end_idx + 2,  data_store._statisitc_period


# Example Usage:
if __name__ == "__main__":
    data_store = PerformanceDataStore(max_records_per_type=2, convergence_history_size=3, convergence_threshold=0.005)

    print("--- Testing 3-device system with Edge Server ---")

    # Scenario 1: Client -> Server path
    print("\n--- Client -> Server Path ---")
    data_store.add_client_info(datetime.now(), 100, "buf_client_100", 0.05)
    data_store.add_server_info(datetime.now() + timedelta(milliseconds=10), 101, 500, 0.1, 0.03) # Server latency 0.05 + 0.1 + 0.03 = 0.18

    # Scenario 2: Edge -> Server path
    print("\n--- Edge -> Server Path (Round 1) ---")
    data_store.add_edge_server_info(datetime.now() + timedelta(milliseconds=20), 1000, 1001, "buf_edge_1001", 0.02, 0.01) # Edge data
    data_store.add_server_info(datetime.now() + timedelta(milliseconds=30), 1002, 600, 0.04, 0.005) # Server data matches edge (1001+1=1002)
    # Edge-Server latency: 0.02 (edge_comp) + 0.04 (server_comp) + 0.01 (edge_to_server_comm) = 0.07

    opt_edge_1 = calculate_opt(data_store, record_type="edge_to_server")
    if opt_edge_1:
        key, latency, conv, diff = opt_edge_1
        print(f"Optimal Edge-Server Latency (1): Key={key}, Latency={latency:.4f}, Converging={conv}, Diff={diff}")
    print(f"Edge Latency History (1): {[f'{l:.4f}' for l in data_store.get_optimal_latency_history()]}")
    print(f"Pending status (1): {data_store.get_pending_data_status()}")
    print(f"Total stored records (1): {data_store.get_total_stored_records()}")


    # Scenario 2 cont.: More Edge -> Server data (Round 2)
    print("\n--- Edge -> Server Path (Round 2) ---")
    data_store.add_edge_server_info(datetime.now() + timedelta(milliseconds=40), 2000, 2001, "buf_edge_2001", 0.015, 0.008) # New edge data
    data_store.add_server_info(datetime.now() + timedelta(milliseconds=50), 2002, 601, 0.03, 0.004) # Server data matches new edge (2001+1=2002)
    # Edge-Server latency: 0.015 + 0.03 + 0.008 = 0.053 (potentially new optimal)

    opt_edge_2 = calculate_opt(data_store, record_type="edge_to_server")
    if opt_edge_2:
        key, latency, conv, diff = opt_edge_2
        print(f"Optimal Edge-Server Latency (2): Key={key}, Latency={latency:.4f}, Converging={conv}, Diff={diff}")
    #print(f"Edge Latency History (2): {[f'{l:.4f}' for l in data_store.get_optimal_latency_history()]}")
    print(f"Pending status (2): {data_store.get_pending_data_status()}")
    print(f"Total stored records (2): {data_store.get_total_stored_records()}")


    # Scenario 2 cont.: More Edge -> Server data (Round 3 - Trigger Convergence Check)
    print("\n--- Edge -> Server Path (Round 3 - Convergence Check) ---")
    data_store.add_edge_server_info(datetime.now() + timedelta(milliseconds=60), 3000, 3001, "buf_edge_3001", 0.016, 0.007)
    data_store.add_server_info(datetime.now() + timedelta(milliseconds=70), 3002, 602, 0.031, 0.005)
    # Edge-Server latency: 0.016 + 0.031 + 0.007 = 0.054 (close to previous 0.053)

    opt_edge_3 = calculate_opt(data_store, record_type="edge_to_server")
    if opt_edge_3:
        key, latency, conv, diff = opt_edge_3
        print(f"Optimal Edge-Server Latency (3): Key={key}, Latency={latency:.4f}, Converging={conv}, Diff={diff}")
    #print(f"Edge Latency History (3): {[f'{l:.4f}' for l in data_store.get_optimal_latency_history()]}")
    print(f"Pending status (3): {data_store.get_pending_data_status()}")
    print(f"Total stored records (3): {data_store.get_total_stored_records()}")


    # Test client-to-server path separately
    print("\n--- Calculating Optimal for Client -> Server Path ---")
    opt_client_server = calculate_opt(data_store, record_type="client_to_server")
    if opt_client_server:
        key, latency, conv, diff = opt_client_server
        print(f"Optimal Client-Server Latency: Key={key}, Latency={latency:.4f}, Converging={conv}, Diff={diff}")
    else:
        print("No complete Client-Server data available.")

    print("\n--- Final Pending Data Status ---")
    print(data_store.get_pending_data_status())

    print("\n--- All Stored Data By Type ---")
    print("Client To Server:")
    print(data_store.get_all_data_by_type("client_to_server"))
    print("\nEdge To Server:")
    print(data_store.get_all_data_by_type("edge_to_server"))