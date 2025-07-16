from collections import deque
import utils

class PredictiveSplittingManager:
    def __init__(self, lm_manager, logger, shock_alpha=2, window_size=10, shock_threshold=7):
        self.alpha = shock_alpha
        self.window_size = window_size
        self.shock_threshold = shock_threshold

        self.avg_client = {}
        self.avg_comm = {}
        self.avg_server = {}

        self.history = deque(maxlen=window_size)
        self.history_k = deque(maxlen=window_size)
        self.history_latency = deque(maxlen=window_size)

        self.last_shock_flags = (False, False, False)
        self.last_obs_latency = (0.0, 0.0, 0.0)
        self.last_k = None

        self.lm_manager = lm_manager
        self.logger = logger

    def reset_history(self):
        self.history.clear()
        self.history_k.clear()
        self.history_latency.clear()

    def reset_avg(self):
        self.avg_client.clear()
        self.avg_comm.clear()
        self.avg_server.clear()

    def set_avg_latency(self, k, client_time, comm_time, server_time):
        self.avg_client[k] = client_time
        self.avg_comm[k] = comm_time
        self.avg_server[k] = server_time

    def is_shock(self, k, obs_client, obs_comm, obs_server):
        try:
            base_client = self.avg_client[k]
            base_comm = self.avg_comm[k]
            base_server = self.avg_server[k]
        except KeyError:
            return False

        shock_c = obs_client > base_client * self.alpha
        shock_m = obs_comm   > base_comm   * self.alpha
        shock_s = obs_server > base_server * self.alpha

        self.last_shock_flags = (shock_c, shock_m, shock_s)
        return shock_c or shock_m or shock_s

    def record_latency(self, k, obs_client, obs_comm, obs_server):
        self.last_k = k
        self.last_obs_latency = (obs_client, obs_comm, obs_server)

        shock = self.is_shock(k, obs_client, obs_comm, obs_server)
        self.history.append(shock)
        self.history_k.append(k)
        self.history_latency.append((obs_client, obs_comm, obs_server))

    def is_trigger_override(self):
        return sum(self.history) >= self.shock_threshold

    def decide_k(self, ppl, k_opt):
        shock_c, shock_m, shock_s = self.last_shock_flags

        client_k = 0
        server_k = 0
        comm_k = 0
        client_total_latency = 0
        server_total_latency = 0
        comm_total_latency = 0

        for k, (obs_client, obs_comm, obs_server) in zip(self.history_k, self.history_latency):
            client_k += k + 1
            server_k += (34 - k)
            comm_k += 1

            client_total_latency += obs_client
            server_total_latency += obs_server
            comm_total_latency += obs_comm

        # 防止除以零
        client_comp_per_layer = client_total_latency / max(client_k, 1e-6)
        server_comp_per_layer = server_total_latency / max(server_k, 1e-6)
        comm_avg = comm_total_latency / max(comm_k, 1e-6)

        best_k = None
        best_est = float('inf')

        for k in self.avg_client:
            head_name, _ = utils.get_lm_head_idx(k)
            if k not in self.avg_comm or k not in self.avg_server:
                continue

            client_part = (
                k * client_comp_per_layer if shock_c else self.avg_client[k]
            )
            comm_part = (
                comm_avg * (1 - self.lm_manager.predict_exit_rate(head_name, ppl)) if shock_m else self.avg_comm[k]
            )
            server_part = (
                (34 - k) * server_comp_per_layer * (1 - self.lm_manager.predict_exit_rate(head_name, ppl)) if shock_s else self.avg_server[k]
            )

            est = client_part + comm_part + server_part

            self.logger.log(f'k: {k}')
            self.logger.log(f'avg client: {self.avg_client[k]}')
            self.logger.log(f'avg server: {self.avg_server[k]}')
            self.logger.log(f'avg comm: {self.avg_comm[k]}')
            self.logger.log(f'est: {est}')
            self.logger.log(f'client part: {client_part}')
            self.logger.log(f'server part: {server_part}')
            self.logger.log(f'comm poart: {comm_part}')
            if est < best_est:
                best_est = est
                best_k = k



        self.reset_history()
        #self.reset_avg()
        return best_k if best_k is not None else k_opt



    def get_recent_shock_info(self):
        return list(zip(self.history_k, self.history_latency, self.history))
