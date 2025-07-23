from collections import deque
from Linear_compute_time_model import LinearComputeTimeModel
import utils

class EdgePredictiveSplittingManager:
    def __init__(self, start_idx, lm_manager, logger, shock_alpha=1.5, window_size=10):
        self.start_idx = start_idx
        self.end_idx = None  # To be set by pool

        self.alpha = shock_alpha
        self.window_size = window_size

        self.avg_edge = {}
        self.avg_comm = {}
        self.avg_server = {}

        self.history_k = deque(maxlen=window_size)
        self.history_latency = deque(maxlen=window_size)

        self.lm_manager = lm_manager
        self.logger = logger

    def reset_history(self):
        self.history_k.clear()
        self.history_latency.clear()

    def set_avg_latency(self, m, edge_time, comm_time, server_time):
        self.avg_edge[m] = edge_time
        self.avg_comm[m] = comm_time
        self.avg_server[m] = server_time

    def reset_avg(self):
        self.avg_edge.clear()
        self.avg_comm.clear()
        self.avg_server.clear()

    def is_shock(self, m, obs_edge, obs_comm, obs_server):
        try:
            base_edge = self.avg_edge[m]
            base_comm = self.avg_comm[m]
            base_server = self.avg_server[m]
        except KeyError:
            return False

        shock_e = obs_edge > base_edge * self.alpha
        shock_m = obs_comm > base_comm * self.alpha
        shock_s = obs_server > base_server * self.alpha

        return shock_e or shock_m or shock_s

    def record_latency(self, m, obs_edge, obs_comm, obs_server):
        self.history_k.append(m)
        self.history_latency.append((obs_edge, obs_comm, obs_server))

    def decide_m(self, ppl, opt_m, total_layers):
        self.end_idx = total_layers - 1

        edge_layers_list = [m - self.start_idx + 1 for m in self.history_k]
        server_layers_list = [self.end_idx - m for m in self.history_k]

        startup_e, per_layer_e = utils.fit_linear_model_non_negative(edge_layers_list, [e for (e, _, _) in self.history_latency])
        startup_s, per_layer_s = utils.fit_linear_model_non_negative(server_layers_list, [s for (_, _, s) in self.history_latency])

        edge_model = LinearComputeTimeModel(startup_e, per_layer_e)
        server_model = LinearComputeTimeModel(startup_s, per_layer_s)

        comm_total_latency = sum([comm for (_, comm, _) in self.history_latency])
        comm_k = len(self.history_latency)
        comm_avg = comm_total_latency / max(comm_k, 1e-6)

        best_m = None
        best_est = float('inf')

        for m in range(self.start_idx, self.end_idx + 1):
            head_name, _ = utils.get_lm_head_idx(m)
            exit_rate = self.lm_manager.predict_exit_rate(head_name, ppl)

            edge_layers = m - self.start_idx + 1
            server_layers = self.end_idx - m

            edge_part = edge_model.estimate_total_time(edge_layers)
            comm_part = comm_avg * (1 - exit_rate)
            server_part = server_model.estimate_total_time(server_layers) * (1 - exit_rate)

            est = edge_part + comm_part + server_part

            self.logger.log(f'm: {m}, edge: {edge_part}, comm: {comm_part}, server: {server_part}, est: {est}')

            if est < best_est:
                best_est = est
                best_m = m

        self.reset_history()
        return best_m if best_m is not None else opt_m

class EdgeSplittingManagerPool:
    def __init__(self, num_layers, lm_manager, logger, shock_threshold=7, window_size=10):
        self.num_layers = num_layers
        self.lm_manager = lm_manager
        self.logger = logger
        self.shock_threshold = shock_threshold
        self.window_size = window_size
        self.total_shocks = deque(maxlen=window_size)
        self.managers = {}

    def get_or_create_manager(self, start_idx):
        if start_idx not in self.managers:
            self.logger.log(f"Creating edge splitting manager for start_idx = {start_idx}")
            manager = EdgePredictiveSplittingManager(start_idx=start_idx, lm_manager=self.lm_manager, logger=self.logger, window_size=self.window_size)
            self.managers[start_idx] = manager
        return self.managers[start_idx]

    def is_trigger_override(self):
        return sum(self.total_shocks) >= self.shock_threshold

    def reset_total_shocks(self):
        self.total_shocks.clear()

    def reset_history(self):
        for manager in self.managers.values():
            manager.reset_history()
        self.reset_total_shocks()

    def reset_avg(self):
        for manager in self.managers.values():
            manager.reset_avg()


    def record_latency_and_check_shock(self, start_idx, m, edge_time, comm_time, server_time):
        manager = self.get_or_create_manager(start_idx)
        is_shock = manager.is_shock(m, edge_time, comm_time, server_time)
        self.total_shocks.append(is_shock)
        manager.record_latency(m, edge_time, comm_time, server_time)
        #self.logger.log(f"shock: {is_shock}, total recent shocks: {sum(self.total_shocks)}")
        return is_shock

    def set_avg_latency(self, start_idx, m, edge_time, comm_time, server_time):
        manager = self.get_or_create_manager(start_idx)
        manager.set_avg_latency(m, edge_time, comm_time, server_time)

    def decide_m(self, start_idx, end_idx, ppl):
        manager = self.get_or_create_manager(start_idx)
        self.logger.log(f"Shock threshold met. Running decide_m for start_idx = {start_idx}")
        self.reset_history()
        self.reset_avg()

        return manager.decide_m(ppl, end_idx, 34)