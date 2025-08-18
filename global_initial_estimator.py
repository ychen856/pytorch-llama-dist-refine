from Linear_compute_time_model import LinearComputeTimeModel
import utils

class GlobalInitialStageEstimator:
    def __init__(self, lm_manager, logger, total_layers):
        self.lm_manager = lm_manager
        self.logger = logger
        self.total_layers = total_layers

        self.samples = []  # (start_idx, m, edge_time, comm_time, server_time)

    def add_sample(self, start_idx, m, edge_time, comm_time, server_time):
        self.samples.append((start_idx, m, edge_time, comm_time, server_time))

    def predict_best_m(self, ppl, start_idx):
        if not self.samples:
            return self.total_layers - 1  # fallback: last layer

        edge_layer_counts = [m - k + 1 for (k, m, _, _, _) in self.samples]
        edge_latencies = [e for (_, _, e, _, _) in self.samples]

        server_layer_counts = [self.total_layers - 1 - m for (_, m, _, _, _) in self.samples]
        server_latencies = [s for (_, _, _, _, s) in self.samples]

        comm_latencies = [c for (_, _, _, c, _) in self.samples]
        comm_avg = sum(comm_latencies) / max(len(comm_latencies), 1e-6)

        startup_e, per_layer_e = utils.fit_linear_model_non_negative(edge_layer_counts, edge_latencies)
        startup_s, per_layer_s = utils.fit_linear_model_non_negative(server_layer_counts, server_latencies)

        edge_model = LinearComputeTimeModel(startup_e, per_layer_e)
        server_model = LinearComputeTimeModel(startup_s, per_layer_s)

        best_m = None
        best_est = float('inf')

        #for m in range(start_idx, self.total_layers):
        for m in range(start_idx, 10):
            head_name, _ = utils.get_lm_head_idx(m)
            exit_rate = self.lm_manager.predict_exit_rate(head_name, ppl)

            edge_layers = m - start_idx + 1
            server_layers = self.total_layers - 1 - m

            if edge_layers < 1 or server_layers < 0:
                continue

            edge_part = edge_model.estimate_total_time(edge_layers)
            comm_part = comm_avg * (1 - exit_rate)
            server_part = server_model.estimate_total_time(server_layers) * (1 - exit_rate)

            est = edge_part + comm_part + server_part

            self.logger.log(f"Global m={m}, edge={edge_part}, comm={comm_part}, server={server_part}, est={est}")

            if est < best_est:
                best_est = est
                best_m = m

        #return best_m if best_m is not None else self.total_layers - 1
        return best_m if best_m is not None else 9
