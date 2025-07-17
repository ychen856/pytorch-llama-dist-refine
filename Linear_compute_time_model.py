class LinearComputeTimeModel:
    def __init__(self, startup_latency: float, per_layer_latency: float):
        self.t0 = startup_latency
        self.delta_t = per_layer_latency

    def estimate_total_time(self, k: int) -> float:
        return self.t0 + k * self.delta_t