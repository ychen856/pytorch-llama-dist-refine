class ExitWeightManager:
    def __init__(self,
                 lambda_base=0.3,
                 lambda_max=0.8,
                 linear_lambda=0.5,
                 mode="bandwidth-aware"):
        """
        Exit Weight Manager

        Args:
            lambda_base (float): baseline weight
            lambda_max (float): max weight
            linear_lambda (float): for linear mode, interpolation factor (0~1)
            mode (str): fixed | linear-exit-rate | exit-rate | bandwidth-aware
        """
        self.lambda_base = lambda_base
        self.lambda_max = lambda_max
        self.linear_lambda = linear_lambda
        self.mode = mode

    def compute_weight(self, logger,
                        exit_rate,
                        client_compute_time=None,
                        comm_time=None,
                        server_compute_time=None):
        if self.mode == "default":
            return 0
        elif self.mode == "fixed":
            return self.lambda_base

        elif self.mode == "linear-exit-rate":
            # 線性版本：early_weight = λ * exit_rate + (1-λ)
            return self.linear_lambda * exit_rate + (1 - self.linear_lambda)

        elif self.mode == "exit-rate":
            return self.lambda_base + (self.lambda_max - self.lambda_base) * exit_rate

        elif self.mode == "bandwidth-aware":
            if client_compute_time is None or comm_time is None or server_compute_time is None:
                raise ValueError("Compute time and comm time must be provided in bandwidth-aware mode.")

            denom = comm_time + server_compute_time + 1e-6
            logger.log(f'denom: {denom}')
            logger.log(f'client_compute_time: {client_compute_time}')
            #comm_factor = client_compute_time / denom
            comm_factor = denom / client_compute_time
            logger.log(f'comm_factor: {comm_factor}')
            logger.log(f'exit_rate: {exit_rate}')
            if comm_factor < 8:
                comm_factor = 1

            weight = self.lambda_base + (self.lambda_max - self.lambda_base) * exit_rate * comm_factor
            return weight

        else:
            raise ValueError(f"Unknown mode: {self.mode}")
