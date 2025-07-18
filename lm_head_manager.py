class ExitTracker:
    def __init__(self, a=1, b=1, gamma=1.0):
        self.a = a
        self.b = b
        self.gamma = gamma
        self.successes = 0
        self.total = 0

    def update(self, success: bool):
        self.successes += int(success)
        self.total += 1

    def predict_exit_rate(self):
        s, t, a, b, g = self.successes, self.total, self.a, self.b, self.gamma
        return (s + g * a) / (t + g * (a + b))

    def reset(self):
        self.successes = 0
        self.total = 0


class LMHead:
    def __init__(self, head_name, ppl_list, param_dict):
        self.head_name = head_name
        self.trackers = {}
        for ppl in ppl_list:
            key = (head_name, ppl)
            params = param_dict.get(key, {'a': 1, 'b': 1, 'gamma': 1.0})
            self.trackers[ppl] = ExitTracker(**params)

    def update(self, ppl, success: bool):
        self.trackers[ppl].update(success)

    def predict_exit_rate(self, ppl):
        return self.trackers[ppl].predict_exit_rate()

    def get_all_exit_rates(self):
        return {ppl: tracker.predict_exit_rate() for ppl, tracker in self.trackers.items()}

    def reset(self):
        for tracker in self.trackers.values():
            tracker.reset()


class LMHeadManager:
    def __init__(self, head_names, ppl_list, param_dict, logger):
        self.heads = {
            head_name: LMHead(head_name, ppl_list, param_dict)
            for head_name in head_names
        }
        self.logger = logger
    def update(self, head_name, ppl, success: bool):
        self.heads[head_name].update(ppl, success)

    def predict_exit_rate(self, head_name, ppl):
        self.logger.log(f'head name: {head_name}')
        self.logger.log(f'ppl: {ppl}')
        return self.heads[head_name].predict_exit_rate(ppl)

    def get_all_exit_rates(self):
        return {
            head: self.heads[head].get_all_exit_rates()
            for head in self.heads
        }

    def reset(self):
        for head in self.heads.values():
            head.reset()
