import logging

class Logger:
    def __init__(self, filepath=None):
        self.logger = logging.getLogger('MyLogger')
        self.logger.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s | %(message)s')

        console = logging.StreamHandler()
        console.setFormatter(formatter)
        self.logger.addHandler(console)

        if filepath:
            file_handler = logging.FileHandler(filepath)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def log(self, msg):
        self.logger.info(msg)