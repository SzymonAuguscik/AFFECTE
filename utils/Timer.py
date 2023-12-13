import logging
import time


class Timer:
    def __init__(self):
        self.start_time = None
        self.measured_time = None
        self.logger = logging.getLogger(__name__)

    def start(self):
        self.start_time = time.time()
        self.logger.debug(f"Start time = {self.start_time}s")

    def stop(self):
        if self.start_time is None:
            self.logger.error("Timer was not started!")
            return
        self.measured_time = time.time() - self.start_time
        self.logger.debug(f"Measured time = {self.measured_time}s")

    def get_start_time(self):
        return self.start_time

    def get_time(self):
        if self.measured_time is not None:
            return self.measured_time
        self.logger.error("Time was not measured! Has the timer been started?")

