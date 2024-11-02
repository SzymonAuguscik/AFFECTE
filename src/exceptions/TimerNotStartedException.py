class TimerNotStartedException(Exception):
    def __init__(self):
        self.message = f"Timer has not been started! Did you use 'start()' method?"
        super().__init__(self.message)