class TimerNotStoppedException(Exception):
    def __init__(self):
        self.message = f"Timer has not been stopped! Did you use 'stop()' method before trying to get measured time?"
        super().__init__(self.message)