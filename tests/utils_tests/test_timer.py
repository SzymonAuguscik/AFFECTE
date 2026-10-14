from unittest.mock import patch

import pytest

from src.exceptions.TimerNotStartedException import TimerNotStartedException
from src.exceptions.TimerNotStoppedException import TimerNotStoppedException
from src.utils.Timer import Timer


class TestTimer:
    START_TIME: float = 501.6
    STOP_TIME: float = 510.7

    def setup_method(self) -> None:
        self.timer: Timer = Timer()

    @patch("time.time", side_effect=[START_TIME, STOP_TIME])
    def test_get_time(self, time_mock) -> None:
        self.timer.start()
        self.timer.stop()
        assert self.timer.get_time() == self.STOP_TIME - self.START_TIME

    def test_timer_not_started(self) -> None:
        with pytest.raises(TimerNotStartedException):
            self.timer.stop()

    def test_timer_not_stopped(self) -> None:
        self.timer.start()
        with pytest.raises(TimerNotStoppedException):
            self.timer.get_time()
