from src.exceptions.TimerNotStartedException import TimerNotStartedException
from src.exceptions.TimerNotStoppedException import TimerNotStoppedException
from src.utils.Timer import Timer
from unittest.mock import patch
from tests import UnitTest


class TimerTest(UnitTest):
    START_TIME: float = 501.6
    STOP_TIME: float = 510.7

    def setUp(self) -> None:
        self.timer: Timer = Timer()

    @patch("time.time", side_effect=[START_TIME, STOP_TIME])
    def test_get_time(self, time_mock) -> None:
        self.timer.start()
        self.timer.stop()
        self.assertEqual(self.timer.get_time(), self.STOP_TIME - self.START_TIME)

    def test_timer_not_started(self) -> None:
        self.assertRaises(TimerNotStartedException, self.timer.stop)

    def test_timer_not_stopped(self) -> None:
        self.timer.start()
        self.assertRaises(TimerNotStoppedException, self.timer.get_time)

