from src.utils.Visualizer import Visualizer
from tests import UnitTest
from typing import List


class VisualizerTest(UnitTest):
    METRIC_VALUES: List[float] = [0.78, 0.81, 0.76, 0.83]
    NEW_VALUE: float = 0.85
    METRIC_NAME: str = "some metric"

    def setUp(self) -> None:
        self.visualizer: Visualizer = Visualizer()

    def test_update_metric(self) -> None:
        self.visualizer._metrics = {
            self.METRIC_NAME : self.METRIC_VALUES.copy()
        }
        self.visualizer.update_metric(self.METRIC_NAME, self.NEW_VALUE)
        self.assertEqual(self.visualizer._metrics[self.METRIC_NAME], self.METRIC_VALUES + [self.NEW_VALUE])

