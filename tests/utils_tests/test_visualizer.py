from src.utils.Visualizer import Visualizer


class TestVisualizer:
    METRIC_VALUES: list[float] = [0.78, 0.81, 0.76, 0.83]
    NEW_VALUE: float = 0.85
    METRIC_NAME: str = "some metric"

    def setup_method(self) -> None:
        self.visualizer: Visualizer = Visualizer()

    def test_update_metric(self) -> None:
        self.visualizer._metrics = {self.METRIC_NAME: self.METRIC_VALUES.copy()}
        self.visualizer.update_metric(self.METRIC_NAME, self.NEW_VALUE)
        assert self.visualizer._metrics[self.METRIC_NAME] == self.METRIC_VALUES + [self.NEW_VALUE]
