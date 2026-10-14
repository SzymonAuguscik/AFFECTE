import numpy as np
import pytest
import torch

from src.utils.LinkConstraintsLoss import LinkConstraintsLoss


class TestLinkConstraintsLoss:
    DEVICE: torch.device = torch.device("cpu")

    def setup_method(self) -> None:
        self.lc_loss: LinkConstraintsLoss = LinkConstraintsLoss(self.DEVICE)

    def test_link_constraints_loss(self) -> None:
        beta: torch.Tensor = torch.Tensor(
            np.array([[0.7, 0.1, 0.15], [0.5, 0.2, 0.3], [0.65, 0.25, 0.1], [0.3, 0.5, 0.7]])
        )

        e: torch.Tensor = torch.Tensor(
            np.array([[0, 1, 1, -1], [1, 0, 1, -1], [1, 1, 0, -1], [-1, -1, -1, 0]])
        )

        loss: float = self.lc_loss(beta, e)
        expected_loss: float = 1.271564
        assert loss == pytest.approx(expected_loss)
