from src.utils.LinkConstraintsLoss import LinkConstraintsLoss
from tests import UnitTest

import numpy as np

import torch


class LinkConstraintsLossTest(UnitTest):
    DEVICE: str = "cpu"

    def setUp(self) -> None:
        self.lc_loss: LinkConstraintsLoss = LinkConstraintsLoss(self.DEVICE)

    def test_link_constraints_loss(self) -> None:
        beta: torch.Tensor = torch.Tensor(np.array([[0.7, 0.1, 0.15],
                                                    [0.5, 0.2, 0.3],
                                                    [0.65, 0.25, 0.1],
                                                    [0.3, 0.5, 0.7]]))

        e: torch.Tensor = torch.Tensor(np.array([[0, 1, 1, -1],
                                                 [1, 0, 1, -1],
                                                 [1, 1, 0, -1],
                                                 [-1, -1, -1, 0]]))

        loss: float = self.lc_loss(beta, e)
        expected_loss: float = 1.271564
        self.assertAlmostEqual(loss, expected_loss)

