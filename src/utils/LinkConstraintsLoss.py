import numpy as np

import warnings
import torch
warnings.filterwarnings("ignore", category=UserWarning)


class LinkConstraintsLoss(torch.nn.Module):
    """
    LinkConstraintsLoss is used as a regularization method. It relies on creating links between vectors/embeddings
    from the same class.

    Attributes
    ----------
    _device : torch.device
        The device to perform calculation on.

    Examples
    --------
    model_loss = torch.nn.BCELoss()
    lc_loss = LinkConstraintsLoss("gpu")
    model = ConvolutionalNeuralNetwork() # exemplary torch.nn.Module
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    loss = 0
    for X, y in zip(features, labels):
        pred = model(X)
        loss += model_loss(pred, y)
        loss += lc_loss(X, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    See also
    --------
    https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/s12911-021-01546-2

    """
    def __init__(self, device: torch.device) -> None:
        """
        Initiate LinkConstraintsLoss with calculation device.

        Parameters
        ----------
        device : torch.device
            The device to perform calculation on.

        """
        super(LinkConstraintsLoss, self).__init__()
        self._device: torch.device = device

    def forward(self, beta: torch.Tensor, e: np.ndarray) -> float:
        """
        Perform forward pass of the data.

        Parameters
        ----------
        beta : torch.Tensor
            The input data to be propagated.
        e : np.ndarray
            Link coefficients to be used for link constraints.

        Returns
        -------
        loss : float
            The link constraints loss.

        """
        i: torch.Tensor = torch.arange(beta.size(0)).reshape(-1, 1).to(self._device)
        j: torch.Tensor = torch.arange(beta.size(0)).to(self._device)
        loss: float = 0.5 * torch.norm(beta[i, 0] - e[i, j] * beta[j, 0], p=2)
        return loss

