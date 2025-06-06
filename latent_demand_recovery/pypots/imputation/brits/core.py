"""
The implementation of BRITS for the partially-observed time-series imputation task.

Refer to the paper "Cao, W., Wang, D., Li, J., Zhou, H., Li, L., & Li, Y. (2018).
BRITS: Bidirectional Recurrent Imputation for Time Series. NeurIPS 2018."

Notes
-----
Partial implementation uses code from https://github.com/caow13/BRITS. The bugs in the original implementation
are fixed here.

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import torch.nn as nn

from ...nn.modules.brits import BackboneBRITS
from ...utils.metrics import calc_mae

from typing import Callable


class _BRITS(nn.Module):
    """model BRITS: Bidirectional RITS
    BRITS consists of two RITS, which take time-series data from two directions (forward/backward) respectively.

    Parameters
    ----------
    n_steps :
        sequence length (number of time steps)

    n_features :
        number of features (input dimensions)

    rnn_hidden_size :
        the hidden size of the RNN cell
        
    loss_calc_func :
        the loss function. Callable. default=calc_mae

    """

    def __init__(
        self,
        n_steps: int,
        n_features: int,
        rnn_hidden_size: int,
        loss_calc_func: Callable = calc_mae
    ):
        super().__init__()
        self.n_steps = n_steps
        self.n_features = n_features
        self.rnn_hidden_size = rnn_hidden_size
        self.loss_calc_func = loss_calc_func

        self.model = BackboneBRITS(n_steps, n_features, rnn_hidden_size, self.loss_calc_func)

    def forward(self, inputs: dict, training: bool = True) -> dict:
        (
            imputed_data,
            f_reconstruction,
            b_reconstruction,
            f_hidden_states,
            b_hidden_states,
            consistency_loss,
            reconstruction_loss,
        ) = self.model(inputs)

        results = {
            "imputed_data": imputed_data,
        }

        # if in training mode, return results with losses
        if training:
            results["consistency_loss"] = consistency_loss
            results["reconstruction_loss"] = reconstruction_loss
            loss = consistency_loss + reconstruction_loss

            # `loss` is always the item for backward propagating to update the model
            results["loss"] = loss
            results["reconstruction"] = (f_reconstruction + b_reconstruction) / 2
            results["f_reconstruction"] = f_reconstruction
            results["b_reconstruction"] = b_reconstruction

        return results
