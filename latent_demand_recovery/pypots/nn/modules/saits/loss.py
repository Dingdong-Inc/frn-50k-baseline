"""

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


from typing import Callable

import torch.nn as nn

from ....utils.metrics import calc_mae, calc_mse, calc_rmse, calc_mre, calc_reg_focal, calc_mbe

class SaitsLoss(nn.Module):
    def __init__(
        self,
        ORT_weight,
        MIT_weight,
        loss_calc_func: Callable = calc_mae,
    ):
        super().__init__()
        self.ORT_weight = ORT_weight
        self.MIT_weight = MIT_weight
        self.loss_calc_func = loss_calc_func

    def forward(self, reconstruction, X_ori, missing_mask, indicating_mask, norm_val):
        # calculate loss for the observed reconstruction task (ORT)
        reconstruction = reconstruction * norm_val
        X_ori = X_ori * norm_val
        ORT_loss = self.ORT_weight * self.loss_calc_func(
            reconstruction, X_ori, missing_mask
        )
        # calculate loss for the masked imputation task (MIT)
        MIT_loss = self.MIT_weight * self.loss_calc_func(
            reconstruction, X_ori, indicating_mask
        )
        # calculate the loss to back propagate for model updating
        loss = ORT_loss + MIT_loss
        
        
        return loss, ORT_loss, MIT_loss
