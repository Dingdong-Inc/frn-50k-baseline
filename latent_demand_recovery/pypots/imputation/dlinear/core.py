"""

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from typing import Optional

import torch
import torch.nn as nn

from ...nn.modules.autoformer import SeriesDecompositionBlock
from ...nn.modules.dlinear import BackboneDLinear
from ...nn.modules.saits import SaitsLoss


class _DLinear(nn.Module):
    def __init__(
        self,
        n_steps: int,
        n_features: int,
        moving_avg_window_size: int,
        individual: bool = False,
        d_model: Optional[int] = None,
        ORT_weight: float = 1,
        MIT_weight: float = 1,
    ):
        super().__init__()

        self.n_steps = n_steps
        self.n_features = n_features
        self.individual = individual

        self.series_decomp = SeriesDecompositionBlock(moving_avg_window_size)
        self.backbone = BackboneDLinear(n_steps, n_features, individual, d_model)

        if not individual:
            self.linear_seasonal_embedding = nn.Linear(n_features * 2, d_model)
            self.linear_trend_embedding = nn.Linear(n_features * 2, d_model)
            self.linear_seasonal_output = nn.Linear(d_model, n_features)
            self.linear_trend_output = nn.Linear(d_model, n_features)

        # apply SAITS loss function to Transformer on the imputation task
        self.saits_loss_func = SaitsLoss(ORT_weight, MIT_weight)

    def forward(self, inputs: dict, training: bool = True) -> dict:
        X, missing_mask = inputs["X"], inputs["missing_mask"]

        # input preprocessing and embedding for DLinear
        seasonal_init, trend_init = self.series_decomp(X)

        if not self.individual:
            # WDU: the original DLinear paper isn't proposed for imputation task. Hence the model doesn't take
            # the missing mask into account, which means, in the process, the model doesn't know which part of
            # the input data is missing, and this may hurt the model's imputation performance. Therefore, I add the
            # embedding layers to project the concatenation of features and masks into a hidden space, as well as
            # the output layers to project the seasonal and trend from the hidden space to the original space.
            # But this is only for the non-individual mode.
            seasonal_init = torch.cat([seasonal_init, missing_mask], dim=2)
            trend_init = torch.cat([trend_init, missing_mask], dim=2)
            seasonal_init = self.linear_seasonal_embedding(seasonal_init)
            trend_init = self.linear_trend_embedding(trend_init)

        seasonal_output, trend_output = self.backbone(seasonal_init, trend_init)

        if not self.individual:
            seasonal_output = self.linear_seasonal_output(seasonal_output)
            trend_output = self.linear_trend_output(trend_output)

        reconstruction = seasonal_output + trend_output

        imputed_data = missing_mask * X + (1 - missing_mask) * reconstruction
        results = {
            "imputed_data": imputed_data,
        }

        # if in training mode, return results with losses
        if training:
            X_ori, indicating_mask, norm_val = inputs["X_ori"], inputs["indicating_mask"], inputs["norm_val"]
            loss, ORT_loss, MIT_loss = self.saits_loss_func(
                reconstruction, X_ori, missing_mask, indicating_mask, norm_val
            )
            results["ORT_loss"] = ORT_loss
            results["MIT_loss"] = MIT_loss
            # `loss` is always the item for backward propagating to update the model
            results["loss"] = loss

        return results
