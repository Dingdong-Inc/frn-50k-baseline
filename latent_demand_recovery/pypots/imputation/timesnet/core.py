"""

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import torch.nn as nn

from ...nn.functional import nonstationary_norm, nonstationary_denorm
from ...nn.modules.timesnet import BackboneTimesNet
from ...nn.modules.transformer.embedding import DataEmbedding
from ...utils.metrics import calc_mse, calc_mae


class _TimesNet(nn.Module):
    def __init__(
        self,
        n_layers,
        n_steps,
        n_features,
        top_k,
        d_model,
        d_ffn,
        n_kernels,
        dropout,
        apply_nonstationary_norm,
    ):
        super().__init__()

        self.seq_len = n_steps
        self.n_layers = n_layers
        self.apply_nonstationary_norm = apply_nonstationary_norm

        self.enc_embedding = DataEmbedding(
            n_features,
            d_model,
            dropout=dropout,
        )
        self.model = BackboneTimesNet(
            n_layers,
            n_steps,
            0,  # n_pred_steps should be 0 for the imputation task
            top_k,
            d_model,
            d_ffn,
            n_kernels,
        )
        self.layer_norm = nn.LayerNorm(d_model)

        # for the imputation task, the output dim is the same as input dim
        self.projection = nn.Linear(d_model, n_features)

    def forward(self, inputs: dict, training: bool = True) -> dict:
        X_ori, missing_mask = inputs["X"], inputs["missing_mask"]

        if self.apply_nonstationary_norm:
            # Normalization from Non-stationary Transformer
            X, means, stdev = nonstationary_norm(X_ori, missing_mask)
        else:
            X = X_ori

        # embedding
        input_X = self.enc_embedding(X)  # [B,T,C]
        # TimesNet processing
        enc_out = self.model(input_X)

        # project back the original data space
        dec_out = self.projection(enc_out)

        if self.apply_nonstationary_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = nonstationary_denorm(dec_out, means, stdev)

        imputed_data = missing_mask * X_ori + (1 - missing_mask) * dec_out
        results = {
            "imputed_data": imputed_data,
        }

        if training:
            X_ori, indicating_mask, norm_val = inputs["X_ori"], inputs["indicating_mask"], inputs["norm_val"]
            # `loss` is always the item for backward propagating to update the model
            dec_out = dec_out * norm_val
            X_ori = X_ori * norm_val
            loss = calc_mae(dec_out, X_ori, indicating_mask)
            results["loss"] = loss

        return results
