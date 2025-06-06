"""
The package including the modules of SAITS.

Refer to the paper
`Wenjie Du, David Cote, and Yan Liu.
SAITS: Self-Attention-based Imputation for Time Series.
Expert Systems with Applications, 219:119619, 2023.
<https://arxiv.org/pdf/2202.08516>`_

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from .backbone import BackboneSAITS
from .loss import SaitsLoss
from .embedding import SaitsEmbedding

__all__ = [
    "BackboneSAITS",
    "SaitsLoss",
    "SaitsEmbedding"
]
