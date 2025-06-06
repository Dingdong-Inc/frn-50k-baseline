"""
Dataset class for self-attention models trained with MIT (masked imputation task) task.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from typing import Union

from ..saits.data import DatasetForSAITS


class DatasetForTransformer(DatasetForSAITS):
    def __init__(
        self,
        data: Union[dict, str],
        return_X_ori: bool,
        return_y: bool,
        file_type: str = "hdf5",
        rate: float = 0.2,
        OT: int = 1
    ):
        super().__init__(data, return_X_ori, return_y, file_type, rate, OT)
