from typing import Union

from ..saits.data import DatasetForSAITS


class DatasetForImputeFormer(DatasetForSAITS):
    def __init__(
        self,
        data: Union[dict, str],
        return_X_ori: bool,
        return_y: bool,
        file_type: str = "hdf5",
        rate: float = 0.2,
    ):
        super().__init__(data, return_X_ori, return_y, file_type, rate)