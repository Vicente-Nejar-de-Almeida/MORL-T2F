from typing import List, Tuple
import os
import numpy as np
from aeon.datasets import load_classification
# import pandas as pd

from .reader import load_from_tsfile_to_dataframe


def read_ucr_mts(path: str) -> Tuple[List[np.ndarray], list]:
    """ Wrapper for sktime load_from_tsfile_to_dataframe function """
    # Check mts existence
    if not os.path.isfile:
        raise ValueError(f"THe multivariate time-series file doesn't exist: {path}")

    # Read multivariate time series (mts)
    df, y = load_from_tsfile_to_dataframe(path)

    # Extract list of mts array
    df = df.apply(lambda val: val.to_list())
    ts_list = df.apply(lambda row: np.array(row.to_list()).T, axis=1).to_list()

    # Check mts consistency
    assert len(y) == len(ts_list), 'X and Y have different size'
    cond = [len(ts.shape) == 2 for ts in ts_list]
    assert np.all(cond), 'Error in time series format, shape must be 2d'
    return ts_list, list(y)


def read_ucr_datasets(nameDataset: str, extract_path=None) -> Tuple[np.ndarray, np.ndarray]:
    """ Read ucr datasets
    """
    ts_list, y = load_classification(nameDataset, extract_path=extract_path)
    ts_list = np.transpose(ts_list, (0, 2, 1))
    return ts_list, y
