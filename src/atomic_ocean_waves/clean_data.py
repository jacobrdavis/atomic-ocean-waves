"""
Functions for cleaning ATOMIC datasets.
"""

__all__ = [
    "mask_missing_data",
]


from typing import Any

import numpy as np


def mask_missing_data(ds, missing_data_flag: Any = -999):
    """
    Mask missing data in an xarray dataset.
    """
    # Returns NaN anywhere data is not equal to missing data flag value.
    return ds.where(ds != missing_data_flag, np.nan)

