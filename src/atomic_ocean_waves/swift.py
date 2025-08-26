"""
Surface Wave Instrument Float with Tracking (SWIFT) buoy functions.
"""


__all__ = [
    "read_swift_file",
    "read_swift_directory",
]


from typing import List, Tuple

import numpy as np
import pandas as pd
import xarray as xr

from . import waves


# def _combine_attrs(variable_attrs: List, context=None) -> dict:
#     """ Combine SWIFT attributes.

#     Handle attributes during concatenation of SWIFT Datasets. Where possible,
#     unique values are taken. Otherwise, values are aggregated into a list.
#     This function is passed to xarray's `combine_attrs` argument.

#     Args:
#         variable_attrs (List): Attribute dictionaries to combine.
#         context (optional): Context information. Defaults to None.

#     Returns:
#         dict: Combined attributes.
#     """
#     attr_keys = _get_unique_keys(variable_attrs)
#     attrs = {}
#     for key in attr_keys:

#         # Return a list of unique attributes for this key.
#         unique_attrs = _get_unique_attrs(variable_attrs, key)

#         # Return first value if entirely unique.
#         if unique_attrs.size == 1:
#             unique_attrs = unique_attrs[0]

#         attrs[key] = unique_attrs

#     return attrs


# def _get_unique_keys(variable_attrs):
#     """ Return unique keys from a set of attributes """
#     return list({key: None for attrs in variable_attrs for key in attrs})


# def _get_unique_attrs(variable_attrs, key) -> np.ndarray:
#     """ Return unique values from a set of attributes """
#     all_attrs = _aggregate_attrs(variable_attrs, key)
#     return pd.unique(np.asarray(all_attrs))  # TODO: try replacing with built-in set


# def _aggregate_attrs(variable_attrs, key) -> List:
#     """ Aggregate all attributes into a list """
#     return [attrs[key] for attrs in variable_attrs if key in attrs.keys()]


# def _attrs_to_datetime(variable_attrs, key) -> List:
#     """ Convert date-like attributes to datetimes """
#     all_attrs = _aggregate_attrs(variable_attrs, key)
#     attrs_as_datetimes = np.sort(pd.to_datetime(all_attrs))
#     return list(attrs_as_datetimes)