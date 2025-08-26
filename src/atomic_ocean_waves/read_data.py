"""
Functions for reading ATOMIC data.
"""
# TODO: are single file functions needed? (Probably not, if open_mfdataset
# doesn't raise an error w/ a single file.)


__all__ = [
    "read_swift_file",
    "read_swift_directory",
    "read_wsra_file",
    "read_wsra_directory",
    "read_p3_directory",
]


import glob
import os
import re
from typing import List, Literal, Tuple, Union
from warnings import warn

import numpy as np
import pandas as pd
import xarray as xr


def read_swift_file(filepath: str, **kwargs) -> xr.Dataset:
    """
    Read SWIFT buoy data into an Xarray Dataset.

    Args:
        filepath (str): path to SWIFT netCDF (.nc) file.
        kwargs (optional): additional keyword arguments passed to
            xr.open_dataset.

    Returns:
        xr.Dataset: SWIFT dataset.
    """
    return xr.open_dataset(filepath, **kwargs)


def read_swift_directory(
    directory: str,
    **kwargs,
) -> xr.Dataset:
    """
    Read and concatenate a directory of SWIFT files into a Dataset.

    Args:
        directory (str): directory containing SWIFT files,
        kwargs (optional): additional keyword arguments passed to
            xr.open_mfdataset.

    Raises:
        FileNotFoundError: If no files of type `file_type` are found
            inside of `directory`.

    Returns:
        xr.Dataset: all SWIFT files concatenated into a single Dataset.
    """
    swift_files = get_files_from_directory(directory)

    return xr.open_mfdataset(
        swift_files,
        concat_dim='id',
        combine='nested',
        combine_attrs=combine_attrs,
        preprocess=_preprocess_swift_file,
        **kwargs,
    )


def _preprocess_swift_file(
    swift_ds: xr.Dataset,
) -> xr.Dataset:
    swift_id = _get_swift_id_from_nc(swift_ds)
    return swift_ds.expand_dims(dim={'id': [swift_id]}, axis=0)


def _get_swift_id_from_nc(swift_ds: xr.Dataset) -> str:
    # Extract SWIFT ID from the dataset attributes (e.g., SWIFT 16).
    match = re.search(r'SWIFT [0-9][0-9]', swift_ds.attrs['description'])
    if match:
        # Return ID number only.
        swift_name_id = match.group(0)
        return swift_name_id.split(' ')[-1]
    raise ValueError('SWIFT ID not found.')


def read_wsra_file(filepath: str, index_by_time: bool = True) -> xr.Dataset:
    """
    Read and concatenate a directory of Level 4 WSRA files into a Dataset.

    Args:
        filepath (str): path to WSRA file.
        index_by_time (bool, optional): if True, use time as the primary index.
            Otherwise use the default `trajectory`. Defaults to True.

    Returns:
        xr.Dataset: WSRA dataset.
    """
    wsra_ds = xr.open_dataset(filepath)

    wsra_ds.attrs['pywsra_file'] = os.path.basename(filepath)

    if index_by_time:
        wsra_ds = _replace_coord_with_var(wsra_ds, 'trajectory', 'time')
        wsra_ds = wsra_ds.sortby('time')

    return wsra_ds


def read_wsra_directory(
    directory: str,
    index_by_time: bool = True,
    **kwargs,
) -> xr.Dataset:
    """
    Read a directory of Level 4 WSRA data files and concatenate into a Dataset.

    Args:
        directory (str): directory containing WSRA files
        index_by_time (bool, optional): if True, use time as the primary index.
            Otherwise use the default `trajectory`. Defaults to True.
        kwargs (optional): additional keyword arguments passed to
            xr.open_mfdataset.

    Raises:
        FileNotFoundError: If no files of type `file_type` are found
            inside of `directory`.

    Returns:
        xr.Dataset: all WSRA files concatenated into a single Dataset.
    """
    wsra_files = get_files_from_directory(directory)

    return xr.open_mfdataset(
        wsra_files,
        concat_dim='time' if index_by_time else 'trajectory',
        combine='nested',
        combine_attrs=combine_attrs,
        preprocess=_preprocess_wsra_file,
        decode_timedelta=True,
        **kwargs,
    )


def _preprocess_wsra_file(
    wsra_ds: xr.Dataset,
    index_by_time: bool = True
) -> xr.Dataset:
    if index_by_time:
        wsra_ds = _replace_coord_with_var(wsra_ds, 'trajectory', 'time')
        wsra_ds = wsra_ds.sortby('time')
    else:
        pass
    return wsra_ds


def _replace_coord_with_var(
    ds: xr.Dataset,
    coord: str,
    var: str
) -> xr.Dataset:
    """Replace a Dataset coordinate with another variable of the same shape.

    Note: `coord` and `var` must have the same shape.  The original coord is
    dropped in this process.

    Args:
        ds (xr.Dataset): The xarray Dataset to operate on.
        coord (str): Coordinate to be replaced.
        var (str): Variable to replace it with.

    Returns:
        xr.Dataset: The xarray Dataset with coord replaced by var.
    """
    ds.coords[coord] = ds[var]
    dropped = ds.drop_vars([var])
    renamed = dropped.rename({coord: var})
    return renamed


def read_p3_directory(
    directory: str,
    **kwargs,
) -> xr.Dataset:
    """
    Read and concatenate a directory of SWIFT files into a Dataset.

    Args:
        directory (str): directory containing SWIFT files,
        kwargs (optional): additional keyword arguments passed to
            xr.open_mfdataset.

    Raises:
        FileNotFoundError: If no files of type `file_type` are found
            inside of `directory`.

    Returns:
        xr.Dataset: all WSRA files concatenated into a single Dataset.
    """
    p3_files = get_files_from_directory(directory)

    return xr.open_mfdataset(
        p3_files,
        concat_dim='time',
        combine='nested',
        combine_attrs=combine_attrs,
        **kwargs,
    )


def read_saildrone_asc_file(
    filepath: str,
    data_type: Literal['pandas', 'xarray'] = 'pandas',
) -> Union[pd.DataFrame, xr.Dataset]:
    """ Read Saildrone bulk wave data (.asc) as a DataFrame or Dataset.

    Args:
        filepath (str): Path to data file.
        data_type (Literal['pandas', 'xarray']): Return type for the data.

    Returns:
        Union[pd.DataFrame, xr.Dataset]: Saildrone bulk wave data.
    """
    # Parse column information from the header.
    variables, columns, descriptions, attrs, start_data = _parse_saildrone_asc_header(filepath)

    # Get the file basename, which also contains mission date information.
    file_basename = os.path.basename(filepath)
    attrs['filename'] = file_basename

    # Remaining data is space-delimited.
    # TODO: assumes variables are in order (could sort via `columns`).
    sd_df = pd.read_csv(filepath,
                        sep=r'\s+',
                        names=variables,
                        skiprows=start_data)

    # Convert date and time columns to a datetime index.
    datestr_series = sd_df.index + 'T' + sd_df['TAX_DATESTRING(T[GT=MSS],MSS,"seconds")']
    sd_df['TIME'] = pd.to_datetime(datestr_series, format='%d-%b-%YT%H:%M:%S')
    sd_df = (sd_df
             .drop(columns=['TAX_DATESTRING(T[GT=MSS],MSS,"seconds")'])
             .set_index('TIME'))

    if data_type == 'xarray':
        sd_ds = sd_df.to_xarray()
        sd_ds.attrs.update(attrs)
        sd_ds = _assign_saildrone_variable_attrs(sd_ds, variables, descriptions)
        return sd_ds
    elif data_type == 'pandas':
        return sd_df
    else:
        raise ValueError(f'{data_type} not supported.')


def _assign_saildrone_variable_attrs(sd_ds: xr.Dataset, variables: List, descriptions: List) -> xr.Dataset:
    """ Assign attributes to USAFR met Dataset variables."""
    for var, des in zip(variables, descriptions):
        if var in sd_ds.keys():
            sd_ds[var].attrs['description'] = des
    return sd_ds


def _parse_saildrone_asc_header(filepath: str) -> Tuple[List, List, List, dict, int]:
    """ Parse Saildrone bulk wave .asc file header.

    Returns variable names, column numbers, descriptions, dataset
    attributes, and the row corresponding to the start of data.
    """
    file = open(filepath)

    attrs = {}
    variables = []
    columns = []
    descriptions = []
    start_data = 0

    # Define a regex pattern to capture column number, variable name,
    # and description (if present).
    column_re_pattern = re.compile(r'Column\s+(\d+):(.*?)(?:is(.*?)$|$)')

    for i, line in enumerate(file):
        if 'DATA SET' in line:
            attrs['dataset name'] = line.split(':', maxsplit=1)[-1].strip()
        elif 'TIME' in line:
            attrs['time coverage'] = line.split(':', maxsplit=1)[-1].strip()
        elif 'Column' in line:
            col_var_des_match = column_re_pattern.search(line)
            if col_var_des_match:
                col = col_var_des_match.group(1)
                var = col_var_des_match.group(2)
                des = col_var_des_match.group(3)

                if var is not None:
                    var = var.strip().replace(' ', '_')
                if des is not None:
                    des = des.strip()
                else: des = ''

                variables.append(var)
                columns.append(col)
                descriptions.append(des)

        # Data starts after the last column line.
        else:
            start_data = i
            break

    file.close()
    return variables, columns, descriptions, attrs, start_data


def get_files_from_directory(
    directory: str,
    file_type: str = 'nc',
    sort: bool = True
) -> List:
    """ Return all files of in the top level of a directory.

    Args:
        directory (str): Path to directory of files.
        file_type (str, optional): File type to glob. Defaults to 'nc'.
        sort (bool, optional): Sort files, if True. Defaults to True.

    Raises:
        FileNotFoundError: If no files of type `file_type` are found in
            the `directory`.

    Returns:
        _type_: _description_
    """
    files = glob.glob(os.path.join(directory, '*' + file_type))

    if sort:
        files.sort()

    if not files:
        raise FileNotFoundError(
            f'No files of type "{file_type}" found in "{directory}". '
            'Please double check the directory and file_type.')

    return files


def combine_attrs(variable_attrs: List, context=None) -> dict:
    """ Combine multiple dataset attributes.

    Handle attributes during the concatenation of Xarray Datasets. Where
    possible, unique values are taken. Otherwise, values are aggregated
    into a list. This function is passed to xarray's `combine_attrs`
    argument (e.g., in `xr.open_mfdataset`).

    Args:
        variable_attrs (List): Attribute dictionaries to combine.
        context (optional): Context information. Defaults to None.

    Returns:
        dict: Combined attributes.
    """
    attr_keys = _get_unique_keys(variable_attrs)
    attrs = {}
    for key in attr_keys:

        # Return a list of unique attributes for this key.
        unique_attrs = _get_unique_attrs(variable_attrs, key)

        # Return first value if entirely unique.
        if unique_attrs.size == 1:
            unique_attrs = unique_attrs[0]

        attrs[key] = unique_attrs

    return attrs


def _get_unique_keys(variable_attrs: dict) -> List:
    """ Return unique keys from a set of attributes """
    return list({key: None for attrs in variable_attrs for key in attrs})


def _get_unique_attrs(variable_attrs: dict, key: str) -> np.ndarray:
    """ Return unique values from a set of attributes """
    all_attrs = _aggregate_attrs(variable_attrs, key)
    return pd.unique(np.asarray(all_attrs))


def _aggregate_attrs(variable_attrs: dict, key: str) -> List:
    """ Aggregate all attributes into a list """
    return [attrs[key] for attrs in variable_attrs if key in attrs.keys()]


def _attrs_to_datetime(variable_attrs: dict, key: str) -> List:
    """ Convert date-like attributes to datetimes """
    all_attrs = _aggregate_attrs(variable_attrs, key)
    attrs_as_datetimes = np.sort(pd.to_datetime(all_attrs))
    return list(attrs_as_datetimes)
