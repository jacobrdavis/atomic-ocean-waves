"""
Wide Swath Radar Altimeter (WSRA) functions.
"""

__all__ = [
    "wn_spectrum_to_fq_dir_spectrum",
    "calculate_mean_spectral_area",
    "calculate_wn_mag_and_dir",
]


from typing import List, Tuple
from warnings import warn

import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

from . import waves


def wn_spectrum_to_fq_dir_spectrum(
    energy: np.ndarray,
    wavenumber_east: np.ndarray,
    wavenumber_north: np.ndarray,
    depth: float = 1000.0,  #TODO: np.inf?
    var_rtol: float = 0.02,
    regrid: bool = True,
    directional_resolution: float = 1,  # deg
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """_summary_

    THIS IS DIFFERENT THAN STORED IN PYWSRA
    INPUT ENERGY IS NOT A DENSITY
    Note: if `energy` does not contain 3 dimensions, a new axis of size 1 is
        added to represent `t`.
    The returned arrays are gridded (direction increases along axis 0 and
    repeats along axis 1; frequency increases along axis 1 and repeats along
    axis 0).
    #TODO: describe regrid.
    `var_rtol` is computed as abs(`fq_var` - `wn_var`) / abs(`wn_var`) where
    `wn_var` and `fq_var` are the wavenumber and frequency spectra variances,
    respectively.  This value is also for `rtol` in np.allclose(...).

    Args:
        energy (np.ndarray): Wavenumber energy spectrum with shape (x, y, t)
        wavenumber_east (np.ndarray): East wavenumbers with shape (x,)
        wavenumber_north (np.ndarray): North wavenumbers with shape (y,)
        depth (float, optional): Water depth. Defaults to 1000.0. TODO: should have shape t
        var_rtol (float, optional): Relative tolerance (see notes) between
            wavenumber and frequency spectrum variance.  Defaults to 0.02.
        regrid (bool, optional): If True, regrid the output spectra onto
            uniform directions and frequencies. Defaults to True.
        directional_resolution (float, optional): Uniform directional spectrum
            resolution to use if regridding. Defaults to 1 degree.

    Returns:
        If `regrid` == `True`, a Tuple containing
        np.ndarray: Frequency-direction energy density spectrum with shape (d, f, t)
        np.ndarray: Gridded directions with shape (d, f)
        np.ndarray: Gridded frequencies with shape (d, f)

        Otherwise, if `regrid` == `False`, a Tuple containing
        np.ndarray: Frequency-direction energy density spectrum with shape (x, y, t)
        np.ndarray: Gridded directions with shape (x, y)
        np.ndarray: Gridded frequencies with shape (x, y)

    """
    # Require shape (d, f, t).  Create a new axis to represent `t` if `energy`
    # is only 2D.
    if energy.ndim < 3:
        energy = energy[:, :, None]

    # Convert energy to energy density.
    spectral_area = calculate_mean_spectral_area(wavenumber_east,  # rad^2/m^2
                                                 wavenumber_north)
    energy_density_wn = energy / spectral_area  # m^4/rad^2

    if regrid:
        fq_dir_spectrum = _wn_spectrum_to_fq_dir_spectrum_regrid(
            energy_density_wn=energy_density_wn,
            wavenumber_east=wavenumber_east,
            wavenumber_north=wavenumber_north,
            depth=depth,
            directional_resolution=directional_resolution,
            var_rtol=var_rtol,
        )

    else:
        fq_dir_spectrum = _wn_spectrum_to_fq_dir_spectrum_no_regrid(
            energy_density_wn=energy_density_wn,
            wavenumber_east=wavenumber_east,
            wavenumber_north=wavenumber_north,
            depth=depth,
        )

    energy_density_fq_dir, direction, frequency = fq_dir_spectrum

    return energy_density_fq_dir, direction, frequency


def _wn_spectrum_to_fq_dir_spectrum_regrid(
    energy_density_wn: np.ndarray,
    wavenumber_east: np.ndarray,
    wavenumber_north: np.ndarray,
    depth: float,
    directional_resolution: float,
    var_rtol: float = 0.02,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ """

    positive_wavenumber = wavenumber_north[wavenumber_north > 0]
    angular_frequency_1d = waves.intrinsic_dispersion(positive_wavenumber)

    direction_1d = np.deg2rad(np.arange(0, 360 + directional_resolution, directional_resolution))

    angular_frequency, direction = np.meshgrid(angular_frequency_1d,
                                               direction_1d)
    frequency = angular_frequency / (2 * np.pi)

    wavenumber = waves.dispersion_solver(angular_frequency/(2*np.pi),
                                         depth)
    wavenumber_direction_x = wavenumber * np.cos(direction)
    wavenumber_direction_y = wavenumber * np.sin(direction)

    original_points = (wavenumber_east, wavenumber_north)
    interpolation_points = (wavenumber_direction_x, wavenumber_direction_y)
    interpolator = RegularGridInterpolator(original_points,
                                           energy_density_wn)
    energy_density_intp = interpolator(interpolation_points)
    energy_density_fq_dir = waves.wn_energy_to_fq_energy(energy_density_intp,
                                                         wavenumber,
                                                         depth)

    fq_dir_spectrum_var = _calculate_fq_dir_spectrum_var(energy_density_fq_dir,
                                                         direction[:, 0],
                                                         frequency[0])

    wn_spectrum_var = _calculate_wn_spectrum_var(energy_density_wn,
                                                 wavenumber_east,
                                                 wavenumber_north,
                                                 blank_corners=True)

    if not np.allclose(fq_dir_spectrum_var, wn_spectrum_var,
                       rtol=var_rtol, equal_nan=True):
        perc_err = _var_error(fq_dir_spectrum_var, wn_spectrum_var)

        # Reject results where the percent error on variance is less than the
        # specified tolerance, `var_rtol`. Values are replaced by an array of
        # NaNs of the same size. #TODO: will want to do this for regrid=False
        valid_var = perc_err < var_rtol * 100
        energy_density_fq_dir = np.where(valid_var,
                                         energy_density_fq_dir,
                                         np.NaN)

        # Warn about the variance mismatch. Omit percent errors that are NaNs.
        num_invalid_var = np.sum(~valid_var)
        perc_err_notnan = perc_err[~np.isnan(perc_err)]
        max_err = perc_err_notnan.max().round(2)
        warn(
            f'Variance mismatch in {num_invalid_var} values: '
            f'maximum percent error between frequency and wavenumber '
            f'spectra variance is {max_err}%.'
        )
    return energy_density_fq_dir, direction, frequency


def _wn_spectrum_to_fq_dir_spectrum_no_regrid(
    energy_density_wn: np.ndarray,
    wavenumber_east: np.ndarray,
    wavenumber_north: np.ndarray,
    depth: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    wavenumber, direction = calculate_wn_mag_and_dir(wavenumber_east,
                                                     wavenumber_north)
    direction = waves.trig_to_met(direction)
    angular_frequency = waves.intrinsic_dispersion(wavenumber, depth)
    frequency = angular_frequency / (2 * np.pi)
    energy_density_fq_dir = waves.wn_energy_to_fq_energy(energy_density_wn,
                                                         wavenumber,
                                                         depth)
    return energy_density_fq_dir, direction, frequency


def _calculate_fq_dir_spectrum_var(
    energy_density,
    direction,
    frequency,
):
    scalar_energy_density = np.trapz(energy_density, direction, axis=0)
    variance = np.trapz(scalar_energy_density, frequency, axis=0)
    return variance


def _calculate_wn_spectrum_var(
    energy_density,
    wavenumber_east,
    wavenumber_north,
    blank_corners: bool = False,
):
    if blank_corners:
        energy_density = _blank_wn_spectrum_corners(energy_density,
                                                    wavenumber_east,
                                                    wavenumber_north)

    energy_density_north = np.trapz(energy_density, wavenumber_east, axis=0)
    variance = np.trapz(energy_density_north, wavenumber_north, axis=0)
    return variance


def _var_error(estimated_var, actual_var):
    return np.abs(estimated_var - actual_var) / np.abs(actual_var) * 100


def _blank_wn_spectrum_corners(
    energy_density,
    wavenumber_east,
    wavenumber_north,
):
    #ASSUMES square!
    # Blank the variance in the corners which are not regridded
    wavenumber = calculate_wn_mag_and_dir(wavenumber_east, wavenumber_north)[0]
    in_circle = wavenumber >= wavenumber_north.max()
    energy_density_no_corners = energy_density.copy()
    energy_density_no_corners[in_circle] = 0
    return energy_density_no_corners


def calculate_mean_spectral_area(
    wavenumber_east: np.ndarray,
    wavenumber_north: np.ndarray,
) -> float:
    east_spacing = np.diff(wavenumber_east)
    north_spacing = np.diff(wavenumber_north)
    areas = np.outer(east_spacing, north_spacing)
    return areas.mean() # rad^2/m^2


def calculate_wn_mag_and_dir(
    wavenumber_east: np.ndarray,
    wavenumber_north: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    wavenumber_east_grid, wavenumber_north_grid = np.meshgrid(wavenumber_east,
                                                              wavenumber_north,
                                                              indexing='xy')
    magnitude = np.sqrt(wavenumber_east_grid**2 + wavenumber_north_grid**2)
    direction = np.arctan2(wavenumber_north_grid, wavenumber_east_grid)  # TODO: swap for met conv?
    return magnitude, direction


#TODO: use Chris table as test
def correct_mss_for_rain(mss_0, rain_rate, altitude):
    """Correct WSRA mean square slopes for rain attenuation.

    WSRA mean square slope is calculated from the slope of the backscattered
    power versus the tangent squared of the off-nadir angle.  Since scans
    further from nadir have a slightly longer path length, they experience more
    rain attenuation.  This function accounts for this effect by adjusting the
    original mean square slope, `mss_0`, by an attenuation factor, DN:

    mss = (1/mss_0 - DN/const)^(-1)

    where const = 20log10(e).  The attenuation DN is a function of rainfall
    rate and radar altitude.

    Args:
        mss_0 (np.ndarray): original WSRA mean square slope with shape (n,)
        rain_rate (np.ndarray): rainfall rate in mm/hr with shape (n,)
        altitude (np.ndarray): radar altitude in m with shape (n,)

    Returns:
        np.ndarray: mean square slope corrected for rain attenuation with
            shape (n,).
    """
    altitude_km = altitude * 10**(-3)
    atten = calculate_rain_attenuation(rain_rate, altitude_km)
    const = 20 * np.log10(np.e)
    return (1/mss_0 - atten/const)**(-1)


def calculate_rain_attenuation(
    rain_rate: np.ndarray,
    altitude_km: np.ndarray,
) -> np.ndarray:
    """Calculate rain attenuation based on rainfall rate and radar altitude.

    The attenuation DN, expressed in dB, is:

    DN = alpha * rain_rate * altitude_km

    where `alpha` = 0.16 dBZ/km/(mm/hr) is a 2-way attenuation coefficient,
    `rain_rate` is the rainfall rate in mm/hr, and `altitude_km` is the radar
    altitude in km.

    Args:
        rain_rate (np.ndarray): rainfall rate in mm/hr with shape (n,)
        altitude_km (np.ndarray): radar altitude in km with shape (n,)

    Returns:
        np.ndarray: attenuation, in dB, at each rain rate and altitude with
            shape (n,).
    """
    alpha = 0.16  # 2-way attenuation coeff in dBZ/km/(mm/hr)
    return alpha * rain_rate * altitude_km


# def _combine_attrs(variable_attrs: List, context=None) -> dict:
#     """ WSRA attribute handler passed to xr.concat.

#     If `variable_attrs` contains metadata at the Dataset level, concatenate the
#     attributes accordingly.  Otherwise, if `variable_attrs` contains variable
#     descriptions at the DataArray level, pass back the first set of attributes.

#     Args:
#         variable_attrs (List): Attribute dictionaries to combine.
#         TODO: context (_type_, optional): _description_. Defaults to None.

#     Returns:
#         dict: Combined attributes.
#     """
#     # Check if the keys are at the Dataset level. If so, concatenate them.
#     # Otherwise, they are DataArray attributes and only the first is taken.
#     if 'title' in variable_attrs[0].keys():  #  all(key in variable_attrs[0].keys() for key in ATTR_KEYS)
#         # TODO: check if any of ATTR_KEYS in keys?
#         attrs = _concat_attrs(variable_attrs)
#     else:
#         attrs = variable_attrs[0]
#     return attrs


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

# def _concat_attrs(variable_attrs: List):
#     """Concatenate WSFA metadata attributes.

#     Handle attributes during concatenation of WSRA Datasets.  Explicit
#     handling is defined for standard WSRA attributes.  Where possible, unique
#     values are taken. Otherwise, values are aggregated into a list.

#     For all non-standard WSRA attributes, only unique values are taken.

#     Args:
#         variable_attrs (List): Attribute dictionaries to combine.

#     Raises:
#         KeyError: if `variable_attrs.keys()` contains a key not in `ATTR_KEYS`.

#     Returns:
#         dict: Combined attributes.
#     """
#     attr_keys = _get_unique_keys(variable_attrs)
#     attrs = {k: [] for k in attr_keys}
#     for key in attr_keys:
#         if key == 'title':
#             attrs[key] = _get_unique_attrs(variable_attrs, key)
#         elif key == 'history':
#             attrs[key] = _get_unique_attrs(variable_attrs, key)
#         elif key == 'flight_id':
#             attrs[key] = _aggregate_attrs(variable_attrs, key)
#         elif key == 'mission_id':
#             attrs[key] = _aggregate_attrs(variable_attrs, key)
#         elif key == 'storm_id':
#             # TODO: this can be misleading and should be fixed to return a
#             # single value if len=1 and a list otherwise.
#             attrs[key] = _get_unique_attrs(variable_attrs, key)[0]
#         elif key == 'date_created':
#             attrs[key] = _aggregate_attrs(variable_attrs, key)
#         elif key == 'time_coverage_start':
#             attrs[key] = _attrs_to_datetime(variable_attrs, key)[0].isoformat()
#         elif key == 'time_coverage_end':
#             attrs[key] = _attrs_to_datetime(variable_attrs, key)[-1].isoformat()
#         else:
#             attrs[key] = _get_unique_attrs(variable_attrs, key)
#     return attrs


# def _get_unique_keys(variable_attrs):
#     """ Return unique keys from a set of attributes """
#     # return [key for key in {key:None for attrs in variable_attrs for key in attrs}]
#     return list({key: None for attrs in variable_attrs for key in attrs})


# def _get_unique_attrs(variable_attrs, key) -> List:
#     """ Return unique values from a set of attributes """
#     all_attrs = _aggregate_attrs(variable_attrs, key)
#     return list(np.unique(all_attrs))  # TODO: try replacing with built-in set


# def _aggregate_attrs(variable_attrs, key) -> List:
#     """ Aggregate all attributes into a list """
#     return [attrs[key] for attrs in variable_attrs if key in attrs.keys()]


# def _attrs_to_datetime(variable_attrs, key) -> List:
#     """ Convert date-like attributes to datetimes """
#     all_attrs = _aggregate_attrs(variable_attrs, key)
#     attrs_as_datetimes = np.sort(pd.to_datetime(all_attrs))
#     return list(attrs_as_datetimes)
