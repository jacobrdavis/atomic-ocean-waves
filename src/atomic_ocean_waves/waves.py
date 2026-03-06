"""
Ocean wave functions.
"""

__all__ = [
    "significant_wave_height",
    "energy_period",
    "mean_square_slope",
    "moment_weighted_mean",
    "spectral_moment",
    "intrinsic_dispersion",
    "phase_velocity",
    "group_to_phase_ratio",
    "depth_regime",
    "intrinsic_group_velocity",
    "dispersion_solver",
    "frequency_to_angular_frequency",
    "deep_water_dispersion",
    "trig_to_met",
    "wn_energy_to_fq_energy",
]

from typing import Optional, Union, Tuple

import numpy as np
from scipy.optimize import newton

GRAVITY = 9.81
TWO_PI = 2 * np.pi

# TODO: need to move _mask_spectra down to spectral_moment level but
# keep as explicit arguments in higher level functions.
def significant_wave_height(
    energy_density: np.ndarray,
    frequency: np.ndarray,
    min_frequency: Optional[float] = None,
    max_frequency: Optional[float] = None,
) -> Union[float, np.ndarray]:
    """
    Calculate significant wave height as four times the square root of the
    spectral variance.

    Note:
        This function requires that the frequency dimension is along the last
        axis of `energy_density`.

    Args:
        energy_density (np.ndarray): 1-D energy density frequency spectrum with
            shape (f,) or (..., f).
        frequency (np.ndarray): 1-D frequencies with shape (f,).
        min_frequency (float, optional): lower frequency bound.
        max_frequency (float, optional): upper frequency bound.

    Returns:
    Significant wave height as a
        float: if the shape of `energy_density` is (f,).
        np.ndarray: if the shape of `energy_density` is (..., f).

    """
    # Trim spectra to specified frequency range.
    energy_density, frequency = _mask_spectra(energy_density,
                                              frequency,
                                              min_frequency,
                                              max_frequency)

    zeroth_moment = spectral_moment(energy_density=energy_density,
                                    frequency=frequency,
                                    n=0,
                                    axis=-1)

    return 4 * np.sqrt(zeroth_moment)


def energy_period(
    energy_density: np.ndarray,
    frequency: np.ndarray,
    return_as_frequency: bool = False,
    min_frequency: Optional[float] = None,
    max_frequency: Optional[float] = None,
) -> Union[float, np.ndarray]:
    """
    Calculate energy-weighted frequency as the ratio of the first and zeroth
    moments of the one-dimensional frequency spectrum.

    Note:
        This function requires that the frequency dimension is along the last
        axis of `energy_density`.

    Args:
        energy_density (np.ndarray): 1-D energy density frequency spectrum with
            shape (f,) or (..., f).
        frequency (np.ndarray): 1-D frequencies with shape (f,).
        return_as_frequency (bool): if True, return frequency in Hz.
        min_frequency (float, optional): lower frequency bound.
        max_frequency (float, optional): upper frequency bound.

    Returns:
    Energy-weighted period as a
        float: if the shape of `energy_density` is (f,).
        np.ndarray: if the shape of `energy_density` is (..., f).

    """
    # Trim spectra to specified frequency range.
    energy_density, frequency = _mask_spectra(energy_density,
                                              frequency,
                                              min_frequency,
                                              max_frequency)

    # Ratio of the 1st and 0th moments is equivalent to 0th moment-
    # weighted frequency.
    energy_frequency = moment_weighted_mean(arr=frequency,
                                            energy_density=energy_density,
                                            frequency=frequency,
                                            n=0,
                                            axis=-1)
    if return_as_frequency:
        return energy_frequency
    else:
        return energy_frequency**(-1)


def mean_square_slope(
    energy_density: np.ndarray,
    frequency: np.ndarray,
    min_frequency: Optional[float] = None,
    max_frequency: Optional[float] = None,
) -> Union[float, np.ndarray]:
    """
    Calculate spectral mean square slope as the fourth moment of the one-
    dimensional frequency spectrum.

    Note:
        This function requires that the frequency dimension is along the last
        axis of `energy_density`.

    Args:
        energy_density (np.ndarray): 1-D energy density frequency spectrum with
            shape (f,) or (..., f).
        frequency (np.ndarray): 1-D frequencies with shape (f,).
        min_frequency (float, optional): lower frequency bound.
        max_frequency (float, optional): upper frequency bound.

    Returns:
    Mean square slope as a
        float: if the shape of `energy_density` is (f,).
        np.ndarray: if the shape of `energy_density` is (..., f).
    """
    # Trim spectra to specified frequency range.
    energy_density, frequency = _mask_spectra(energy_density,
                                              frequency,
                                              min_frequency,
                                              max_frequency)

    # Calculate the fourth moment of the energy density spectrum.
    fourth_moment = spectral_moment(energy_density=energy_density,
                                    frequency=frequency,
                                    n=4,
                                    axis=-1)
    return (TWO_PI**4 * fourth_moment) / (GRAVITY**2)


def direction(a1: np.ndarray, b1: np.ndarray) -> np.ndarray:
    """ Return the frequency-dependent direction from the directional moments.

    Calculate the direction at each frequency from the first two Fourier
    coefficients of the directional spectrum (see Sofar and Kuik et al.).

    Note: If a1 and b1 are defined (+W) and (+S), respectively, multiply
    inputs by -1 to convert to (+E) and (+N). (Equivalent to adding 180
    deg to the result.) This can arise due to differences in
    cross-spectra ordering.

    References:
        Sofar (n.d.) Spotter Technical Reference Manual

        A J Kuik, G P Van Vledder, and L H Holthuijsen (1988) "A method for the
        routine analysis of pitch-and-roll buoy wave data" JPO, 18(7), 1020-
        1034, 1988.

    Args:
        a1 (np.ndarray): Normalized spectral directional moment (+E).
        b1 (np.ndarray): Normalized spectral directional moment (+N).

    Returns:
        np.ndarray: Direction at each spectral frequency in the metereological
            convention (degrees clockwise from North).
    """
    return (90 - np.rad2deg(np.arctan2(b1, a1))) % 360


def directional_spread(a1: np.ndarray, b1: np.ndarray) -> np.ndarray:
    """ Return the frequency-dependent directional spread from the moments.

    Calculate the direction at each frequency from the first two Fourier
    coefficients of the directional spectrum (see Sofar and Kuik et al.).

    References:
        Sofar (n.d.) Spotter Technical Reference Manual

        A J Kuik, G P Van Vledder, and L H Holthuijsen (1988) "A method for the
        routine analysis of pitch-and-roll buoy wave data" JPO, 18(7), 1020-
        1034, 1988.

    Args:
        a1 (np.ndarray): Normalized spectral directional moment (+E).
        b1 (np.ndarray): Normalized spectral directional moment (+N).

    Returns:
        np.ndarray: Directional spread at each spectral frequency in degrees.
    """
    directional_spread_rad = np.sqrt(2 * (1 - np.sqrt(a1**2 + b1**2)))
    return np.rad2deg(directional_spread_rad)


def moment_weighted_mean(
    arr: np.ndarray,
    energy_density: np.ndarray,
    frequency: np.ndarray,
    n: float,
    axis: int = -1,
) -> Union[float, np.ndarray]:
    """ Compute the 'nth' moment-weighted mean of an array.

    Note:
        The `energy_density` and `arr` arrays must have the same shape.

    Args:
        arr (np.ndarray): Array to calculate the moment-weighted mean of.
        energy_density (np.ndarray): 1-D energy density frequency spectrum.
        frequency (np.ndarray): 1-D frequencies.
        n (float): Moment order (e.g., `n=1` is returns the first moment).
        axis (int, optional): Axis to calculate the moment along. Defaults
            to -1.

    Returns:
    The result of weighting `arr` by the nth moment as a
        float: if the shape of `energy_density` is (f,).
        np.ndarray: if `energy_density` has more than one dimension.  The shape
            of the returned array is reduced along `axis`.
    """
    moment_n = spectral_moment(energy_density=energy_density,
                               frequency=frequency,
                               n=n,
                               axis=axis)

    weighted_moment_n = spectral_moment(energy_density=energy_density * arr,
                                        frequency=frequency,
                                        n=n,
                                        axis=axis)
    return weighted_moment_n / moment_n


def spectral_moment(
    energy_density: np.ndarray,
    frequency: np.ndarray,
    n: float,
    axis: int = -1,
) -> Union[float, np.ndarray]:
    """ Compute the 'nth' spectral moment.

    Args:
        energy_density (np.ndarray): 1-D energy density frequency spectrum.
        frequency (np.ndarray): 1-D frequencies.
        n (float): Moment order (e.g., `n=1` is returns the first moment).
        axis (int, optional): Axis to calculate the moment along. Defaults
            to -1.

    Returns:
    The nth moment as a
        float: if the shape of `energy_density` is (f,).
        np.ndarray: if `energy_density` has more than one dimension.  The shape
            of the returned array is reduced along `axis`.
    """
    frequency_n = frequency ** n
    moment_n = np.trapezoid(energy_density * frequency_n, x=frequency, axis=axis)
    return moment_n


def intrinsic_dispersion(wavenumber, depth=np.inf):
    GRAVITY = 9.81
    gk = GRAVITY * wavenumber
    kh = wavenumber * depth
    return np.sqrt(gk * np.tanh(kh))  # angular frequency


def phase_velocity(wavenumber, depth=np.inf):
    return intrinsic_dispersion(wavenumber, depth) / wavenumber


def group_to_phase_ratio(
    wavenumber: np.ndarray,
    depth: float = np.inf,
) -> np.ndarray:
    """ Compute the ratio of group velocity to phase velocity.

    Note: to prevent overflows in `np.sinh`, the product of wavenumber and
    depth (relative depth) are used to assign ratios at deep or shallow limits:

        shallow:  Cg = 1.0 if kh < np.pi/10 (h < L/20)
           deep:  Cg = 0.5 if kh > np.pi    (h > L/2)

    Args:
        wavenumber (np.ndarray): of shape (k,) containing wavenumbers
        depth (float, optional): positive water depth. Defaults to np.inf.

    Returns:
        np.ndarray: of shape (k,) containing ratio at each wavenumber.
    """
    kh = wavenumber * depth
    in_deep, in_shallow, in_intermd = depth_regime(kh)
    ratio = np.empty(kh.shape)
    ratio[in_deep] = 0.5
    ratio[in_shallow] = 1.0
    ratio[in_intermd] = 0.5 + kh[in_intermd] / np.sinh(2 * kh[in_intermd])
    return ratio


def depth_regime(kh: np.ndarray) -> Tuple:
    """ Classify depth regime based on relative depth.

    Classify depth regime based on relative depth (product of wavenumber
    and depth) using the shallow and deep limits:

        shallow:  kh < np.pi/10 (h < L/20)
           deep:  kh > np.pi    (h > L/2)

    The depth regime is classified as intermediate if not at the deep or
    shallow limits.

    Args:
        kh (np.ndarray): relative depth of shape (k, )

    Returns:
        np.ndarray[bool]: true where kh is deep, false otherwise
        np.ndarray[bool]: true where kh is shallow, false otherwise
        np.ndarray[bool]: true where kh is intermediate, false otherwise
    """
    in_deep = kh > np.pi
    in_shallow = kh < np.pi/10
    in_intermd = np.logical_and(~in_deep, ~in_shallow)
    return in_deep, in_shallow, in_intermd


def intrinsic_group_velocity(wavenumber, depth=np.inf):
    ratio = group_to_phase_ratio(wavenumber, depth)
    return ratio * phase_velocity(wavenumber, depth)


def dispersion_solver(  #TODO: inverse dispersion?
    frequency: np.ndarray,
    depth: Union[float, np.ndarray],
) -> np.ndarray:
    r"""Solve the linear dispersion relationship.

    Solves the linear dispersion relationship w^2 = gk tanh(kh) using a
    Scipy Newton-Raphson root-finding implementation.

    Note:
        Expects input as numpy.ndarrays of shape (d,f) where f is the number
        of frequencies and d is the number of depths. The input `frequency` is
        the frequency in Hz and NOT the angular frequency, omega or w.

    Args:
        frequency (np.ndarray): of shape (d,f) containing frequencies in [Hz].
        depth (np.ndarray): of shape (d,f) containing water depths.

    Returns:
        np.ndarray: of shape (d,f) containing wavenumbers.
    """

    angular_frequency = frequency_to_angular_frequency(frequency)

    wavenumber_deep = deep_water_dispersion(frequency)

    wavenumber = newton(func=_dispersion_root,
                        x0=wavenumber_deep,
                        args=(angular_frequency, depth),
                        fprime=_dispersion_derivative)
    return np.asarray(wavenumber)


def _dispersion_root(wavenumber, angular_frequency, depth):
    gk = GRAVITY * wavenumber
    kh = wavenumber * depth
    return gk * np.tanh(kh) - angular_frequency**2


def _dispersion_derivative(wavenumber, angular_frequency, depth):
    gk = GRAVITY * wavenumber
    kh = wavenumber * depth
    return GRAVITY * np.tanh(kh) + gk * depth * (1 - np.tanh(kh)**2)


def frequency_to_angular_frequency(frequency):
    """Helper function to convert frequency (f) to angular frequency (omega)"""
    return 2 * np.pi * frequency


def deep_water_dispersion(frequency):
    """Computes wavenumber from the deep water linear dispersion relationship.

    Given frequencies (in Hz) solve the linear dispersion relationship in the
    deep water limit for the corresponding wavenumbers, k. The linear
    dispersion relationship in the deep water limit, tanh(kh) -> 1, has the
    closed form solution k = omega^2 / g and is (approximately) valid for
    kh > np.pi (h > L/2).

    Args:
        frequency (np.ndarray): of any shape containing frequencies
            in [Hz]. NOT the angular frequency, omega or w.

    Returns:
        np.ndarray: (of shape equal to the input shape) containing wavenumbers.
    """
    angular_frequency = frequency_to_angular_frequency(frequency)
    return angular_frequency**2 / GRAVITY


def trig_to_met(  # TODO: broken!
    angle_trig: np.ndarray,
    degrees=False
) -> np.ndarray:
    """Convert an angle from the trigonometric to meterological convention.

    Convert from a trignometric angle convention, defined as counterclockwise
    positive with 0 aligned with the Cartesian x-axis, to the meterological
    convention, clockwise positive with 0 aligned with the Cartesian y-axis
    (which represents North).

    Args:
        angle_trig (np.ndarray): angles in the trigonometric convention.
        degrees (bool, optional): True if input angle is in degrees. Defaults
            to False.

    Returns:
        np.ndarray: angle in meterological convention with equivalent units
            to the input `angle_trig`.
    """
    if degrees:
        offset = 90
        modulus = 360
        # angle_met = (-angle_trig + 90) % 360
    else:
        offset = np.pi/2
        modulus = 2*np.pi
        # angle_met = (-angle_trig + np.pi/2) % (2*np.pi)
    return (-angle_trig + offset) % modulus


def wn_energy_to_fq_energy(
    energy_density_wavenumber: np.ndarray,
    wavenumber: np.ndarray,
    depth: float = np.inf,
) -> np.ndarray:
    """ Transform energy density from wavenumber space to frequency space.

    Transform energy density, defined on a 2-D wavenumber grid, to energy
    density on a frequency-direction grid using the appropriate Jacobian.
    The calculation follows that of the WaMoS II processing in [1]:

    E(w, theta) = E(kx, ky) k dk/dw

    and

    E(f, theta) = E(w, theta) * 2 pi

    Where E(w, theta) is the energy density as a function of angular frequency
    (w) and direction (theta), E(kx, ky) is the energy density as a function of
    the east wavenumber (kx) and north wavenumber (kx), k is the
    scalar wavenumber computed as magnitude{kx, ky}, and k dk/dw comprise the
    Jacobian with dk/dw equal to the inverse of the group velocity
    (i.e., 1/Cg). The final result is converted from angular frequency to
    frequency (f) such that E(f, theta) is returned. This transformation
    assumes linear wave dispersion.

    References:
        1. Stephen F. Barstow, Jean-Raymond Bidlot, Sofia Caires, Mark A.
        Donelan, William M. Drennan, et al.. Measuring and Analysing the
        directional spectrum of ocean waves. D. Hauser, K. Kahma, H. Krogstad,
        S. Monbaliu, S. Lehner et L. Wyatt. COST Office, pp.465, 2005,
        COST 714; EUR 21367. ffhal-00529755f

    Args:
        energy_density_wavenumber (np.ndarray): Energy density in 2-D
            wavenumber space with shape (x, y).
        wavenumber (np.ndarray): Wavenumber magnitudes with shape (x, y).
        depth (float, optional): Positive depth. Defaults to np.inf.

    Returns:
        np.ndarray: energy density in frequency-direction space, E(f, theta),
            with shape (x, y).
    """
    dk_dw = 1 / intrinsic_group_velocity(wavenumber, depth)
    jacobian = wavenumber * dk_dw
    return energy_density_wavenumber * jacobian[:, :, None] * 2*np.pi  # [:, :, None]

# For testing:
# w = intrinsic_dispersion(1, 1000)
# cp = phase_velocity(1, 1000)
# cg = intrinsic_group_velocity(1, 1000)
# print(w)
# print(cp)
# print(cg)

# wavenumber = 0.5
# group_velocity = intrinsic_group_velocity(1, 1000)
# jacobian = 1 / group_velocity


# TODO: name implies mask with NaN, but values are dropped
def _mask_spectra(
    spectra: np.ndarray,
    coordinate: np.ndarray,
    min_coordinate: Optional[float] = None,
    max_coordinate: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """ Return 1D spectra between a min and max coordinate.

    Note: For input spectra with shape `(..., f)` and coordinate with
    shape (f), values where `coordinate` is outside the closed interval
    [`min_coordinate`, `max_coordinate`] are dropped such that output
    has shape `(..., f')` and (f'), respectively, where `f'` is the
    length of the coordinate inside the interval.  The `f` axis must be
    the last axis of `spectra`.

    Args:
        spectra (np.ndarray): spectra with shape (..., f).
        coordinate (np.ndarray): spectral coordinate (e.g., frequency
            or wavenumber) with shape  (f).
        min_coordinate (Optional[float], optional): maximum coordinate
            value to mask. Defaults to None.
        max_coordinate (Optional[float], optional): minimum coordinate
            value to mask. Defaults to None.

    Returns:
        _type_: spectra and coordinate between min_coordinate and
            max_coordinate with shapes (..., f') and (f'), respectively.
    """
    # If a coordinate has ndim > 1, the masked spectra will be flattened
    # (since the result might be ragged) which could result in
    # unexpected behavior when passed to other functions.
    if coordinate.ndim > 1:
        raise ValueError('Spectral `coordinate` must be 1-dimensional.')
    if min_coordinate is None:
        min_coordinate = coordinate.min()
    if max_coordinate is None:
        max_coordinate = coordinate.max()

    # Mask spectra and coordinate outside of the specified range.
    coordinate_mask = np.logical_and(coordinate >= min_coordinate,
                                     coordinate <= max_coordinate)

    return spectra[..., coordinate_mask], coordinate[coordinate_mask]
