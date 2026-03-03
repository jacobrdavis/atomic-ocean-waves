import xarray as xr


def apply_frequency_reduction_ufunc(
    *args,
    frequency_dim: str = 'frequency',
    **kwargs,
):
    # Determine the number of frequency-dependent input arrays from
    # *args. Subtract 1 since the first input to apply_ufunc is always
    # the function to apply.
    n_arrays = len(args) - 1
    return xr.apply_ufunc(
        *args,
        input_core_dims=[[frequency_dim]] * n_arrays,
        output_core_dims=[[]],
        **kwargs,
    )