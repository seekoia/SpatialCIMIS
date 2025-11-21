from __future__ import annotations

import numpy as np
import xarray as xr
import rioxarray  # noqa: F401


def _ensure_week_dimension(data: xr.DataArray) -> tuple[xr.DataArray, str]:
    if "week" in data.dims:
        return data, "week"
    if "weekofyear" in data.dims:
        return data.rename({"weekofyear": "week"}), "week"
    raise ValueError("Expected a 'week' or 'weekofyear' dimension for Fourier analysis.")


def fourier_transform_climatology_components(
    weekly_climatology: xr.DataArray,
    n_components_to_keep: int = 3,
) -> xr.Dataset:
    """
    Apply an FFT to the weekly climatology and return amplitude/phase datasets.
    """
    if not isinstance(weekly_climatology, xr.DataArray):
        raise TypeError("weekly_climatology must be an xarray.DataArray.")

    weekly_climatology, time_dim = _ensure_week_dimension(weekly_climatology)
    n_time_points = weekly_climatology.sizes[time_dim]
    if n_time_points == 0:
        raise ValueError(f"Dimension '{time_dim}' has size 0.")

    if hasattr(weekly_climatology.data, "chunks"):
        axis_num = weekly_climatology.get_axis_num(time_dim)
        chunks = weekly_climatology.chunks[axis_num]
        if len(chunks) > 1:
            weekly_climatology = weekly_climatology.chunk({time_dim: -1})

    freq_dim = "frequency"
    fft_output_size = n_time_points // 2 + 1

    fft_coefficients = xr.apply_ufunc(
        np.fft.rfft,
        weekly_climatology,
        input_core_dims=[[time_dim]],
        output_core_dims=[[freq_dim]],
        dask="parallelized",
        output_dtypes=[np.complex128],
        dask_gufunc_kwargs={"output_sizes": {freq_dim: fft_output_size}},
        keep_attrs=False,
    )

    frequencies = np.fft.rfftfreq(n_time_points, d=1.0)
    fft_coefficients = fft_coefficients.assign_coords({freq_dim: frequencies})

    if fft_coefficients.sizes[freq_dim] < n_components_to_keep:
        raise ValueError(
            f"Requested {n_components_to_keep} components but only "
            f"{fft_coefficients.sizes[freq_dim]} are available."
        )

    selected = fft_coefficients.isel({freq_dim: slice(0, n_components_to_keep)})
    amplitude_raw = xr.DataArray(
        np.abs(selected.data), coords=selected.coords, dims=selected.dims
    )
    phase = xr.DataArray(
        np.angle(selected.data), coords=selected.coords, dims=selected.dims, name="phase"
    )

    amplitude = _scale_physical_amplitude(amplitude_raw, time_dim, n_time_points)
    amplitude.name = "amplitude"
    amplitude.attrs["long_name"] = "Scaled Fourier Component Amplitude"
    amplitude.attrs["units"] = weekly_climatology.attrs.get("units", "original units")

    phase.attrs["long_name"] = "Fourier Component Phase"
    phase.attrs["units"] = "radians"

    amplitude.coords[freq_dim].attrs.update(
        {"units": "cycles/week", "long_name": "Frequency"}
    )
    phase.coords[freq_dim].attrs.update(
        {"units": "cycles/week", "long_name": "Frequency"}
    )

    return xr.Dataset({"amplitude": amplitude, "phase": phase})


def _scale_physical_amplitude(
    amplitude_raw: xr.DataArray,
    time_dim: str,
    n_time_points: int,
) -> xr.DataArray:
    freq_dim = "frequency"
    scaled = amplitude_raw.copy(deep=True)
    nyquist_freq = 0.5 if n_time_points % 2 == 0 else None

    for freq_val in scaled.coords[freq_dim].values:
        raw_slice = amplitude_raw.sel({freq_dim: freq_val})
        if np.isclose(freq_val, 0.0):
            scaled.loc[{freq_dim: freq_val}] = raw_slice / n_time_points
        elif nyquist_freq is not None and np.isclose(freq_val, nyquist_freq):
            scaled.loc[{freq_dim: freq_val}] = raw_slice / n_time_points
        else:
            scaled.loc[{freq_dim: freq_val}] = raw_slice * (2.0 / n_time_points)

    return scaled


def reconstruct_climatology_from_fourier_components(
    fourier_components: xr.Dataset,
    original_n_time_points: int,
    *,
    original_time_dim_name: str = "week",
    original_week_coordinates: xr.DataArray | None = None,
    reference_attributes: dict | None = None,
    reference_name: str = "reconstructed_climatology",
) -> xr.DataArray:
    """
    Reconstruct a climatology from scaled Fourier amplitude and phase components.
    """
    if not isinstance(fourier_components, xr.Dataset):
        raise TypeError("fourier_components must be an xarray.Dataset.")
    if not {"amplitude", "phase"}.issubset(fourier_components.data_vars):
        raise ValueError("Dataset must contain 'amplitude' and 'phase' variables.")

    amplitude = fourier_components["amplitude"]
    phase = fourier_components["phase"]

    is_even = original_n_time_points % 2 == 0
    nyquist_val = 0.5 if is_even else None

    magnitudes = xr.zeros_like(amplitude)
    for freq_val in amplitude.frequency.values:
        amp_slice = amplitude.sel(frequency=freq_val)
        if np.isclose(freq_val, 0.0):
            coeff = amp_slice * original_n_time_points
        elif nyquist_val is not None and np.isclose(freq_val, nyquist_val):
            coeff = amp_slice * original_n_time_points
        else:
            coeff = amp_slice * original_n_time_points / 2.0
        magnitudes.loc[{"frequency": freq_val}] = coeff

    rfft_coeffs = magnitudes * np.exp(1j * phase)

    reconstructed = xr.apply_ufunc(
        np.fft.irfft,
        rfft_coeffs,
        kwargs={"n": original_n_time_points},
        input_core_dims=[["frequency"]],
        output_core_dims=[[original_time_dim_name]],
        dask="parallelized",
        output_dtypes=[amplitude.dtype],
        dask_gufunc_kwargs={"output_sizes": {original_time_dim_name: original_n_time_points}},
    )

    if original_week_coordinates is not None:
        reconstructed = reconstructed.assign_coords(
            {original_time_dim_name: (original_time_dim_name, original_week_coordinates.data)}
        )
    else:
        reconstructed = reconstructed.assign_coords(
            {original_time_dim_name: (original_time_dim_name, np.arange(original_n_time_points))}
        )

    attrs = reference_attributes.copy() if reference_attributes else {}
    attrs.setdefault(
        "description",
        f"Weekly climatology reconstructed from {rfft_coeffs.sizes['frequency']} Fourier component(s).",
    )
    reconstructed.attrs = attrs
    reconstructed.name = reference_name

    if "spatial_ref" in amplitude.coords:
        reconstructed = reconstructed.assign_coords({"spatial_ref": amplitude.spatial_ref})

    if hasattr(amplitude, "rio") and hasattr(amplitude.rio, "crs") and amplitude.rio.crs:
        try:
            reconstructed.rio.write_crs(amplitude.rio.crs, inplace=True)
        except Exception:
            pass

    return reconstructed

