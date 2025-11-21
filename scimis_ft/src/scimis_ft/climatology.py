from __future__ import annotations

import xarray as xr


def calculate_weekly_climatology(data_array: xr.DataArray) -> xr.DataArray:
    """
    Compute ISO-week climatology (mean per week) for a time-indexed DataArray.

    Parameters
    ----------
    data_array:
        Input array with a ``time`` coordinate containing datetime-like values.

    Returns
    -------
    xarray.DataArray
        DataArray reduced over time dimension with a new ``week`` coordinate.
    """
    if "time" not in data_array.coords:
        raise ValueError("Input DataArray must include a 'time' coordinate.")

    try:
        iso_week_numbers = data_array.time.dt.isocalendar().week
    except AttributeError as exc:
        raise TypeError(
            "The 'time' coordinate must be datetime-like and support .dt accessors."
        ) from exc

    weekly = data_array.groupby(iso_week_numbers).mean(dim="time")
    weekly.attrs.update(data_array.attrs)

    if "weekofyear" in weekly.coords:
        weekly = weekly.rename({"weekofyear": "week"})

    return weekly

