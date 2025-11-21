from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import xarray as xr


def _frequency_label(freq: float) -> str:
    return f"{freq:.4f} cycles/week" if isinstance(freq, float) else str(freq)


def plot_fourier_components(
    components: xr.Dataset,
    *,
    output_path: str | Path | None = None,
    show: bool = False,
) -> Path | None:
    """Plot amplitude/phase panels for the leading three Fourier components."""

    amplitude = components["amplitude"]
    phase = components["phase"]

    freq_values = np.array(amplitude.frequency.values, dtype=float)
    if freq_values.size < 3:
        raise ValueError("At least three Fourier components are required for plotting.")

    f0, f1, f2 = freq_values[:3]

    fig = plt.figure(figsize=(16, 8))
    grid = gridspec.GridSpec(
        2,
        4,
        figure=fig,
        width_ratios=[1.2, 1.2, 1, 1],
        height_ratios=[1, 1],
        wspace=0.15,
        hspace=0.2,
    )

    ax_f0 = fig.add_subplot(grid[:, 0:2])
    _plot_imshow(
        ax_f0,
        amplitude.sel(frequency=f0),
        cmap="viridis",
        label=amplitude.attrs.get("units", "Amplitude"),
        title=f"f={f0:.4f}\nAmplitude (Component 1)",
    )

    ax_f1_amp = fig.add_subplot(grid[0, 2])
    _plot_imshow(
        ax_f1_amp,
        amplitude.sel(frequency=f1),
        cmap="viridis",
        label=amplitude.attrs.get("units", "Amplitude"),
        title=f"f={f1:.4f}\nAmplitude (Component 2)",
    )

    ax_f1_phase = fig.add_subplot(grid[0, 3])
    a1_peak_days = _phase_to_peak_day(phase.sel(frequency=f1), f1)
    _plot_imshow(
        ax_f1_phase,
        a1_peak_days,
        cmap="inferno",
        label=a1_peak_days.attrs.get("units", "Day of year"),
        title=f"f={f1:.4f}\nPeak day (Component 2)",
    )

    ax_f2_amp = fig.add_subplot(grid[1, 2])
    _plot_imshow(
        ax_f2_amp,
        amplitude.sel(frequency=f2),
        cmap="viridis",
        label=amplitude.attrs.get("units", "Amplitude"),
        title=f"f={f2:.4f}\nAmplitude (Component 3)",
    )

    ax_f2_phase = fig.add_subplot(grid[1, 3])
    a2_peak_days = _phase_to_peak_day(phase.sel(frequency=f2), f2)
    _plot_imshow(
        ax_f2_phase,
        a2_peak_days,
        cmap="inferno",
        label=a2_peak_days.attrs.get("units", "Day of year"),
        title=f"f={f2:.4f}\nPeak day (Component 3)",
    )

    fig.suptitle("Fourier Component Analysis", fontsize=16, y=0.95)

    saved_path: Path | None = None
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        saved_path = output_path

    if show:
        plt.show()
    else:
        plt.close(fig)

    return saved_path


def _phase_to_peak_day(phase: xr.DataArray, frequency: float) -> xr.DataArray:
    if frequency <= 0:
        raise ValueError("Frequency must be positive for peak day conversion.")
    peak_time_weeks = (np.mod(-phase, 2 * np.pi)) / (2 * np.pi * frequency)
    peak_day = peak_time_weeks * 7
    peak_day.attrs["units"] = "Day of year"
    peak_day.attrs["long_name"] = "Timing of Fourier component peak"
    return peak_day


def plot_climatology_timeseries(
    original: xr.DataArray,
    reconstructed: xr.DataArray,
    *,
    output_path: str | Path | None = None,
    show: bool = False,
    x_index: Optional[int] = None,
    y_index: Optional[int] = None,
) -> Path | None:
    """Plot original vs reconstructed weekly climatology for a single pixel."""

    if "week" not in original.dims:
        raise ValueError("Original climatology must include a 'week' dimension.")

    x_dim = _locate_dimension(original, ("x", "lon", "cols"))
    y_dim = _locate_dimension(original, ("y", "lat", "rows"))

    if x_dim is None or y_dim is None:
        raise ValueError("Unable to determine spatial dimensions for plotting.")

    if x_index is None:
        x_index = original.sizes[x_dim] // 2
    if y_index is None:
        y_index = original.sizes[y_dim] // 2

    orig_series = original.isel({x_dim: x_index, y_dim: y_index}).compute()
    recon_series = reconstructed.isel({x_dim: x_index, y_dim: y_index}).compute()

    x_coord = _coordinate_value(original, x_dim, x_index)
    y_coord = _coordinate_value(original, y_dim, y_index)

    weeks = orig_series["week"].values

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(weeks, orig_series.values, label="Original", linewidth=2)
    ax.plot(weeks, recon_series.values, label="Reconstructed", linewidth=2, linestyle="--")
    ax.set_xlabel("ISO Week")
    ax.set_ylabel(original.attrs.get("units", "Value"))
    ax.set_title(
        "Weekly Climatology: Original vs Reconstructed\n"
        f"Pixel {y_dim}={y_coord}, {x_dim}={x_coord}"
    )
    ax.grid(True, alpha=0.4)
    ax.legend()

    saved_path: Path | None = None
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        saved_path = output_path

    if show:
        plt.show()
    else:
        plt.close(fig)

    return saved_path


def plot_rmse_map(
    rmse: xr.DataArray,
    *,
    output_path: str | Path | None = None,
    show: bool = False,
) -> Path | None:
    """Plot spatial RMSE between original and reconstructed climatology."""

    rmse_data = rmse.compute()
    fig, ax = plt.subplots(figsize=(7, 6))
    im = rmse_data.plot.imshow(
        ax=ax,
        cmap="inferno",
        cbar_kwargs={"label": rmse.attrs.get("units", "RMSE")},
    )
    ax.set_title("Reconstruction RMSE")
    x_dim = _locate_dimension(rmse_data, ("x", "lon", "cols"))
    y_dim = _locate_dimension(rmse_data, ("y", "lat", "rows"))
    ax.set_xlabel(f"{x_dim} coordinate" if x_dim else "X coordinate")
    ax.set_ylabel(f"{y_dim} coordinate" if y_dim else "Y coordinate")
    ax.set_aspect("equal")

    saved_path: Path | None = None
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        saved_path = output_path

    if show:
        plt.show()
    else:
        plt.close(fig)

    return saved_path


def plot_component_histograms(
    components: xr.Dataset,
    *,
    output_path: str | Path | None = None,
    show: bool = False,
    bins: int = 50,
) -> Path | None:
    """Plot histograms for key Fourier component fields."""

    amplitude = components["amplitude"]
    phase = components["phase"]
    freq_values = np.array(amplitude.frequency.values, dtype=float)
    if freq_values.size < 3:
        raise ValueError("At least three Fourier components are required for histogram plotting.")

    f0, f1, f2 = freq_values[:3]
    peak_day_f1 = _phase_to_peak_day(phase.sel(frequency=f1), f1)
    peak_day_f2 = _phase_to_peak_day(phase.sel(frequency=f2), f2)

    entries = [
        (f"Component 1 amplitude (f={f0:.4f})", amplitude.sel(frequency=f0), None),
        (f"Component 2 amplitude (f={f1:.4f})", amplitude.sel(frequency=f1), None),
        (f"Component 2 peak day (f={f1:.4f})", peak_day_f1, np.linspace(0, 365, bins + 1)),
        (f"Component 3 amplitude (f={f2:.4f})", amplitude.sel(frequency=f2), None),
        (f"Component 3 peak day (f={f2:.4f})", peak_day_f2, np.linspace(0, 365, bins + 1)),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    axes = axes.flatten()

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for idx, (title, data, custom_bins) in enumerate(entries):
        ax = axes[idx]
        values = _flatten_valid(data)
        if values.size == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_title(title)
            continue
        hist_bins = custom_bins if custom_bins is not None else bins
        ax.hist(values, bins=hist_bins, color=colors[idx % len(colors)], alpha=0.85)
        ax.set_title(title)
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.3)

    axes[-1].axis("off")

    plt.tight_layout()

    saved_path: Path | None = None
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        saved_path = output_path

    if show:
        plt.show()
    else:
        plt.close(fig)

    return saved_path


def _locate_dimension(data: xr.DataArray, candidates: Sequence[str]) -> Optional[str]:
    for name in candidates:
        if name in data.dims:
            return name
    return None


def _coordinate_value(data: xr.DataArray, dim: str, index: int) -> float | int:
    if dim in data.coords:
        coord = data.coords[dim].values
        if coord.shape:
            return coord[index].item() if hasattr(coord[index], "item") else coord[index]
    return index


def _plot_imshow(
    ax: plt.Axes,
    data: xr.DataArray,
    *,
    cmap: str,
    label: str,
    title: str,
) -> None:
    data.plot.imshow(ax=ax, cmap=cmap, cbar_kwargs={"label": label})
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(labelleft=False, labelbottom=False)
    ax.set_aspect("equal")


def _flatten_valid(data: xr.DataArray) -> np.ndarray:
    arr = np.asarray(data.values).ravel()
    return arr[np.isfinite(arr)]

