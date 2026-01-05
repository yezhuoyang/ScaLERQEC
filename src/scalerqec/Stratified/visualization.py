"""
Common visualization functions for S-curve analysis and LER estimation.

This module provides flexible, reusable plotting functions that can handle
various circuit sizes, error rates, and parameter configurations.
"""

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from .ScurveModel import modified_linear_function_with_d, modified_sigmoid_function


def _format_sample_count(count: int) -> str:
    """Format sample count in human-readable form (K, M format)."""
    if count >= 1e6:
        return f"{count / 1e6:.1f}M"
    elif count >= 1e3:
        return f"{count / 1e3:.1f}K"
    else:
        return str(count)


def plot_log_scurve(
    x_list: List[int],
    y_list: List[float],
    sigma_list: List[float],
    sample_cost_list: List[int],
    # S-curve parameters
    a: float,
    b: float,
    c: float,
    t: int,
    # Region bounds
    minw: int,
    maxw: int,
    saturatew: int,
    sweet_spot: int,
    has_logical_errorw: int,
    # Circuit info
    num_noise: int,
    num_detector: int,
    error_rate: float,
    code_distance: int,
    # Quality metrics
    r_squared: float,
    ler: float = 0.0,
    time_elapsed: Optional[float] = None,
    total_samples: Optional[int] = None,
    # Plot configuration
    filename: Optional[str] = None,
    title: Optional[str] = None,
    k_range: int = 5,
    show_annotations: bool = True,
    figsize: Tuple[float, float] = (10, 6),
    # Hyperparameters for info box (optional)
    min_le_event: int = 100,
    max_sample_gap: int = 5000,
    max_subspace_sample: int = 50000,
    ratio: float = 0.05,
) -> plt.Figure:
    """
    Create a visualization of the log-S curve fitting results.

    Focus on the critical region with positive y-values only.
    """
    if not x_list:
        x_list = [t + 1]
        y_list = [0.0]
        sigma_list = [0.0]
        sample_cost_list = [0]

    # Filter to only positive y values for display
    positive_indices = [i for i, y in enumerate(y_list) if y > 0]
    if not positive_indices:
        # If no positive values, use all data
        positive_indices = list(range(len(x_list)))

    # Compute alpha
    alpha = -1.0 / a if a != 0 else 0.0

    # Create figure with two subplots: main plot and info panel
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 1], wspace=0.05)
    ax = fig.add_subplot(gs[0])
    ax_info = fig.add_subplot(gs[1])
    ax_info.axis("off")

    # Determine x-axis range: focus on the data region
    x_min_data = min(x_list)
    x_max_data = max(x_list)

    # X-axis: start slightly before first data point, end after saturatew
    x_plot_min = max(0, x_min_data - 2)
    x_plot_max = saturatew + (saturatew - t) * 0.5  # Extend past saturation a bit

    # Y-axis: start at 0, extend above max positive value
    max_positive_y = (
        max([y_list[i] for i in positive_indices]) if positive_indices else 1.0
    )
    y_plot_max = max_positive_y * 1.15

    # x-range for fitted curve
    x_fit = np.linspace(t + 0.5, x_plot_max, 500)
    y_fit = modified_linear_function_with_d(x_fit, a, b, c, t)

    # Clip fitted curve to positive values for display
    y_fit_clipped = np.maximum(y_fit, 0)

    # Sweet spot y-value
    sweet_spot_y = modified_linear_function_with_d(sweet_spot, a, b, c, t)

    # Region shading (draw first so they're behind everything)
    # Fault-tolerant region (green) - w <= t
    ax.axvspan(
        x_plot_min, t, color="green", alpha=0.15, label=f"Fault-tolerant (w ≤ t)"
    )

    # Critical region (yellow/gold) - minw to maxw
    ax.axvspan(
        minw, maxw, color="gold", alpha=0.20, label=f"Critical region ({k_range}σ)"
    )

    # Saturation line (red dotted)
    ax.axvline(
        saturatew,
        color="red",
        linestyle=":",
        linewidth=1.5,
        alpha=0.8,
        label=f"Saturation (w={saturatew})",
    )

    # minw/maxw vertical lines (dashed)
    ax.axvline(minw, color="orange", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.axvline(maxw, color="orange", linestyle="--", linewidth=1.2, alpha=0.7)

    # Data bars (blue, only for positive values)
    bar_x = []
    bar_y = []
    bar_sigma = []
    bar_samples = []
    for i, (x, y, s, samp) in enumerate(
        zip(x_list, y_list, sigma_list, sample_cost_list)
    ):
        if y > 0:
            bar_x.append(x)
            bar_y.append(y)
            bar_sigma.append(s)
            bar_samples.append(samp)

    if bar_x:
        # Determine bar width based on data spacing
        if len(bar_x) >= 2:
            avg_spacing = (max(bar_x) - min(bar_x)) / max(1, len(bar_x) - 1)
            bar_width = max(0.4, min(0.8, avg_spacing * 0.7))
        else:
            bar_width = 0.6

        ax.bar(
            bar_x,
            bar_y,
            width=bar_width,
            align="center",
            color="steelblue",
            edgecolor="steelblue",
            alpha=0.8,
            label="Data (log-transformed)",
        )

        # Error bars
        ax.errorbar(
            bar_x,
            bar_y,
            yerr=bar_sigma,
            fmt="none",
            color="black",
            capsize=3,
            elinewidth=1,
            label="Error bars (σ)",
        )

    # Fitted curve (red, solid)
    ax.plot(
        x_fit,
        y_fit_clipped,
        label=f"Fitted curve (R²={r_squared:.4f})",
        color="crimson",
        linestyle="-",
        linewidth=2,
    )

    # Sweet spot marker (purple star)
    if sweet_spot_y > 0:
        ax.scatter(
            sweet_spot,
            sweet_spot_y,
            color="purple",
            marker="*",
            s=200,
            zorder=5,
            label=f"Sweet spot (w={sweet_spot})",
        )

    # Sample cost annotations
    if show_annotations and bar_x:
        # Annotate a subset of points to avoid clutter
        num_annotations = min(6, len(bar_x))
        indices = np.linspace(0, len(bar_x) - 1, num=num_annotations, dtype=int)
        indices = sorted(set(indices))

        for idx, i in enumerate(indices):
            x, y, s = bar_x[i], bar_y[i], bar_samples[i]
            if s > 0:
                label_text = _format_sample_count(s)
                # Alternate annotation positions
                y_offset = (
                    0.05 * max_positive_y if idx % 2 == 0 else -0.08 * max_positive_y
                )
                va = "bottom" if y_offset > 0 else "top"

                ax.annotate(
                    label_text,
                    (x, y),
                    textcoords="offset points",
                    xytext=(0, 8 if y_offset > 0 else -12),
                    ha="center",
                    va=va,
                    fontsize=8,
                    color="gray",
                )

    # Set axis limits
    ax.set_xlim(x_plot_min, x_plot_max)
    ax.set_ylim(0, y_plot_max)

    # Axis labels and title
    ax.set_xlabel("Weight (w)", fontsize=11)
    ax.set_ylabel(r"$\log\left(\frac{0.5}{P_L(w)} - 1\right)$", fontsize=11)

    if title is not None:
        ax.set_title(title, fontsize=12)
    else:
        ax.set_title(
            f"Log-S Curve Fit (d={code_distance}, p={error_rate:.2e})", fontsize=12
        )

    # Legend
    ax.legend(fontsize=8, loc="upper right", framealpha=0.9)

    # Grid
    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)

    # Integer x-axis ticks
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # Information panel on the right
    if total_samples is None:
        total_samples = sum(sample_cost_list)

    info_lines = [
        "=== Parameters ===",
        f"α = {alpha:.4f}",
        f"a = {a:.4f}",
        f"b = {b:.4f}",
        f"c (β) = {c:.4f}",
        "",
        "=== Weights ===",
        f"t = {t}",
        f"w_sweet = {sweet_spot}",
        f"w_min = {minw}",
        f"w_max = {maxw}",
        f"w_sat = {saturatew}",
        "",
        "=== Circuit ===",
        f"#noise = {num_noise}",
        f"#detector = {num_detector}",
        "",
        "=== Quality ===",
        f"R² = {r_squared:.4f}",
    ]

    if ler > 0:
        info_lines.append(f"P_L = {ler:.3e}")

    if time_elapsed is not None:
        info_lines.append(f"Time = {time_elapsed:.1f}s")

    if total_samples > 0:
        info_lines.append(f"Samples = {_format_sample_count(total_samples)}")

    info_text = "\n".join(info_lines)
    ax_info.text(
        0.1,
        0.5,
        info_text,
        fontsize=9,
        va="center",
        ha="left",
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="whitesmoke", alpha=0.9),
        transform=ax_info.transAxes,
    )

    plt.tight_layout()

    if filename is not None:
        fig.savefig(filename, format="pdf", bbox_inches="tight", dpi=150)

    return fig


def plot_scurve(
    estimated_subspaceLER: Dict[int, float],
    subspace_sample_used: Dict[int, int],
    subspace_LE_count: Dict[int, int],
    # S-curve parameters
    a: float,
    b: float,
    c: float,
    t: int,
    # Region bounds
    minw: int,
    maxw: int,
    saturatew: int,
    # Result info
    ler: float,
    # Plot configuration
    filename: Optional[str] = None,
    title: Optional[str] = None,
    savefigure: bool = False,
    figsize: Tuple[float, float] = (8, 5),
) -> plt.Figure:
    """
    Plot the S-curve (not log-transformed) showing subspace logical error rates.
    """
    from .ScurveModel import subspace_sigma_estimator

    keys = list(estimated_subspaceLER.keys())
    values = [estimated_subspaceLER[k] for k in keys]
    sigma_list = [
        subspace_sigma_estimator(subspace_sample_used[k], subspace_LE_count[k])
        for k in keys
    ]

    fig, ax = plt.subplots(figsize=figsize)

    # Data bars
    ax.bar(
        keys,
        values,
        color="steelblue",
        alpha=0.8,
        label="Estimated subspace LER",
    )

    # Error bars
    ax.errorbar(
        keys,
        values,
        yerr=sigma_list,
        fmt="none",
        ecolor="black",
        capsize=3,
        elinewidth=1,
        label="Error bars",
    )

    # Smooth S-curve
    if len(keys) > 0:
        x = np.linspace(t + 0.1, saturatew + 5, 500)
        y = modified_sigmoid_function(x, a, b, c, t)
        ax.plot(
            x,
            y,
            color="crimson",
            linewidth=2.0,
            label="Fitted S-curve",
            linestyle="-",
        )

    # Region shading
    max_val = max(values) if values else 0.5

    # Fault-tolerant region
    ax.axvspan(0, t, color="green", alpha=0.15, label="Fault-tolerant")

    # Critical region
    ax.axvspan(minw, maxw, color="gold", alpha=0.20, label="Critical region")

    # Saturation line
    ax.axvline(
        saturatew,
        color="red",
        linestyle=":",
        linewidth=1.5,
        alpha=0.8,
        label=f"Saturation (w={saturatew})",
    )

    # Labels and formatting
    ax.set_xlabel("Weight")
    ax.set_ylabel("Logical Error Rate in subspace")

    if title is not None:
        ax.set_title(f"S-curve of {title} (PL={ler:.2e})")
    else:
        ax.set_title(f"S-curve (PL={ler:.2e})")

    ax.legend(fontsize=8)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.set_ylim(0, None)  # Start y-axis at 0

    plt.tight_layout()

    if savefigure and filename is not None:
        fig.savefig(filename + ".pdf", format="pdf", bbox_inches="tight")

    return fig


def plot_linear_fit(
    x_list: List[int],
    y_list: List[float],
    a: float,
    b: float,
    minw: int,
    maxw: int,
    filename: Optional[str] = None,
    savefigure: bool = True,
) -> plt.Figure:
    """
    Plot the linear fit of the S-curve (initial fitting step).
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    # Filter to positive values
    pos_x = [x for x, y in zip(x_list, y_list) if y > 0]
    pos_y = [y for y in y_list if y > 0]

    if not pos_x:
        pos_x, pos_y = x_list, y_list

    # Data points
    ax.scatter(pos_x, pos_y, label="Data points", color="steelblue", s=50)

    # Fitted line
    x_fit = np.linspace(min(pos_x), max(pos_x), 100)
    y_fit = a * x_fit + b
    y_fit_clipped = np.maximum(y_fit, 0)
    ax.plot(x_fit, y_fit_clipped, label="Fitted line", color="crimson", linewidth=2)

    # Critical region
    ax.axvspan(minw, maxw, color="gold", alpha=0.2, label="Critical region")

    # Parameter annotations
    alpha = -1 / a if a != 0 else 0
    mu = alpha * b
    textstr = "\n".join((rf"$\alpha={alpha:.4f}$", rf"$\mu={mu:.4f}$"))
    ax.text(
        0.05,
        0.95,
        textstr,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="black"),
    )

    ax.set_xlabel("Weight")
    ax.set_ylabel("Linear")
    ax.set_title("Linear Fit of S-curve")
    ax.legend()
    ax.set_ylim(0, None)

    plt.tight_layout()

    if savefigure and filename is not None:
        fig.savefig(filename, format="pdf", dpi=300)

    return fig
