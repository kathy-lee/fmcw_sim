"""Minimal quick-look plots for extracted cubes (standalone; no layered_rd_view).

Each helper takes the extract_cubes dict and indexes its precomputed `axes`,
the same convention streaming_subscriber_mimo.py uses (plot from the axis arrays).
Returns the matplotlib Figure so callers can save or show it.
"""
from __future__ import annotations

import numpy as np


def show_adc(cubes, channel=0):
    """ADC magnitude for one virtual channel (chirp x range-sample)."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.imshow(np.abs(cubes["adc"][channel]), aspect="auto", origin="lower",
              cmap="viridis")
    ax.set(xlabel="range sample", ylabel="chirp",
           title=f"ADC magnitude (channel {channel})")
    return fig


def show_rd_map(cubes):
    """Integrated power map (range x Doppler), axes in physical units."""
    import matplotlib.pyplot as plt
    power = cubes["power"]
    ra = cubes["axes"]["range_m"]
    va = cubes["axes"]["velocity_mps"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.imshow(10.0 * np.log10(power + 1e-12), aspect="auto", origin="lower",
              cmap="viridis",
              extent=[ra[0], ra[-1], va[0], va[-1]])
    ax.set(xlabel="range [m]", ylabel="velocity [m/s]", title="RD power [dB]")
    return fig


def show_rad_slice(cubes, doppler_bin):
    """Range-azimuth image of the RAD tensor at one Doppler bin."""
    import matplotlib.pyplot as plt
    rad = cubes["rad"]
    ra = cubes["axes"]["range_m"]
    az = cubes["axes"]["azimuth_deg"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.imshow(10.0 * np.log10(rad[:, doppler_bin, :] + 1e-12), aspect="auto",
              origin="lower", cmap="viridis",
              extent=[ra[0], ra[-1], az[0], az[-1]])
    ax.set(xlabel="range [m]", ylabel="azimuth [deg]",
           title=f"RAD slice (Doppler bin {doppler_bin})")
    return fig
