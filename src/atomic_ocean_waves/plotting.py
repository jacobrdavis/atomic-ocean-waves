"""
Shared plotting functions.
"""

from typing import Optional, Union, Tuple, Sequence, Any

# import cartopy
# import cmocean
import colorcet
# import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import xarray as xr
# from cartopy.mpl.geoaxes import GeoAxes
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.axes import Axes
from matplotlib.collections import QuadMesh, PathCollection, LineCollection
from matplotlib.colors import Colormap, LinearSegmentedColormap
from matplotlib.contour import QuadContourSet, ContourSet
from matplotlib.lines import Line2D
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.patches import FancyArrow, Arc
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from atomic_ocean_waves import mission_specifications


figure_full_width = 5.5
normal_font_size = 10
small_font_size = 8


rc_params = {
    'font.size': normal_font_size,
    'font.family': 'sans-serif',
    'font.sans-serif': 'Helvetica',
    'axes.titlesize': normal_font_size,
    'axes.linewidth': 0.5,
    'axes.labelsize': normal_font_size,
    'lines.markersize': 3,
    'legend.fontsize': small_font_size,
    'xtick.labelsize': small_font_size,
    'ytick.labelsize': small_font_size,
    'figure.dpi': 300,
    'figure.figsize': (figure_full_width, 4.125),
}
plt.rcParams.update(rc_params)


def get_ax(ax):
    "Helper function to get current axes if `ax` is None."
    if ax is None:
        ax = plt.gca()
    return ax

def set_time_series_xaxis(
    ax: Axes,
    plot_time_start: pd.Timestamp,
    plot_time_end: pd.Timestamp,
    freq: str = '12h',
    format: str = '%m-%d %HZ',
) -> None:
    """ Format time series date axis. """
    date_ticks = pd.date_range(plot_time_start, plot_time_end, freq=freq)
    ax.set_xticks(date_ticks)
    ax.set_xlim([plot_time_start, plot_time_end])
    date_format = mdates.DateFormatter(format)
    ax.xaxis.set_major_formatter(date_format)


def remove_top_and_right_spines(ax):
    """ Remove the top and right spines from an axis. """
    ax.spines[['right', 'top']].set_visible(False)


def set_square_aspect(ax):
    """ Set the aspect ratio of the axes to be square. """
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    if ax.get_xscale() == 'log':
        xlim = np.log10(xlim)
    if ax.get_yscale() == 'log':
        ylim = np.log10(ylim)
    aspect = (xlim[1]-xlim[0]) / (ylim[1]-ylim[0])
    ax.set_aspect(aspect, adjustable='box')


def create_inset_colorbar(plot_handle, ax, bounds=None, **kwargs):
    """ Create an inset colorbar. """
    # bounds = [x0, y0, width, height]
    if bounds is None:
        bounds = [0.93, 0.5, 0.02, 0.45]
    cax = ax.inset_axes(bounds, axes_class=Axes)
    cbar = plt.colorbar(plot_handle, cax=cax, **kwargs)
    return cbar, cax


def set_gridlines(ax, **kwargs):
    """ Plot gridlines on an axis. """
    kwargs.setdefault('color', 'k')
    kwargs.setdefault('alpha', 0.075)
    kwargs.setdefault('linestyle', '-')
    kwargs.setdefault('linewidth', 0.5)

    ax.grid(**kwargs)


# Subplot functions and classes
class SubplotLabeler:
    """ Create an object to label subplots. """
    def __init__(self):
        # self.ax = ax
        self.count = 0

    def increment_counter(self):
        self.count += 1

    def add_label(self, ax, **kwargs):
        label_letter = chr(ord('@') + (self.count % 26 + 1))
        text = f'({label_letter.lower()})'
        label_subplot(ax, text, **kwargs)
        self.increment_counter()


def label_subplot(
    ax,
    text,
    fontsize=normal_font_size,
    loc='upper left',
    nudge_x=0,
    nudge_y=0,
    **kwargs,
):
    """ Add text to subplot in the specified location. """
    if loc == 'upper left':
        xy = (0.05 + nudge_x, 0.95 + nudge_y)
        ha = 'left'
        va = 'top'
    elif loc == 'upper right':
        xy = (0.95 + nudge_x, 0.95 + nudge_y)
        ha = 'right'
        va = 'top'

    ax.annotate(
        text=text,
        xy=xy,
        xycoords='axes fraction',
        ha=ha,
        va=va,
        fontsize=fontsize,
        **kwargs,
    )


def axes_to_iterator(axes):
    """ Convert a 2D array of axes to an iterator. """
    return iter(axes.ravel())


def spectrogram(
    time: np.ndarray,
    frequency: np.ndarray,
    spectra: np.ndarray,
    ax: Optional[Axes] = None,
    **kwargs
) -> QuadMesh:
    """Plot a spectrogram (frequency spectra over time).

    Args:
        time (np.ndarray): Times with shape (t,).
        frequency (np.ndarray): Frequencies with shape (f,).
        spectra (np.ndarray): Spectral values with shape (t, f).
        ax (Optional[Axes], optional): Matplotlib Axes to plot on.
            Defaults to None.

    Returns:
        QuadMesh: Spectrogram plotted using pcolormesh.
    """
    kwargs.setdefault('cmap', colorcet.cm.CET_L19)

    ax = get_ax(ax)

    frequency_grid, time_grid = np.meshgrid(frequency, time, indexing='xy')
    return ax.pcolormesh(time_grid, frequency_grid, spectra, **kwargs)


# Project-specific


def annotate_legs(ax, scale=1.05):
    y_max = ax.get_ylim()[1] * scale

    leg_1_time_start, leg_1_time_end = mission_specifications.leg_1_times
    leg_2_time_start, leg_2_time_end = mission_specifications.leg_2_times

    ax.annotate(
        '',
        xy=(leg_1_time_start, y_max), xycoords='data',
        xytext=(leg_1_time_end, y_max), textcoords='data',
        arrowprops={'arrowstyle': '<->', 'linewidth': 0.5},
        annotation_clip=False,
    )
    ax.annotate(
        'Leg 1',
        xy=(leg_1_time_start + (leg_1_time_end - leg_1_time_start)/2, 1.01 * y_max),
        xycoords='data',
        ha='center',
        va='bottom',
        fontsize=small_font_size,
        annotation_clip=False,
    )
    ax.annotate(
        '',
        xy=(leg_2_time_start, y_max), xycoords='data',
        xytext=(leg_2_time_end, y_max), textcoords='data',
        arrowprops={'arrowstyle': '<->', 'linewidth': 0.5},
        annotation_clip=False,
    )
    ax.annotate(
        'Leg 2',
        xy=(leg_2_time_start + (leg_2_time_end - leg_2_time_start)/2, 1.01 * y_max),
        xycoords='data',
        ha='center',
        va='bottom',
        fontsize=small_font_size,
        annotation_clip=False,
    )