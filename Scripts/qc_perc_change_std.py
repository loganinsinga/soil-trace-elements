import logging
import os
import sys
import math
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from math import sqrt

from scipy.interpolate import griddata
from scipy.spatial import cKDTree as KDTree

import cartopy.crs as ccrs
import cartopy.feature as cfeature

EXECUTION_ID = "Run1"
ELEMENT = "SE"
MODEL = "SVR"  # used for naming
FILTER_THRESHOLD = 0.3
MASTER_TABLE_ORIGINAL = r"C:\Publications\soil trace elements\European_mastertable_trimmed_V2_14April2021.xlsx"
MODEL_RESULTS_DIR = r"C:\Publications\soil trace elements\3_SVR_Se"
OUTPUT_DIR = r"C:\Publications\soil trace elements\3_SVR_Se"
NUM_PREDICTIONS = 100

LOGGER = logging.getLogger("my_logger")
LOGGER.setLevel(logging.DEBUG)

# define colors using RGB values for plots
d_brown = (0.451, 0.353, 0)
l_brown = (0.824, 0.706, 0.549)
tan = (1, 1, 0.784)
vl_green = (0.745, 0.902, 0.392)
l_green = (0.392, 0.588, 0)
d_green = (0.078, 0.235, 0)

red = (0.843, 0.196, 0.157)
really_red = (1, 0, 0)
orange = (0.961, 0.588, 0.392)
l_blue = (0.627, 0.706, 0.745)
d_blue = (0.275, 0.471, 0.706)

PLOT_INFO = {
    # "obs_v_pred_bins": [0, 100, 200, 300, 400, 500, 1000],  # S
    "obs_v_pred_bins": [0, 0.2, 0.3, 0.4, 0.5, 0.6, 100],  # Se
    "obs_v_pred_colors": [d_brown, l_brown, tan, vl_green, l_green, d_green],
    # "residuals_bins": [-1000000, -200, -100, 100, 200, 1000000],  # S
    "residuals_bins": [-100, -0.2, -0.1, 0.1, 0.2, 100],  # Se
    "residuals_colors": [d_blue, l_blue, tan, orange, red],
    "perc_res_bins": [-1, -0.2, -0.1, 0.1, 0.2, 1],
    "coeff_var_bins": [0, 0.1, 0.2, 0.3, 1],
    "coeff_var_colors": ["lightsteelblue", tan, "peachpuff", orange],
    "error_ratio_bins": [-1, -0.3, 0.3, 1],
    "error_ratio_colors": [l_blue, tan, orange],
    "perc_change_bins": [-150, -5, -3, -1, 1, 3, 5, 150],
    "perc_change_colors": [red, orange, "peachpuff", tan, "lightsteelblue", d_blue, "navy"],
    "perc_change_std_bins": [0, 0.25, 0.5, 0.75, 1, 10],  # SVR
    # "perc_change_std_bins": [0, 0.5, 1, 1.5, 2, 10],  # MLP
    "perc_change_std_colors": [d_blue, l_blue, tan, orange, red],
}


def plot_percchange_std_future(perc_chng_std_future_dropna, grid_info: dict):
    """Plots the std of n percent change iterations"""

    # interpolate obs and pred data to grid using griddata
    Z_perc_change_e = griddata(
        (grid_info["lon_f"], grid_info["lat_f"]),
        perc_chng_std_future_dropna["perc_change_std"],
        (grid_info["Xi_f"], grid_info["Yi_f"]),
        method="nearest",
    )

    tree = KDTree(np.c_[grid_info["lon_f"], grid_info["lat_f"]])  # concatenate lon lat arrays, define tree
    dist, _ = tree.query(
        np.c_[grid_info["Xi_f"].ravel(), grid_info["Yi_f"].ravel()], k=1
    )  # concatentate raveled (flattened) grid, k is # of neighbors
    dist = dist.reshape(grid_info["Xi_f"].shape)  # reshape distance back to grid (2d array)
    Z_perc_change_e[dist > 0.6] = np.nan

    # specify colorbar information
    perc_change_bins = PLOT_INFO["perc_change_std_bins"]
    perce_change_colors = PLOT_INFO["perc_change_std_colors"]
    cm = mpl.colors.LinearSegmentedColormap.from_list(
        "cmap_name", perce_change_colors, N=len(perc_change_bins)
    )  # create a colormap
    norm = mpl.colors.BoundaryNorm(perc_change_bins, len(perc_change_bins))

    # plot the figure
    fig = plt.figure(figsize=(18, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())

    ax.set_extent(
        [grid_info["min_lon_f"], grid_info["max_lon_f"], grid_info["min_lat_f"], grid_info["max_lat_f"]],
        crs=ccrs.PlateCarree(),
    )
    ax.coastlines("50m", zorder=1)  #'50m' means 1:50M resolution
    ax.add_feature(cfeature.BORDERS, zorder=3)

    data = ax.pcolormesh(
        grid_info["xi_shift_f"],
        grid_info["yi_shift_f"],
        Z_perc_change_e,
        transform=ccrs.PlateCarree(),
        zorder=1,
        cmap=cm,
        norm=norm,
        snap=True,
        alpha=1,
    )

    cbar = fig.colorbar(data, pad=0.02, format="%0.3f", shrink=0.93)
    cbar.set_ticks([])

    plt.savefig(
        os.path.join(OUTPUT_DIR, f"{EXECUTION_ID}_{ELEMENT}_percent_change_std_original_{MODEL}.png"),
        bbox_inches="tight",
    )


def main():

    # import results
    # setup_logging()

    # LOGGER.info("Gerating maps")

    # preparations
    # observed, predictions = import_results()
    # lat_lon_c, lat_lon_f = get_lat_lon_values(predictions)
    # predictions = calculate_percent_change(predictions)
    # predictions = calculate_residuals(predictions, observed)
    # predictions = calculate_and_apply_filters(predictions)

    perc_chng_std_future = pd.read_csv(
        r"C:\Publications\soil trace elements\Resources\SVR_PercChange_all_iter_eu_STD_Se_run2_linear.csv",
        names=["lon", "lat", "perc_change_std"],
    )
    perc_chng_std_future.set_index(keys=["lon", "lat"], drop=True, inplace=True)
    lat_lon_c = perc_chng_std_future.index

    perc_chng_std_future_dropna = perc_chng_std_future.dropna(how="all")
    lat_lon_f = perc_chng_std_future_dropna.index

    grid_info = get_grid_info(lat_lon_c, lat_lon_f)

    # generate plots
    # plot_obs_vs_pred(predictions, grid_info)
    # plot_obs_vs_pred_residuals(predictions, grid_info)
    # plot_coeff_var(predictions, grid_info)
    # plot_error_ratio(predictions, grid_info)
    # plot_perc_residuals(predictions, grid_info)
    # plot_percent_change(predictions, grid_info)
    plot_percchange_std_future(perc_chng_std_future_dropna, grid_info)
    # plot_percent_change_singlepredictor(predictions, grid_info, "AI")
    # plot_percent_change_singlepredictor(predictions, grid_info, "ET")
    # plot_percent_change_singlepredictor(predictions, grid_info, "precip")
    # plot_percent_change_singlepredictor(predictions, grid_info, f"{ELEMENT}dep")
    # plot_percent_change_singlepredictor(predictions, grid_info, "toc")

    # generate charst
    # plot_obs_vs_pred_chart(predictions)


def get_grid_info(lat_lon_c: pd.MultiIndex, lat_lon_f: pd.MultiIndex):
    """Retrieves grid information for plotting"""

    lon_c = lat_lon_c.get_level_values(level="lon").to_numpy()
    lat_c = lat_lon_c.get_level_values(level="lat").to_numpy()

    lon_f = lat_lon_f.get_level_values(level="lon").to_numpy()
    lat_f = lat_lon_f.get_level_values(level="lat").to_numpy()

    Xi_c, Yi_c, xi_shift_c, yi_shift_c, min_lon_c, max_lon_c, min_lat_c, max_lat_c = lat_lon_grid(lat_c, lon_c)
    Xi_f, Yi_f, xi_shift_f, yi_shift_f, min_lon_f, max_lon_f, min_lat_f, max_lat_f = lat_lon_grid(lat_f, lon_f)

    grid_info = {
        "lon_c": lon_c,
        "lat_c": lat_c,
        "lon_f": lon_f,
        "lat_f": lat_f,
        "Xi_c": Xi_c,
        "Yi_c": Yi_c,
        "xi_shift_c": xi_shift_c,
        "yi_shift_c": yi_shift_c,
        "min_lon_c": min_lon_c,
        "max_lon_c": max_lon_c,
        "min_lat_c": min_lat_c,
        "max_lat_c": max_lat_c,
        "Xi_f": Xi_f,
        "Yi_f": Yi_f,
        "xi_shift_f": xi_shift_f,
        "yi_shift_f": yi_shift_f,
        "min_lon_f": min_lon_f,
        "max_lon_f": max_lon_f,
        "min_lat_f": min_lat_f,
        "max_lat_f": max_lat_f,
    }

    return grid_info


def lat_lon_grid(lat, lon):
    """Prepares the lat lon grid for plotting maps"""

    # define the limits of the map, with an extra few pixels for some space
    min_lon = math.floor(min(lon)) - 2
    max_lon = math.ceil(max(lon)) + 2
    min_lat = math.floor(min(lat)) - 2
    max_lat = math.ceil(max(lat)) + 2

    # define the pixel size in degrees
    cell_size = 1

    # how many cells are there for vertical and horizontal
    lon_num = int((max_lon - min_lon) / cell_size)
    lat_num = int((max_lat - min_lat) / cell_size)

    xi = np.linspace(min_lon, max_lon, lon_num)  # create array defining limits for lat lon grid
    yi = np.linspace(min_lat, max_lat, lat_num)

    Xi, Yi = np.meshgrid(xi, yi)  # use np.meshgrid create grid, Xi and Yi are now 2D arrays

    xi_shift = Xi - cell_size / 2  # Subtract 1/2 the grid size from both lon and lat arrays
    yi_shift = Yi - cell_size / 2

    # Add 1 grid spacing to the right column of lon array and concatenate it as an additional column to the right
    xi_shift = np.c_[xi_shift, xi_shift[:, -1] + cell_size]

    # Duplicate the bottom row of the lon array and concatenate it to the bottom
    xi_shift = np.r_[xi_shift, [xi_shift[-1, :]]]

    # Duplicate the right-most column of lats array and concatenate it on the right
    yi_shift = np.c_[yi_shift, yi_shift[:, -1]]

    # Add 1 grid spacing to the bottom row of lat array and concatenate it as an additional row below
    yi_shift = np.r_[yi_shift, [yi_shift[-1, :] + cell_size]]

    return (
        Xi,
        Yi,
        xi_shift,
        yi_shift,
        min_lon,
        max_lon,
        min_lat,
        max_lat,
    )  # return both the shifted grid and the normal grid


if __name__ == "__main__":
    main()
