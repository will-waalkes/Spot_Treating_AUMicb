from jax import jit, numpy as jnp
import numpy as np
import pandas as pd
from glob import glob
from scipy.stats import binned_statistic
import astropy.units as u
from tensorflow_probability.substrates.jax.math import batch_interp_rectilinear_nd_grid as nd_interp


def bt_settl_model_spectra(bin_wl):
    paths = glob('/Users/wiwa8630/bt-settl/lte*.txt')
    bt_settl_temperature_grid = {}

    for i, path in enumerate(paths):
        with open(path, 'r') as w:
            temperature = int(w.read(1000).splitlines()[1].split('=')[1].split(' ')[1])
        bt_settl_temperature_grid[temperature] = path

    bt_settl_grid = np.zeros((len(bt_settl_temperature_grid), len(bin_wl)-1))

    bt_settl_temperature_grid_keys = np.sort(list(bt_settl_temperature_grid.keys())).astype(np.float64)

    for i, temp in enumerate(bt_settl_temperature_grid_keys):
        path = bt_settl_temperature_grid[temp]
        spectrum = pd.read_csv(
            path,
            comment='#',
            delimiter='\s+',
            names=['wavelength', 'flux']
        )

        bt_settl_grid[i] = binned_statistic(
            (spectrum['wavelength'].values * u.AA).to(u.um).value, spectrum['flux'].values,
            bins=bin_wl, statistic=np.nanmean
        ).statistic

    return bt_settl_temperature_grid_keys, bt_settl_grid


def get_interp_stellar_spectrum(bin_wl):

    bt_settl_temperature_grid_keys, bt_settl_grid = bt_settl_model_spectra(bin_wl)

    x_grid_points = (
        bt_settl_temperature_grid_keys.astype(np.float32), 
        bin_wl[:-1].astype(np.float32)
    )

    @jit
    def interp(interp_temperature):
        ones = jnp.ones_like(bin_wl[:-1])
        interp_point = jnp.column_stack([
            interp_temperature * ones,
            bin_wl[:-1]
        ]).astype(np.float32)

        return nd_interp(
            interp_point,
            x_grid_points,
            bt_settl_grid.astype(np.float32),
            axis=0
        )

    return interp
