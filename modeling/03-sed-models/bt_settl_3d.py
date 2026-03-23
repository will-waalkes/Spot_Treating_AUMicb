from jax import jit, numpy as jnp
import numpy as np
import pandas as pd
from glob import glob
from scipy.stats import binned_statistic
import astropy.units as u
from tensorflow_probability.substrates.jax.math import batch_interp_rectilinear_nd_grid as nd_interp
import re

def bt_settl_model_spectra_3d(bin_wl, path=None):
    """
    Load BT-Settl models and create a 3D grid with temperature, logg, and metallicity dimensions.
    
    Parameters:
    bin_wl: wavelength bin edges
    path: glob pattern for BT-Settl model files
    
    Returns:
    temp_grid: sorted array of unique temperatures
    logg_grid: sorted array of unique logg values
    metal_grid: sorted array of unique metallicities
    bt_settl_grid: 4D grid of spectra [n_temp, n_logg, n_metal, n_wavelength]
    """
    if path is None:
        path = '../../../../model-spectra/bt-settl/lte*.txt'

    paths = glob(path)
    
    # Extract parameters from filenames or file contents
    temperatures = []
    # loggs = []
    metallicities = []
    spectra_list = []
    
    for path in paths:
        with open(path, 'r') as f:
            content = f.read(2000)  # Read first 2000 chars to get headers
            lines = content.splitlines()
            
            # Extract temperature
            temp_line = [l for l in lines if 'teff' in l.lower()][0]
            temperature = float(temp_line.split('=')[1].split(' ')[1].strip())
            
            # Extract logg
            # logg_line = [l for l in lines if 'logg' in l.lower()][0]
            # logg = float(logg_line.split('=')[1].split(' ')[1].strip())
            
            # Extract metallicity
            meta_line = [l for l in lines if 'meta' in l.lower()][0]
            metallicity = float(meta_line.split('=')[1].split(' ')[1].strip())
        
        # Read spectrum
        spectrum = pd.read_csv(
            path,
            comment='#',
            delimiter=r'\s+',
            names=['wavelength', 'flux']
        )
        
        # Bin spectrum (convert wavelength from Angstrom to micron)
        binned_flux = binned_statistic(
            (spectrum['wavelength'].values * u.AA).to(u.um).value, 
            spectrum['flux'].values,
            bins=bin_wl, 
            statistic=np.nanmean
        ).statistic
        
        temperatures.append(temperature)
        # loggs.append(logg)
        metallicities.append(metallicity)
        spectra_list.append(binned_flux)
    
    # Get unique sorted grids
    temp_grid = np.sort(np.unique(temperatures)).astype(np.float64)
    # logg_grid = np.sort(np.unique(loggs)).astype(np.float64)
    metal_grid = np.sort(np.unique(metallicities)).astype(np.float64)
    
    # Create 4D grid [n_temp, n_logg, n_metal, n_wavelength]
    bt_settl_grid = np.zeros((len(temp_grid), #len(logg_grid), 
                              len(metal_grid), len(bin_wl)-1))
    
    # Fill the grid using a lookup dictionary
    lookup = {}
    for t, m, spec in zip(temperatures, #loggs, 
                             metallicities, spectra_list):
        lookup[(t, m)] = spec
    
    # Fill the grid, warning about missing combinations
    missing_count = 0
    for i, temp in enumerate(temp_grid):
        # for j, logg in enumerate(logg_grid):
        for k, metal in enumerate(metal_grid):
            key = (temp, #logg,
                    metal)
            if key in lookup:
                bt_settl_grid[i, k, :] = lookup[key]
            else:
                bt_settl_grid[i, k, :] = np.nan
                missing_count += 1
                # Only print first few warnings to avoid spam
                if missing_count <= 10:
                    print(f"Warning: Missing spectrum for T={temp}, metal={metal}")
    
    if missing_count > 0:
        print(f"Total missing grid points: {missing_count}")
    
    return temp_grid, metal_grid, bt_settl_grid


def get_interp_stellar_spectrum_3d(bin_wl, path=None):
    """
    Create an interpolation function for BT-Settl models with temperature, logg, and metallicity.
    
    Parameters:
    bin_wl: wavelength bin edges
    path: glob pattern for BT-Settl model files
    
    Returns:
    interp: JIT-compiled interpolation function that takes (temperature, logg, metallicity)
    """
    temp_grid, metal_grid, bt_settl_grid = bt_settl_model_spectra_3d(bin_wl, path)
    
    x_grid_points = (
        temp_grid.astype(jnp.float64),
        # logg_grid.astype(jnp.float64),
        metal_grid.astype(jnp.float64),
        bin_wl[:-1].astype(jnp.float64)
    )
    
    @jit
    def interp(interp_temperature, interp_metallicity):
        ones = jnp.ones_like(bin_wl[:-1])
        interp_point = jnp.column_stack([
            interp_temperature * ones,
            # interp_logg * ones,
            interp_metallicity * ones,
            bin_wl[:-1]
        ]).astype(jnp.float64)
        
        return nd_interp(
            interp_point,
            x_grid_points,
            bt_settl_grid.astype(jnp.float64),
            axis=0
        )
    
    return interp