from jax import jit, numpy as jnp
import numpy as np
import pandas as pd
from glob import glob
from scipy.stats import binned_statistic
import astropy.units as u
from tensorflow_probability.substrates.jax.math import batch_interp_rectilinear_nd_grid as nd_interp
import re

def sphinx_model_spectra(bin_wl, path=None):
    if path is None:
        path = '../../../../model-spectra/sphinx/SPECTRA/*_spectra.txt'

    paths = glob(path)
    
    # Extract parameters from filenames
    temperatures = []
    metallicities = []
    cto_ratios = []
    loggs = []  # Add logg to store surface gravity values
    spectra_list = []
    
    for path in paths:
        # Parse filename: Teff_2000.0_logg_4.0_logZ_-0.5_CtoO_0.3_spectra.txt
        filename = path.split('/')[-1]
        pattern = r'Teff_([\d\.]+)_logg_([\d\.]+)_logZ_([-\d\.]+)_CtoO_([-\d\.]+)_spectra\.txt'
        match = re.search(pattern, filename)
        
        if match:
            teff = float(match.group(1))
            logg = float(match.group(2))
            logz = float(match.group(3))
            ctoo = float(match.group(4))
            
            # Read spectrum
            spectrum = np.loadtxt(path)
            
            # Interpolate spectrum to bin_wl
            interp_flux = np.interp(bin_wl, spectrum[:, 0], spectrum[:, 1])
            
            temperatures.append(teff)
            loggs.append(logg)
            metallicities.append(logz)
            cto_ratios.append(ctoo)
            spectra_list.append(interp_flux)
    
    # Convert to unique sorted arrays for grid points
    unique_temps = np.sort(np.unique(temperatures))
    unique_loggs = np.sort(np.unique(loggs))
    unique_metals = np.sort(np.unique(metallicities))
    unique_ctos = np.sort(np.unique(cto_ratios))
    
    # Create 4D grid
    n_temps = len(unique_temps)
    n_loggs = len(unique_loggs)
    n_metals = len(unique_metals)
    n_ctos = len(unique_ctos)
    n_wl = len(bin_wl)
    
    # Create mapping dictionaries
    temp_to_idx = {t: i for i, t in enumerate(unique_temps)}
    logg_to_idx = {g: i for i, g in enumerate(unique_loggs)}
    metal_to_idx = {m: i for i, m in enumerate(unique_metals)}
    cto_to_idx = {c: i for i, c in enumerate(unique_ctos)}
    
    # Initialize 4D grid: [n_temps, n_loggs, n_metals, n_ctos, n_wl]
    sphinx_grid = np.zeros((n_temps, n_loggs, n_metals, n_ctos, n_wl))
    
    # Fill the grid
    for temp, logg, metal, cto, spectrum in zip(temperatures, loggs, metallicities, cto_ratios, spectra_list):
        i_temp = temp_to_idx[temp]
        i_logg = logg_to_idx[logg]
        i_metal = metal_to_idx[metal]
        i_cto = cto_to_idx[cto]
        sphinx_grid[i_temp, i_logg, i_metal, i_cto, :] = spectrum
    
    # Return grid points and the grid
    return (unique_temps.astype(jnp.float64),
            unique_loggs.astype(jnp.float64),
            unique_metals.astype(jnp.float64),
            unique_ctos.astype(jnp.float64),
            bin_wl.astype(jnp.float64),
            sphinx_grid.astype(jnp.float64))

def get_interp_stellar_spectrum(bin_wl, path=None):
    """
    Returns a function that interpolates spectra in Teff, logg, metallicity, C/O, and wavelength.
    """
    temp_grid, logg_grid, metal_grid, cto_grid, wl_grid, sphinx_grid = sphinx_model_spectra(bin_wl, path)
    
    x_grid_points = (
        temp_grid.astype(jnp.float64),
        logg_grid.astype(jnp.float64),
        metal_grid.astype(jnp.float64),
        cto_grid.astype(jnp.float64),
        wl_grid.astype(jnp.float64)
    )
    
    @jit
    def interp(interp_temperature, interp_logg, interp_metallicity, interp_cto):
        ones = jnp.ones_like(bin_wl)
        interp_point = jnp.column_stack([
            interp_temperature * ones,
            interp_logg * ones,
            interp_metallicity * ones,
            interp_cto * ones,
            bin_wl
        ]).astype(jnp.float64)
        
        return nd_interp(
            interp_point,
            x_grid_points,
            sphinx_grid.astype(jnp.float64),
            axis=0
        )
    
    return interp

# interp_func = get_interp_stellar_spectrum(bin_wl, path='/path/to/sphinx/files/Teff*_spectra.txt')
# spectrum = interp_func(2500.0, -0.5, 0.3)  # Teff, logZ, CtoO