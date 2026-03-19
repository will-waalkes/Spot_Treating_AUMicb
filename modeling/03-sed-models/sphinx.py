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
        path = '/Users/wiwa8630/model-spectra/sphinx/SPECTRA/*_spectra.txt'

    paths = glob(path)
    
    # Extract parameters from filenames
    temperatures = []
    metallicities = []
    cto_ratios = []
    spectra_list = []
    
    for path in paths:
        # Parse filename: Teff_2000.0_logg_4.0_logZ_-0.5_CtoO_0.3_spectra.txt
        filename = path.split('/')[-1]
        pattern = r'Teff_([\d\.]+)_logg_[\d\.]+_logZ_([-\d\.]+)_CtoO_([-\d\.]+)_spectra\.txt'
        match = re.search(pattern, filename)
        
        if match:
            teff = float(match.group(1))
            logz = float(match.group(2))
            ctoo = float(match.group(3))
            
            # Read spectrum
            spectrum = pd.read_csv(
                path,
                comment='#',
                delimiter='\s+',
                names=['wavelength', 'flux']
            )
            
            # Bin spectrum
            binned_flux = binned_statistic(
                (spectrum['wavelength'].values * u.micrometer).to(u.um).value, 
                spectrum['flux'].values,
                bins=bin_wl, 
                statistic=np.nanmean
            ).statistic
            
            temperatures.append(teff)
            metallicities.append(logz)
            cto_ratios.append(ctoo)
            spectra_list.append(binned_flux)
    
    # Get unique sorted grids
    temp_grid = np.sort(np.unique(temperatures)).astype(np.float64)
    metal_grid = np.sort(np.unique(metallicities)).astype(np.float64)
    cto_grid = np.sort(np.unique(cto_ratios)).astype(np.float64)
    
    # Create 3D grid
    sphinx_grid = np.zeros((len(temp_grid), len(metal_grid), len(cto_grid), len(bin_wl)-1))
    
    # Fill the grid
    for i, temp in enumerate(temp_grid):
        for j, metal in enumerate(metal_grid):
            for k, cto in enumerate(cto_grid):
                # Find matching spectrum
                for idx, (t, m, c) in enumerate(zip(temperatures, metallicities, cto_ratios)):
                    if np.isclose(t, temp) and np.isclose(m, metal) and np.isclose(c, cto):
                        sphinx_grid[i, j, k, :] = spectra_list[idx]
                        break
    
    return temp_grid, metal_grid, cto_grid, sphinx_grid


def get_interp_stellar_spectrum(bin_wl, path=None):
    
    temp_grid, metal_grid, cto_grid, sphinx_grid = sphinx_model_spectra(bin_wl, path)
    
    x_grid_points = (
        temp_grid.astype(jnp.float32),
        metal_grid.astype(jnp.float32),
        cto_grid.astype(jnp.float32),
        bin_wl[:-1].astype(jnp.float32)
    )
    
    @jit
    def interp(interp_temperature, interp_metallicity, interp_cto):
        ones = jnp.ones_like(bin_wl[:-1])
        interp_point = jnp.column_stack([
            interp_temperature * ones,
            interp_metallicity * ones,
            interp_cto * ones,
            bin_wl[:-1]
        ]).astype(jnp.float32)
        
        return nd_interp(
            interp_point,
            x_grid_points,
            sphinx_grid.astype(jnp.float32),
            axis=0
        )
    
    return interp


# interp_func = get_interp_stellar_spectrum(bin_wl, path='/path/to/sphinx/files/Teff*_spectra.txt')
# spectrum = interp_func(2500.0, -0.5, 0.3)  # Teff, logZ, CtoO