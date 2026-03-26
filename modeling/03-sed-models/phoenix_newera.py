import h5py
import f90nml
import numpy as np
import jax.numpy as jnp
from jax import jit
from glob import glob
from scipy.stats import binned_statistic
import astropy.units as u
from tensorflow_probability.substrates.jax.math import batch_interp_rectilinear_nd_grid as nd_interp

def phoenix_newera_model_spectra(bin_wl, path=None):
    """
    Read Phoenix NewEra model spectra from HDF5 files and bin them.
    
    Parameters:
    bin_wl : array-like
        Bin edges for wavelength binning
    path : str, optional
        Glob pattern for HDF5 files. Default pattern expects:
        lte{teff:05d}-{logg:4.2f}-{zscale:4.2f}.PHOENIX-NewEra-ACES-COND-2023.HSR.h5
        or similar naming convention.
    
    Returns:
    temp_grid : ndarray
        Effective temperature grid (K)
    logg_grid : ndarray
        Surface gravity grid (log10(cm/s²))
    metal_grid : ndarray
        Metallicity grid ([M/H])
    phoenix_grid : ndarray
        4D grid of binned spectra (temp, logg, metal, wavelength_bin)
    """
    if path is None:
        # Adjust pattern based on your file naming convention
        path = '/path/to/phoenix/lte*.PHOENIX-NewEra-ACES-COND-2023.HSR.h5'
    
    paths = glob(path)
    
    # Extract parameters from files
    temperatures = []
    loggs = []
    metals = []
    spectra_list = []
    
    for filepath in paths:
        try:
            with h5py.File(filepath, 'r') as fh5:
                # Read namelist to get parameters
                nml_str = (str(fh5['/PHOENIX_NAMELIST/phoenix_nml'][()].tobytes()))[2:-1]
                target_nml = f90nml.reads(nml_str)
                
                # Extract parameters
                teff = float(target_nml['phoenix']['teff'])
                logg = float(target_nml['phoenix']['logg'])
                zscale = float(target_nml['phoenix']['zscale'])  # Metallicity
                
                # Read spectrum: wavelength in Angstrom, flux (linear scale)
                wl = fh5['/PHOENIX_SPECTRUM/wl'][()]  # Angstrom
                fl = 10.**fh5['/PHOENIX_SPECTRUM/flux'][()]  # Convert from log10
                
                # Convert wavelength from Angstrom to microns
                wl_um = wl * 1e-4  # Angstrom to microns
                
                # Bin spectrum to common wavelength grid
                binned_flux = binned_statistic(
                    wl_um,
                    fl,
                    bins=bin_wl,
                    statistic=np.nanmean
                ).statistic
                
                temperatures.append(teff)
                loggs.append(logg)
                metals.append(zscale)
                spectra_list.append(binned_flux)
                
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            continue
    
    # Get unique sorted grids
    temp_grid = np.sort(np.unique(temperatures)).astype(np.float64)
    logg_grid = np.sort(np.unique(loggs)).astype(np.float64)
    metal_grid = np.sort(np.unique(metals)).astype(np.float64)
    
    print(f"Grid sizes: Teff={len(temp_grid)}, logg={len(logg_grid)}, [M/H]={len(metal_grid)}")
    
    # Create 4D grid: [temp, logg, metal, wavelength_bin]
    phoenix_grid = np.zeros((len(temp_grid), len(logg_grid), len(metal_grid), len(bin_wl)-1))
    
    # Fill the grid
    for i, temp in enumerate(temp_grid):
        for j, logg_val in enumerate(logg_grid):
            for k, metal_val in enumerate(metal_grid):
                for idx, (t, g, m) in enumerate(zip(temperatures, loggs, metals)):
                    if (np.isclose(t, temp) and 
                        np.isclose(g, logg_val) and 
                        np.isclose(m, metal_val)):
                        phoenix_grid[i, j, k, :] = spectra_list[idx]
                        break
    
    return temp_grid, logg_grid, metal_grid, phoenix_grid


def get_interp_phoenix_spectrum(bin_wl, path=None):
    """
    Return a JIT-compiled interpolation function for Phoenix NewEra models.
    
    Parameters:
    bin_wl : array-like
        Bin edges for wavelength binning
    path : str, optional
        Glob pattern for HDF5 files
    
    Returns:
    interp : function
        Interpolation function with signature interp(teff, logg, metallicity)
    """
    temp_grid, logg_grid, metal_grid, phoenix_grid = phoenix_newera_model_spectra(bin_wl, path)
    
    # Prepare grid points for interpolation (must be in increasing order)
    x_grid_points = (
        temp_grid.astype(jnp.float32),
        logg_grid.astype(jnp.float32),
        metal_grid.astype(jnp.float32),
        bin_wl[:-1].astype(jnp.float32)
    )
    
    @jit
    def interp(interp_teff, interp_logg, interp_metallicity):
        """Interpolate spectrum at given Teff, logg, and metallicity."""
        ones = jnp.ones_like(bin_wl[:-1])
        interp_point = jnp.column_stack([
            interp_teff * ones,
            interp_logg * ones,
            interp_metallicity * ones,
            bin_wl[:-1]
        ]).astype(jnp.float32)
        
        return nd_interp(
            interp_point,
            x_grid_points,
            phoenix_grid.astype(jnp.float32),
            axis=0
        )
    
    return interp


def get_phoenix_newera_spectrum(teff, logg, metallicity, grid, wavelengths_um=None):
    """
    Get normalized Phoenix NewEra spectrum for given parameters.
    
    Parameters:
    teff : float
        Effective temperature in K
    logg : float
        Surface gravity in log10(cm/s²)
    metallicity : float
        Metallicity [M/H]
    grid : function
        Interpolation function from get_interp_phoenix_spectrum
    wavelengths_um : array, optional
        Wavelength array in microns (if None, uses bin edges)
    
    Returns:
    wave_jax : jnp.ndarray
        Wavelength array in microns
    flux_jax : jnp.ndarray
        Normalized flux
    """
    # Get interpolated spectrum (already binned to bin_wl)
    gridspec = grid(
        jnp.array(teff, dtype=jnp.float32),
        jnp.array(logg, dtype=jnp.float32),
        jnp.array(metallicity, dtype=jnp.float32)
    )
    
    if wavelengths_um is None:
        # Use the bin centers
        wave_centers = (grid.x_grid_points[-1] + np.diff(grid.x_grid_points[-1])/2)
        wave_jax = jnp.array(wave_centers)
    else:
        wave_jax = jnp.array(wavelengths_um)
    
    # Calculate normalization factor (Stefan-Boltzmann)
    # sigma_sb = 5.67e-5  # erg/cm²/s/K⁴
    # nf = (sigma_sb * teff**4) / jnp.trapezoid(gridspec, x=wave_jax * 1e4)
    re_normed_flux = gridspec #* nf
    
    return wave_jax, re_normed_flux


def get_phoenix_wavelengths(filepath):
    """
    Extract wavelength array from a Phoenix NewEra HDF5 file.
    
    Parameters:
    filepath : str
        Path to the HDF5 file
    
    Returns:
    wavelengths : numpy.ndarray
        Wavelength array in microns
    """
    with h5py.File(filepath, 'r') as fh5:
        wl = fh5['/PHOENIX_SPECTRUM/wl'][()]  # Angstrom
        return wl * 1e-4  # Convert to microns


def get_wavelength_bin_edges(wavelengths):
    """
    Calculate bin edges from wavelength centers.
    
    Parameters:
    wavelengths : numpy.ndarray
        Wavelength centers in microns
    
    Returns:
    bin_edges : numpy.ndarray
        Bin edges with length len(wavelengths) + 1
    """
    half_diffs = np.diff(wavelengths) / 2.0
    bin_edges = np.zeros(len(wavelengths) + 1)
    bin_edges[0] = wavelengths[0] - half_diffs[0]
    bin_edges[1:-1] = wavelengths[:-1] + half_diffs
    bin_edges[-1] = wavelengths[-1] + half_diffs[-1]
    return bin_edges

# EXAMPLE USAGE:
'''
# Get wavelength bin edges from a sample file
sample_wavelengths = get_phoenix_wavelengths('lte05000-5.00-0.0.PHOENIX-NewEra-ACES-COND-2023.HSR.h5')
bin_edges = get_wavelength_bin_edges(sample_wavelengths)

# Create interpolation function
phoenix_interp = get_interp_phoenix_spectrum(
    bin_edges,
    path='/path/to/phoenix/*.PHOENIX-NewEra-ACES-COND-2023.HSR.h5'
)

# Get spectrum for specific parameters
wave, flux = get_phoenix_newera_spectrum(
    teff=5000.0,
    logg=4.5,
    metallicity=0.0,
    grid=phoenix_interp
)
'''