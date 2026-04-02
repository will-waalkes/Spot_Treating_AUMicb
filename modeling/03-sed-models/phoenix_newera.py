# import h5py
# import f90nml
# import numpy as np
# import jax.numpy as jnp
# from jax import jit
# from glob import glob
# from scipy.stats import binned_statistic
# import astropy.units as u
from tensorflow_probability.substrates.jax.math import batch_interp_rectilinear_nd_grid as nd_interp

# def phoenix_newera_model_spectra(bin_wl, path=None):
#     """
#     Read Phoenix NewEra model spectra from HDF5 files and bin them.
    
#     Parameters:
#     bin_wl : array-like
#         Bin edges for wavelength binning
#     path : str, optional
#         Glob pattern for HDF5 files. Default pattern expects:
#         lte{teff:05d}-{logg:4.2f}-{zscale:4.2f}.PHOENIX-NewEra-ACES-COND-2023.HSR.h5
#         or similar naming convention.
    
#     Returns:
#     temp_grid : ndarray
#         Effective temperature grid (K)
#     logg_grid : ndarray
#         Surface gravity grid (log10(cm/s²))
#     metal_grid : ndarray
#         Metallicity grid ([M/H])
#     phoenix_grid : ndarray
#         4D grid of binned spectra (temp, logg, metal, wavelength_bin)
#     """
#     if path is None:
#         # Adjust pattern based on your file naming convention
#         path = '../../../../model-spectra/phoenix-newera/logg*/lte*.HSR.h5'
    
#     paths = glob(path)
    
#     # Extract parameters from files
#     temperatures = []
#     loggs = []
#     metals = []
#     spectra_list = []
    
#     for filepath in paths:
#         try:
#             with h5py.File(filepath, 'r') as fh5:
#                 # Read namelist to get parameters
#                 nml_str = (str(fh5['/PHOENIX_NAMELIST/phoenix_nml'][()].tobytes()))[2:-1]
#                 target_nml = f90nml.reads(nml_str)
                
#                 # Extract parameters
#                 teff = float(target_nml['phoenix']['teff'])
#                 logg = float(target_nml['phoenix']['logg'])
#                 zscale = float(target_nml['phoenix']['zscale'])  # Metallicity
                
#                 # Read spectrum: wavelength in Angstrom, flux (linear scale)
#                 wl = fh5['/PHOENIX_SPECTRUM/wl'][()]  # Angstrom
#                 fl = 10.**fh5['/PHOENIX_SPECTRUM/flux'][()]  # Convert from log10
                
#                 # Convert wavelength from Angstrom to microns
#                 wl_um = wl * 1e-4  # Angstrom to microns
                
#                 # Bin spectrum to common wavelength grid
#                 binned_flux = binned_statistic(
#                     wl_um,
#                     fl,
#                     bins=bin_wl,
#                     statistic=np.nanmean
#                 ).statistic
                
#                 temperatures.append(teff)
#                 loggs.append(logg)
#                 metals.append(zscale)
#                 spectra_list.append(binned_flux)
                
#         except Exception as e:
#             print(f"Error reading {filepath}: {e}")
#             continue
    
#     # Get unique sorted grids
#     temp_grid = np.sort(np.unique(temperatures)).astype(np.float64)
#     logg_grid = np.sort(np.unique(loggs)).astype(np.float64)
#     metal_grid = np.sort(np.unique(metals)).astype(np.float64)
    
#     print(f"Grid sizes: Teff={len(temp_grid)}, logg={len(logg_grid)}, [M/H]={len(metal_grid)}")
    
#     # Create 4D grid: [temp, logg, metal, wavelength_bin]
#     phoenix_grid = np.zeros((len(temp_grid), len(logg_grid), len(metal_grid), len(bin_wl)-1))
    
#     # Fill the grid
#     for i, temp in enumerate(temp_grid):
#         for j, logg_val in enumerate(logg_grid):
#             for k, metal_val in enumerate(metal_grid):
#                 for idx, (t, g, m) in enumerate(zip(temperatures, loggs, metals)):
#                     if (np.isclose(t, temp) and 
#                         np.isclose(g, logg_val) and 
#                         np.isclose(m, metal_val)):
#                         phoenix_grid[i, j, k, :] = spectra_list[idx]
#                         break
    
#     return temp_grid, logg_grid, metal_grid, phoenix_grid


# def get_interp_phoenix_spectrum(bin_wl, path=None):
#     """
#     Return a JIT-compiled interpolation function for Phoenix NewEra models.
    
#     Parameters:
#     bin_wl : array-like
#         Bin edges for wavelength binning
#     path : str, optional
#         Glob pattern for HDF5 files
    
#     Returns:
#     interp : function
#         Interpolation function with signature interp(teff, logg, metallicity)
#     """
#     temp_grid, logg_grid, metal_grid, phoenix_grid = phoenix_newera_model_spectra(bin_wl, path)
    
#     # Prepare grid points for interpolation (must be in increasing order)
#     x_grid_points = (
#         temp_grid.astype(jnp.float64),
#         logg_grid.astype(jnp.float64),
#         metal_grid.astype(jnp.float64),
#         bin_wl[:-1].astype(jnp.float64)
#     )
    
#     @jit
#     def interp(interp_teff, interp_logg, interp_metallicity):
#         """Interpolate spectrum at given Teff, logg, and metallicity."""
#         ones = jnp.ones_like(bin_wl[:-1])
#         interp_point = jnp.column_stack([
#             interp_teff * ones,
#             interp_logg * ones,
#             interp_metallicity * ones,
#             bin_wl[:-1]
#         ]).astype(jnp.float64)
        
#         return nd_interp(
#             interp_point,
#             x_grid_points,
#             phoenix_grid.astype(jnp.float64),
#             axis=0
#         )
    
#     return interp


# def get_phoenix_newera_spectrum(teff, logg, metallicity, grid, wavelengths_um=None):
#     """
#     Get normalized Phoenix NewEra spectrum for given parameters.
    
#     Parameters:
#     teff : float
#         Effective temperature in K
#     logg : float
#         Surface gravity in log10(cm/s²)
#     metallicity : float
#         Metallicity [M/H]
#     grid : function
#         Interpolation function from get_interp_phoenix_spectrum
#     wavelengths_um : array, optional
#         Wavelength array in microns (if None, uses bin edges)
    
#     Returns:
#     wave_jax : jnp.ndarray
#         Wavelength array in microns
#     flux_jax : jnp.ndarray
#         Normalized flux
#     """
#     # Get interpolated spectrum (already binned to bin_wl)
#     gridspec = grid(
#         jnp.array(teff, dtype=jnp.float64),
#         jnp.array(logg, dtype=jnp.float64),
#         jnp.array(metallicity, dtype=jnp.float64)
#     )
    
#     if wavelengths_um is None:
#         # Use the bin centers
#         wave_centers = (grid.x_grid_points[-1] + np.diff(grid.x_grid_points[-1])/2)
#         wave_jax = jnp.array(wave_centers)
#     else:
#         wave_jax = jnp.array(wavelengths_um)
    
#     # Calculate normalization factor (Stefan-Boltzmann)
#     # sigma_sb = 5.67e-5  # erg/cm²/s/K⁴
#     # nf = (sigma_sb * teff**4) / jnp.trapezoid(gridspec, x=wave_jax * 1e4)
#     re_normed_flux = gridspec #* nf
    
#     return wave_jax, re_normed_flux


# def get_phoenix_wavelengths(filepath):
#     """
#     Extract wavelength array from a Phoenix NewEra HDF5 file.
    
#     Parameters:
#     filepath : str
#         Path to the HDF5 file
    
#     Returns:
#     wavelengths : numpy.ndarray
#         Wavelength array in microns
#     """
#     with h5py.File(filepath, 'r') as fh5:
#         wl = fh5['/PHOENIX_SPECTRUM/wl'][()]  # Angstrom
#         return wl * 1e-4  # Convert to microns


# def get_wavelength_bin_edges(wavelengths):
#     """
#     Calculate bin edges from wavelength centers.
    
#     Parameters:
#     wavelengths : numpy.ndarray
#         Wavelength centers in microns
    
#     Returns:
#     bin_edges : numpy.ndarray
#         Bin edges with length len(wavelengths) + 1
#     """
#     half_diffs = np.diff(wavelengths) / 2.0
#     bin_edges = np.zeros(len(wavelengths) + 1)
#     bin_edges[0] = wavelengths[0] - half_diffs[0]
#     bin_edges[1:-1] = wavelengths[:-1] + half_diffs
#     bin_edges[-1] = wavelengths[-1] + half_diffs[-1]
#     return bin_edges

# # EXAMPLE USAGE:
# '''
# # Get wavelength bin edges from a sample file
# sample_wavelengths = get_phoenix_wavelengths('lte05000-5.00-0.0.PHOENIX-NewEra-ACES-COND-2023.HSR.h5')
# bin_edges = get_wavelength_bin_edges(sample_wavelengths)

# # Create interpolation function
# phoenix_interp = get_interp_phoenix_spectrum(
#     bin_edges,
#     path='/path/to/phoenix/*.PHOENIX-NewEra-ACES-COND-2023.HSR.h5'
# )

# # Get spectrum for specific parameters
# wave, flux = get_phoenix_newera_spectrum(
#     teff=5000.0,
#     logg=4.5,
#     metallicity=0.0,
#     grid=phoenix_interp
# )
# '''

from jax import jit, vmap
import jax.numpy as jnp
import numpy as np
import h5py
import f90nml
import os
import glob
from functools import partial
from typing import Dict, Tuple, Optional, List

class PhoenixNewEraLibrary:
    """
    A JAX-compatible class to provide access to Phoenix NewEra model stellar spectra
    from locally stored HDF5 files.
    
    The library expects HDF5 files in the format:
    lte{teff:05d}-{logg:4.2f}-{zscale:4.2f}.PHOENIX-NewEra-ACES-COND-2023.HSR.h5
    """
    
    def __init__(self, directory: str = ".", photons: bool = True):
        """
        Initialize a Phoenix NewEra model library.
        
        Parameters
        ----------
        directory : str
            The path to the directory containing the HDF5 model files.
        photons : bool
            Should the units be in photons, rather than power?
            (Note: For JAX compatibility, units are stripped but this flag
            affects the stored flux values.)
        """
        self._are_the_units_photons = photons
        self._model_directory = directory
        
        # Initialize containers
        self._temperature_grid = None
        self._logg_grid = None
        self._metallicity_grid = None
        self._wavelength_grid = None
        self._spectrum_grid = None  # 4D array: [temp_idx, logg_idx, metal_idx, wavelength_idx]
        
        # Build the grid
        self._build_grid()
    
    def _build_grid(self):
        """
        Scan the model directory and build the complete grid.
        This is called once at initialization.
        """
        # Find all HDF5 files
        pattern = os.path.join(self._model_directory, "*.h5")
        filepaths = glob.glob(pattern)
        
        if len(filepaths) == 0:
            raise ValueError(f"No HDF5 files found in {self._model_directory}")
        
        print(f"Found {len(filepaths)} model files. Building grid...")
        
        # Extract parameters and spectra
        temperatures = []
        loggs = []
        metallicities = []
        spectra_list = []
        wavelengths = None
        
        for filepath in filepaths:
            try:
                with h5py.File(filepath, 'r') as fh5:
                    # Read namelist to get parameters
                    nml_str = (str(fh5['/PHOENIX_NAMELIST/phoenix_nml'][()].tobytes()))[2:-1]
                    target_nml = f90nml.reads(nml_str)
                    
                    # Extract parameters
                    teff = float(target_nml['phoenix']['teff'])
                    logg = float(target_nml['phoenix']['logg'])
                    zscale = float(target_nml['phoenix']['zscale'])
                    
                    # Read spectrum
                    wl = fh5['/PHOENIX_SPECTRUM/wl'][()]  # Angstrom
                    fl = 10.**fh5['/PHOENIX_SPECTRUM/flux'][()]  # Convert from log10
                    
                    # Convert wavelength from Angstrom to microns
                    wl_um = wl * 1e-4
                    
                    # Store wavelength grid from first file
                    if wavelengths is None:
                        wavelengths = wl_um
                    
                    # Convert to photon flux if requested
                    if self._are_the_units_photons:
                        # h = 6.626e-34 J·s, c = 2.998e8 m/s, convert to photons
                        # W/m²/nm = J/s/m²/nm, divide by photon energy in J
                        # Photon energy = h*c / wavelength (in meters)
                        wl_m = wl_um * 1e-6  # Convert microns to meters
                        photon_energy = 6.626e-34 * 2.998e8 / wl_m  # J per photon
                        fl = fl / photon_energy  # photons/s/m²/nm
                    
                    temperatures.append(teff)
                    loggs.append(logg)
                    metallicities.append(zscale)
                    spectra_list.append(fl)
                    
            except Exception as e:
                print(f"Warning: Could not read {filepath}: {e}")
                continue
        
        # Convert to numpy arrays
        temperatures = np.array(temperatures)
        loggs = np.array(loggs)
        metallicities = np.array(metallicities)
        wavelengths = np.array(wavelengths)
        
        # Get unique sorted grids
        self._temperature_grid = np.sort(np.unique(temperatures)).astype(np.float64)
        self._logg_grid = np.sort(np.unique(loggs)).astype(np.float64)
        self._metallicity_grid = np.sort(np.unique(metallicities)).astype(np.float64)
        self._wavelength_grid = wavelengths.astype(np.float64)
        
        # Create mapping from parameter values to indices
        temp_to_idx = {t: i for i, t in enumerate(self._temperature_grid)}
        logg_to_idx = {g: i for i, g in enumerate(self._logg_grid)}
        metal_to_idx = {m: i for i, m in enumerate(self._metallicity_grid)}
        
        # Initialize the 4D grid
        n_temps = len(self._temperature_grid)
        n_loggs = len(self._logg_grid)
        n_metals = len(self._metallicity_grid)
        n_wave = len(self._wavelength_grid)
        
        self._spectrum_grid = np.full(
            (n_temps, n_loggs, n_metals, n_wave), 
            np.nan, 
            dtype=np.float64
        )
        
        # Fill the grid
        for t, g, m, spec in zip(temperatures, loggs, metallicities, spectra_list):
            i = temp_to_idx[t]
            j = logg_to_idx[g]
            k = metal_to_idx[m]
            self._spectrum_grid[i, j, k, :] = spec
        
        # Check for missing grid points
        if np.any(np.isnan(self._spectrum_grid)):
            n_missing = np.sum(np.isnan(self._spectrum_grid[:, :, :, 0]))
            print(f"Warning: {n_missing} grid points missing")
        
        # Convert to JAX arrays for fast interpolation
        self._temperature_grid_jax = jnp.array(self._temperature_grid)
        self._logg_grid_jax = jnp.array(self._logg_grid)
        self._metallicity_grid_jax = jnp.array(self._metallicity_grid)
        self._wavelength_grid_jax = jnp.array(self._wavelength_grid)
        self._spectrum_grid_jax = jnp.array(self._spectrum_grid)
        
        print(f"Grid built: Teff={n_temps}, logg={n_loggs}, [M/H]={n_metals}, wavelengths={n_wave}")
    
    def _find_indices_and_weights(self, value: float, grid: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Find the indices and interpolation weights for a given value on a grid.
        
        Parameters
        ----------
        value : float
            The value to interpolate to
        grid : jnp.ndarray
            The grid of possible values (sorted)
            
        Returns
        -------
        indices : jnp.ndarray
            The indices of the bounding grid points (1 or 2 elements)
        weights : jnp.ndarray
            The interpolation weights (sums to 1)
        """
        # Find the index where value would be inserted
        idx = jnp.searchsorted(grid, value)
        
        # Handle boundaries
        idx = jnp.clip(idx, 1, len(grid) - 1)
        
        # Get lower and upper indices
        idx_low = idx - 1
        idx_high = idx
        
        # Get grid values
        low_val = grid[idx_low]
        high_val = grid[idx_high]
        
        # Check if exact match
        is_exact = jnp.isclose(value, low_val) | jnp.isclose(value, high_val)
        
        # For exact matches, use only one index
        def exact_case():
            # Find which one is exact
            exact_idx = jnp.where(jnp.isclose(grid, value), size=1)[0]
            return exact_idx, jnp.array([1.0])
        
        def interpolate_case():
            # Linear interpolation weight
            weight_high = (value - low_val) / (high_val - low_val)
            weight_low = 1.0 - weight_high
            return jnp.array([idx_low, idx_high]), jnp.array([weight_low, weight_high])
        
        indices, weights = jax.lax.cond(
            is_exact,
            exact_case,
            interpolate_case
        )
        
        return indices, weights
    
    @partial(jit, static_argnums=(0,))
    def get_spectrum(
        self,
        temperature: jnp.ndarray,
        logg: jnp.ndarray,
        metallicity: jnp.ndarray,
        return_wavelengths: bool = True
    ) -> jnp.ndarray:
        """
        Get a Phoenix NewEra model spectrum for given parameters.
        JIT-compiled for performance.
        
        Parameters
        ----------
        temperature : jnp.ndarray
            Temperature(s) in K (scalar or 1D array)
        logg : jnp.ndarray
            Surface gravity log10[g/(cm/s²)] (scalar or 1D array)
        metallicity : jnp.ndarray
            Metallicity [M/H] (scalar or 1D array)
        return_wavelengths : bool
            If True, return both wavelengths and spectrum
            If False, return only spectrum
            
        Returns
        -------
        wavelengths : jnp.ndarray (if return_wavelengths=True)
            The wavelength grid in microns
        spectrum : jnp.ndarray
            The interpolated spectrum (same shape as input parameters + wavelength dimension)
        """
        # Handle broadcasting for multiple parameters
        temperature = jnp.atleast_1d(temperature)
        logg = jnp.atleast_1d(logg)
        metallicity = jnp.atleast_1d(metallicity)
        
        # Find indices and weights for each parameter
        temp_indices, temp_weights = self._find_indices_and_weights(
            temperature, self._temperature_grid_jax
        )
        logg_indices, logg_weights = self._find_indices_and_weights(
            logg, self._logg_grid_jax
        )
        metal_indices, metal_weights = self._find_indices_and_weights(
            metallicity, self._metallicity_grid_jax
        )
        
        # We need to handle potentially different numbers of indices
        # For scalar inputs, we have 1 or 2 indices each
        # For vector inputs, we need to vectorize the interpolation
        
        # Check if we're dealing with scalar or vector
        is_scalar = (len(temperature) == 1 and len(logg) == 1 and len(metallicity) == 1)
        
        if is_scalar:
            # Single point interpolation
            spectrum = self._interpolate_single(
                temp_indices, temp_weights,
                logg_indices, logg_weights,
                metal_indices, metal_weights
            )
        else:
            # Vectorized interpolation (broadcast to common shape)
            # This is more complex - for now, we'll use vmap
            # First, broadcast all parameters to the same shape
            # We'll assume the user passes consistent shapes
            pass  # TODO: Implement vectorized version
        
        if return_wavelengths:
            return self._wavelength_grid_jax, spectrum
        else:
            return spectrum
    
    @partial(jit, static_argnums=(0,))
    def _interpolate_single(
        self,
        temp_indices: jnp.ndarray,
        temp_weights: jnp.ndarray,
        logg_indices: jnp.ndarray,
        logg_weights: jnp.ndarray,
        metal_indices: jnp.ndarray,
        metal_weights: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Perform 3D linear interpolation for a single point.
        """
        n_wave = len(self._wavelength_grid_jax)
        
        # Initialize result
        result = jnp.zeros(n_wave, dtype=jnp.float64)
        
        # Loop over all combinations (max 2x2x2 = 8 combinations)
        for i in range(len(temp_indices)):
            ti = temp_indices[i]
            tw = temp_weights[i]
            for j in range(len(logg_indices)):
                lj = logg_indices[j]
                lw = logg_weights[j]
                for k in range(len(metal_indices)):
                    mk = metal_indices[k]
                    mw = metal_weights[k]
                    
                    weight = tw * lw * mw
                    if weight > 0:
                        result += weight * self._spectrum_grid_jax[ti, lj, mk, :]
        
        return result
    
    def get_wavelengths(self) -> jnp.ndarray:
        """Return the wavelength grid in microns."""
        return self._wavelength_grid_jax
    
    def get_available_temperatures(self) -> jnp.ndarray:
        """Return the available temperature grid."""
        return self._temperature_grid_jax
    
    def get_available_loggs(self) -> jnp.ndarray:
        """Return the available logg grid."""
        return self._logg_grid_jax
    
    def get_available_metallicities(self) -> jnp.ndarray:
        """Return the available metallicity grid."""
        return self._metallicity_grid_jax


# Global instance (to be initialized with your directory)
_phoenix_newera_library = None


def init_phoenix_library(directory: str = ".", photons: bool = True) -> PhoenixNewEraLibrary:
    """
    Initialize the global Phoenix NewEra library instance.
    
    Parameters
    ----------
    directory : str
        Path to directory containing HDF5 files
    photons : bool
        Use photon units if True, else power units
        
    Returns
    -------
    library : PhoenixNewEraLibrary
        The initialized library
    """
    global _phoenix_newera_library
    _phoenix_newera_library = PhoenixNewEraLibrary(directory=directory, photons=photons)
    return _phoenix_newera_library


def get_phoenix_spectrum(
    temperature: jnp.ndarray,
    logg: jnp.ndarray,
    metallicity: jnp.ndarray,
    return_wavelengths: bool = True
) -> jnp.ndarray:
    """
    Convenience function to get a Phoenix NewEra spectrum.
    Requires init_phoenix_library to be called first.
    
    Parameters
    ----------
    temperature : jnp.ndarray
        Temperature(s) in K
    logg : jnp.ndarray
        Surface gravity log10[g/(cm/s²)]
    metallicity : jnp.ndarray
        Metallicity [M/H]
    return_wavelengths : bool
        If True, return both wavelengths and spectrum
        
    Returns
    -------
    wavelengths : jnp.ndarray (if return_wavelengths=True)
        The wavelength grid in microns
    spectrum : jnp.ndarray
        The interpolated spectrum
    """
    if _phoenix_newera_library is None:
        raise ValueError("Phoenix library not initialized. Call init_phoenix_library first.")
    
    return _phoenix_newera_library.get_spectrum(
        temperature=temperature,
        logg=logg,
        metallicity=metallicity,
        return_wavelengths=return_wavelengths
    )