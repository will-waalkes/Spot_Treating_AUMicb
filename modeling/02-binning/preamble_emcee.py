import warnings
warnings.filterwarnings("ignore", message="It appears that you're using a Mac with one of Apple's ARM-based processors")

# We need to import numpyro first (though we use it last)
import numpyro
from numpyro.infer import MCMC, NUTS
from numpyro import distributions as dist
cpu_cores = 7
numpyro.set_host_device_count(cpu_cores)
numpyro.set_platform("cpu")

import jax
from jax import config
config.update("jax_enable_x64", True)
from jax import jit
from jax.scipy.optimize import minimize
from jax import random
from jax.random import PRNGKey, split
from jax import numpy as jnp
from jax.scipy.signal import fftconvolve
from jax.scipy.signal import convolve as jax_convolve

import shone
from shone.opacity.dace import download_molecule
from shone.chemistry import FastchemWrapper
from shone.chemistry import FastchemWrapper
from shone.opacity import Opacity
from shone.transmission import de_wit_seager_2013

import fleck
from fleck.jax import ActiveStar

from specutils.manipulation import box_smooth, gaussian_smooth, trapezoid_smooth
from specutils.spectra import Spectrum1D, SpectralRegion

import scipy.io
from scipy.optimize import fmin_powell, curve_fit
from scipy.interpolate import interp1d
from scipy.signal import correlate

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, to_hex

from IPython.display import display
from ipywidgets import interactive, VBox, HBox, FloatSlider

from astropy.convolution import convolve, convolve_fft
from astropy.convolution import Gaussian1DKernel
import astropy.units as u
import astropy.constants as c
from astropy.constants import G, m_p
from astropy.table import Table

import pandas as pd
import emcee
import random
import os
import corner
from chromatic import *
from svo_filters import svo
import numpy as np
import arviz
import arviz as az
from bt_settl import get_interp_stellar_spectrum

panchromatic_bin_edges = jnp.geomspace(0.3, 30, 5000)
panchromatic_wavelengths = panchromatic_bin_edges[:-1] + jnp.diff(panchromatic_bin_edges)
panchromatic_btsettl_grid = get_interp_stellar_spectrum(panchromatic_bin_edges)

species = ['H2O', 'CH4', 'CO2', 'CO','NH3']

visits = {
    'F21': {
        'Grism': 'G141',
        # 'Forward': G_141_for_dict,
        # 'Backward': G_141_back_dict,
        'BJD_times': np.array(pd.read_csv('../data/F21_bjdtimes.csv')['BJD'][:]) * u.day,
        'time_lower': 2459455.708 * u.day,
        'time_upper': 2459455.738 * u.day,
        'T0 (BJD_TDB)': 2459455.9895 * u.day,
        'exp (s)': 4.9784 * u.s,
        'native resolution': 46.3 * u.angstrom
    },
    'S22': {
        'Grism': 'G102',
        # 'Forward': G_102_for_dict,
        # 'Backward': G_102_back_dict,
        'BJD_times': np.array(pd.read_csv('../data/S22_bjdtimes.csv')['BJD'][:]) * u.day,
        'time_lower': 2459684.215 * u.day,
        'time_upper': 2459684.243 * u.day,
        'T0 (BJD_TDB)': 2459684.4959 * u.day, # This is 27 planetary orbits after the first transit, + 0.0054 days (the transit arrived 7 minutes late)
        'exp (s)': 9.67632 * u.s,
        'native resolution': 24.6 * u.angstrom
    }
}

systeminfo = {
    'duration (hr)': 3.5 * u.hr,
    'T_orb (d)': 8.463 * u.day,
    'T_rot (d)': 4.86 * u.day,
    'inclination': 89.5,
    'eccentricity': 0.05,
    'longitude_of_periastron': 88.4
}

def read_sensitivity_curve(grism='G141'):
    path = f'../data/WFC3.IR.{grism}.1st.sens.2.fits'

    response = fits.open(path)

    w = response[1].data['wavelength']/1e4 * u.micron
    s = response[1].data['sensitivity'] * u.cm * u.cm / u.erg
    e = response[1].data['error'] * u.cm * u.cm / u.erg
    
    return w, s, e

# Example Usage:
# w,s,e = read_sensitivity_curve(grism='G102')
# plt.errorbar(w,s,yerr=e)
# plt.xlabel(f'Wavelength ({w.unit})')
# plt.ylabel(f'{s.unit}')

def calculate_logg(Rstar):

    M = (0.6*u.M_sun)
    G = c.G
    R = (Rstar*u.R_sun)
    gstar = (G*M/(R**2)).decompose().to('u.cm/u.s**2')
    logg = np.log10( gstar.value )
    
    return logg

def initialize_walkers(nwalkers, params_config):
    """
    Initializes parameter values for MCMC walkers based on the given configuration.

    Parameters:
    - nwalkers: int, number of walkers.
    - params_config: dict, keys are parameter names, and values are (min, max) tuples for uniform distribution.

    Returns:
    - p0: np.ndarray, initial walker positions (nwalkers x ndim).
    """
    initial_params = []
    for param_name, bounds in params_config.items():
        low, high = bounds
        initial_params.append(np.random.uniform(low, high, nwalkers))
    
    # Transpose values to create the walker initialization array
    p0 = np.transpose(initial_params)
    
    return p0

def breathing_model(phase, b1, b2, b3, b4, **kwargs):

    breathing = 1. + (b1 * phase) + (b2 * phase**2.) + (b3 * phase**3.) + (b4 * phase**4.)
    
    return breathing

def ramp_model(phase, r1, r2, r3, **kwargs):
    
    ramp = 1. - jnp.exp( (-r1 * phase) + r2) + (r3 * phase)

    return ramp

# https://shone.readthedocs.io/en/latest/shone/examples/transmission.html#general-transmission-spectra
def transmission_spectrum_HengKitz17(log_atm_pressure = -1,
                                     atm_temp = 600 * u.K,
                                     log_kappa_cloud = -2,  # cloud opacity [cm2 / g]
                                     mmw=2.5,
                                     R_0=3.5 * u.R_earth,
                                     M_p = 10.2 * u.M_earth, **kwargs):
    """
    Compute a transmission spectrum for an atmosphere
    using the isothermal/isobaric approximation
    from Heng & Kitzmann (2017).
    """

    P_0 = 10**log_atm_pressure
    temperature = jnp.array([atm_temp.value])  # [K]
    pressure = jnp.array([P_0]) # [bar]
    chem = FastchemWrapper(temperature, pressure)
    vmr = chem.vmr()
    weights = chem.get_weights()

    weighted_opacities = []
    for i, spec in enumerate(species):
        op = binned_opacities[i](atm_temp.value, P_0)[0]  # cm2 / g
        col_idx = chem.get_column_index(species_name=spec)[0]
        species_weight = weights[col_idx] / mmw
    
        abund_weighted_opacity = op * species_weight * vmr[:, col_idx]
        weighted_opacities.append(abund_weighted_opacity)

    total_mol_opacity = jnp.array(weighted_opacities).sum(axis=0)
    
    g = ( (c.G * M_p) / (R_0)**2 ).decompose() # surface gravity
    
    # compute the planetary radius as a function of wavelength:
    Rp = heng_kitzmann_2017.transmission_radius_isothermal_isobaric(
        total_mol_opacity + (10**log_kappa_cloud),
        R_0.cgs.value, P_0, atm_temp, mmw, g.cgs.value
    )

    # convert to transit depth:
    transit_depth_ppm = 1e6 * (Rp / (0.8*u.R_sun).cgs.value) ** 2
        
    return (Rp / (0.8*u.R_sun).cgs.value), transit_depth_ppm

# Example Usage:

# Rp, transit_depth_ppm = transmission_spectrum_HengKitz17()
# plt.plot(wavelengths, Rp)

def get_BTSettl_spectrum(T=3700, grid=panchromatic_btsettl_grid, w = panchromatic_bin_edges*u.micron,
                         **kwargs):

    panchromatic_gridspec = panchromatic_btsettl_grid(jnp.array(T, dtype=jnp.float64))
    gridspec = grid(jnp.array(T, dtype=jnp.float64))
    
    # Calculate the normalization factor
    nf = (c.sigma_sb*(T*u.K)**4 ) / ( np.trapz(panchromatic_gridspec, x=panchromatic_wavelengths*1e4) * u.Unit('erg cm-2 s-1') )
    norm_factor = nf.decompose()
    normed_flux = gridspec * norm_factor

    flux = normed_flux * u.Unit('erg cm-2 s-1 AA-1')

    spec = Spectrum1D(spectral_axis=w, flux=flux)
    
    return spec

def convolve_spectrum(model_wavelength, model_flux, sigma, method='astropy', kernel_type = 'astropy'):
    """
    Convolve the high-res spectrum with a Gaussian kernel with 
    stddev `sigma`. Then interpolate the result onto the wavelength
    grid of the observations.
    """
    if kernel_type == 'astropy':
        kernel = Gaussian1DKernel(stddev=sigma.value).array

    if kernel_type == 'calculated':
        kernel = jnp.exp(
            -0.5 * (model_wavelength - jnp.mean(model_wavelength))**2 / 
            sigma**2
        )
        kernel = kernel / jnp.sum(kernel)
        
    if method == 'JAX-fft':
        convolved_model_flux = fftconvolve(model_flux, kernel, mode='same')

    if method == 'astropy':
        convolved_model_flux = convolve(model_flux, kernel)

    if method == 'astropy-fft':    
        convolved_model_flux = convolve_fft(model_flux, kernel)
    
    return convolved_model_flux

@jit
def breathing_model_jax(phase, b1, b2, b3, b4, **kwargs):

    phase = jnp.array(phase)
    breathing = 1. + (b1 * phase) + (b2 * phase**2.) + (b3 * phase**3.) + (b4 * phase**4.)
    
    return jnp.array(breathing)

@jit
def ramp_model_jax(phase, r1, r2, r3, **kwargs):

    phase = jnp.array(phase)
    ramp = 1. - jnp.exp( (-r1 * phase) + r2) + (r3 * phase)

    return jnp.array(ramp)

def get_mcmc_samples(visit, model_designation, n_steps, n_bins, n_burnin, n_dim):
    """
    Read and return the trimmed and transposed MCMC samples for a given visit and model.
    
    Parameters:
    -----------
    visit : str
        The visit identifier (e.g., 'visit01')
    model_designation : str
        The model designation used in the MCMC run
    n_steps : int
        Number of steps used in the MCMC run
    n_bins : int
        Number of bins used in the MCMC run (len(speclc_bin_edges)-1)
    n_burnin : int
        Number of burn-in steps to trim from the beginning
    
    Returns:
    --------
    numpy.ndarray
        Trimmed and transposed samples array with shape (n_dim, n_samples)
    """
    # Construct the filename following the same pattern as in the MCMC setup
    label = f'{visit}_{model_designation}_{n_steps}steps_{n_bins}bins_mcmc'
    samples_fname = f"../data/samples/{label}.h5"
    
    try:
        reader = emcee.backends.HDFBackend(samples_fname)
        sampler = reader.get_chain(discard=int(0.5*n_steps), flat=True)
        samples = sampler.reshape((-1, n_dim)).T
        
        return samples
    
    except Exception as e:
        print(f"Error loading samples for {label}: {str(e)}")
        return None

default_params={

    # Ramp Model Parameters
    "r1": 18.1,
    "r2": -6.7,
    "r3": 0,
    #Breathing params
    "b1":0,
    "b2":0,
    "b3":0,
    "b4":0,
    "HST_period":0.066,

    # Planet parameters
    "Mp":8.0,
    "planet_i":89.5,
    "P_orb":8.463,
    "t0":0.0,
    "R0": 0.044,
    "a_rstar": 18.5,
    "ecc": 0.0,

    # Stellar parameters
    "log_g":4.5,
    "stellar_i":85.0,
    'P_rot':4.86,
    "Rs":0.8,
    "Ms":0.6,
    "metallicity":0.0,
    "log_fixedspot_radii":-1.6,
    "f_cool_unocculted":0.2,
    "f_cool_occulted":0.2,
    "T_unocculted": 3100,
    "T_occulted": 3500,
    "T_phot": 3900,
    "spot1_lon":-1.1,
    "spot1_lat":1.2,
    "spot1_rad":0.1,
    "spot2_lon":0.875,
    "spot2_lat":1.79,
    "spot2_rad":0.07,
    "spot3_lon":0.09,
    "spot3_lat":1.48,
    "spot3_rad":0.32,
}

F21_speclc_bin_edges = np.array([1.14064,
1.16381,
1.18234,
1.20087,
1.21477,
1.23793,
1.25647,
1.28426,
1.31669,
1.33059,
1.34449,
1.35839,
1.39082,
1.40935,
1.42325,
1.43715,
1.45568,
1.47421,
1.51127,
1.5298,
1.54834,
1.5715,
1.62246,
1.645]) * u.micron

F21_speclc_err_factor = np.array([1.000	,
1.000	,
1.000	,
1.000	,
1.000	,
1.000	,
1.633	,
1.000	,
1.000	,
1.000	,
1.000	,
1.000	,
1.000	,
1.000	,
1.000	,
1.000	,
1.000	,
1.861	,
1.000	,
1.000	,
1.000	,
2.008	,
1.000	,])


F21_SED_err_factor = np.array([1.00	,
4.18	,
3.59	,
1.14	,
1.00	,
1.00	,
1.41	,
2.29	,
1.00	,
1.25	,
1.13	,
1.01	,
1.00	,
1.00	,
2.00	,
1.00	,
1.00	,
1.00	,
1.00	,
1.81	,
1.56	,
1.00	,
1.00	,
1.12	,
1.00	,
1.13	,
1.00	,
1.00	,
1.00	,
1.00	,
1.00	,
1.29	,
1.00	,
1.00	,
1.42	,
1.07	,
2.49	,
6.44	,
6.51	,
1.88	,
3.06	,
1.00	,
1.10	,
1.00	,
1.00	,
1.51	,
1.94	,
1.00	,
1.00	,
1.00	,
1.45	,
1.65	,
2.74	,
1.00	,
1.00	,
1.44	,
3.45	,
1.00	,
1.00	,
1.49	,
1.00	,
1.07	,
1.09	,
1.25	,
1.00	,
1.54	,
1.00	,
1.44	,
4.01	,
1.67	,
1.00	,
1.00	,
1.00	,
1.00	,
1.00	,
1.00	,
1.37	,
1.00	,
1.00	,
3.11	,
1.00	,
3.05	,
1.00	,
1.96	,
1.67	,
1.00	,
1.00	,
1.00	,
1.00	,
1.22	,
2.26	,
1.00	,
1.00	,
1.00	,
1.00	,
1.04	,
1.09	,
3.68	,
1.07	,
1.00	,
2.71	,
4.16	,
2.09	,
1.05	,
1.38	,
1.75	,
2.53	,
1.00	,
1.00	,
1.00	,
1.00	,
1.00	,])


S22_speclc_bin_edges = np.array([0.805,
0.81568514,
0.82306047,
0.8304358,
0.8697709,
0.87960467,
0.88943845,
0.90418911,
0.91156444,
0.92139821,
0.93369043,
0.94844109,
0.96073331,
0.96810864,
0.97302553,
0.97794241,
0.98531775,
1.01481907,
1.02465284,
1.03202817,
1.03694506,
1.04432039,
1.05661261,
1.06644638,
1.07382171,
1.10086459,
1.11315681,
1.12299059,
1.13]) * u.micron

S22_speclc_err_factor = np.array([1	,
1	,
1	,
2.93	,
1.00	,
1.00	,
1.00	,
1.00	,
1.00	,
1.00	,
1.00	,
1.55	,
1.00	,
1.00	,
1.00	,
1.00	,
1.00	,
1.00	,
1.00	,
1.00	,
1.00	,
1.00	,
1.00	,
1.00	,
2.93	,
1	,
1	,
1	,])

S22_SED_err_factor = ([1.00	,
1.00	,
1.00	,
1.00	,
1.17	,
1.18	,
1.00	,
1.00	,
1.00	,
1.00	,
1.00	,
1.00	,
1.00	,
1.15	,
1.10	,
1.16	,
1.27	,
1.81	,
1.86	,
1.84	,
1.13	,
1.24	,
1.33	,
1.32	,
1.64	,
1.06	,
1.00	,
1.00	,
1.00	,
1.00	,
1.05	,
1.00	,
1.08	,
1.01	,
1.00	,
1.06	,
1.00	,
1.00	,
1.18	,
1.00	,
1.04	,
1.00	,
1.00	,
1.00	,
1.00	,
1.00	,
1.04	,
1.35	,
1.16	,
1.01	,
1.00	,
1.02	,
1.18	,
1.00	,
1.00	,
1.00	,
1.00	,
1.00	,
1.00	,
1.18	,
1.41	,
1.17	,
1.00	,
1.02	,
1.00	,
1.00	,
1.00	,
1.16	,
1.00	,
1.00	,
1.00	,
1.00	,
1.00	,
1.14	,
1.78	,
1.28	,
1.00	,
1.00	,
1.00	,
1.11	,
1.67	,
2.05	,
1.24	,
1.10	,
1.11	,
1.26	,
1.24	,
1.00	,
1.13	,
1.00	,
1.13	,
1.12	,
1.00	,
1.00	,
1.00	,
1.00	,
1.00	,
1.00	,
1.19	,
1.00	,
1.06	,
1.00	,
1.00	,
1.03	,
1.00	,
1.02	,
1.02	,
1.00	,
1.00	,
1.00	,
1.16	,
1.71	,
2.64	,
1.70	,
1.00	,
1.29	,
1.73	,
1.71	,
1.16	,
1.07	,
1.00	,
1.00	,
1.00	,
1.06	,
1.11	,
1.00	,
1.20	,
1.00	,
1.15	,
1.00	,
1.00	,
1.00	,])