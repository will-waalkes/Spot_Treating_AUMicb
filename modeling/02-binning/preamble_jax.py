import os
import warnings
import random
import pickle
from datetime import datetime
from collections import defaultdict
from functools import partial

# Environment configuration
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=8'

import numpyro
from numpyro import distributions as dist
from numpyro.infer import MCMC, NUTS, log_likelihood, hmc
from numpyro.infer.util import initialize_model
from numpyro.util import fori_collect

import jax
from jax import config
config.update("jax_enable_x64", True)
config.update('jax_platform_name', 'cpu')

import jax.numpy as jnp
from jax import jit, pmap, devices, device_get, lax, local_device_count, random, vmap, block_until_ready
from jax.random import PRNGKey, split
from jax.scipy.optimize import minimize
from jax.scipy.signal import fftconvolve
from jax.scipy.signal import convolve as jax_convolve

# Check devices
print('Available devices:', devices())
print('CPU devices:', devices('cpu'))

# Suppress warnings
warnings.filterwarnings('ignore', message="It appears that you're using a Mac with one of Apple's ARM-based processors")

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, to_hex
import corner

import astropy.units as u
import astropy.constants as c
from astropy.constants import G, m_p
from astropy.table import Table

import shone
from shone.opacity.dace import download_molecule
from shone.chemistry import FastchemWrapper
from shone.opacity import Opacity
from shone.transmission import de_wit_seager_2013

import fleck
from fleck.jax import ActiveStar

# from specutils.manipulation import box_smooth, gaussian_smooth, trapezoid_smooth
# from specutils.spectra import Spectrum1D, SpectralRegion

# from scipy.io import *
from scipy.optimize import fmin_powell, curve_fit
# from scipy.interpolate import interp1d
# from scipy.signal import correlate
from scipy.stats import gaussian_kde

# from astropy.convolution import convolve, convolve_fft
# from astropy.convolution import Gaussian1DKernel

# from IPython.display import display
# from ipywidgets import interactive, VBox, HBox, FloatSlider

from chromatic import *
from svo_filters import svo
from bt_settl import get_interp_stellar_spectrum

import arviz
import arviz as az

from tqdm.auto import tqdm

panchromatic_bin_edges = jnp.geomspace(0.3, 30, 5000)
panchromatic_wavelengths = panchromatic_bin_edges[:-1] + jnp.diff(panchromatic_bin_edges)
panchromatic_btsettl_grid = get_interp_stellar_spectrum(panchromatic_bin_edges)

species = ['H2O', 'CH4', 'CO2', 'CO','NH3']

visits = {
    'F21': {
        'Grism': 'G141',
        # 'Forward': G_141_for_dict,
        # 'Backward': G_141_back_dict,
        'BJD_times': np.array(pd.read_csv('../../data/F21_bjdtimes.csv')['BJD'][:]) * u.day,
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
        'BJD_times': np.array(pd.read_csv('../../data/S22_bjdtimes.csv')['BJD'][:]) * u.day,
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
    'eccentricity': 0.0,
    'longitude_of_periastron': 88.4
}

def read_sensitivity_curve(grism='G141'):
    path = f'../data/WFC3.IR.{grism}.1st.sens.2.fits'

    response = fits.open(path)

    w = response[1].data['wavelength']/1e4 * u.micron
    s = response[1].data['sensitivity'] * u.cm * u.cm / u.erg
    e = response[1].data['error'] * u.cm * u.cm / u.erg
    
    return w, s, e

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

@jit
def breathing_model_jax(phase, b1, b2, b3, b4):

    phase = jnp.array(phase)
    _breathing = 1. + (b1 * phase) + (b2 * phase**2.) + (b3 * phase**3.) + (b4 * phase**4.)
    breathing = _breathing/jnp.mean(_breathing)
    
    return jnp.array(breathing)

@jit
def ramp_model_jax(phase, r1, r2, r3):

    phase = jnp.array(phase)
    _ramp = 1. - jnp.exp( (-r1 * phase) + r2) + (r3 * phase)
    ramp = _ramp/jnp.mean(_ramp)
    
    return jnp.array(ramp)

@jit
def linear_model_jax(x, m):

    x = jnp.array(x)
    _line = m * x + 1
    line = _line/jnp.mean(_line)

    return jnp.array(line)

@jit
def get_planck_spectrum_jax(T, **kwargs):
    """
    Calculate the surface flux from a thermally emitted surface,
    according to Planck function.

    Parameters
    ----------
    wavelength : Quantity
        The wavelengths at which to calculate,
        with units of wavelength.
    temperature : Quantity
        The temperature of the thermal emitter,
        with units of K.

    Returns
    -------
    surface_flux : Quantity
        The surface flux, evaluated at the wavelengths.
    """

    # define variables as shortcut to the constants we need
    h = 6.62607e-27 # erg s
    k = 1.380649e-16 # erg/K
    c = 2.9979e18 # angstrom/s
    wavelength = panchromatic_wavelengths*1e4

    z = h * c / (wavelength * k * T) # units check out

    # calculate the intensity from the Planck function
    intensity = (2 * h * c**2 / wavelength**5 / (jnp.exp(z) - 1)) # Units are erg/s/A^3

    # calculate the flux assuming isotropic emission
    flux = jnp.pi * intensity * 1e16 # erg / (s * cm^2 * angstrom)

    # return the intensity
    wave_jax = jnp.array(panchromatic_wavelengths)
    flux_jax = jnp.array(flux)

    return wave_jax, flux_jax

@jit
def convolve_spectrum_jax(model_wavelength, model_flux, sigma, kernel_size=25, **kwargs):
    """
    Properly convolve a spectrum with a Gaussian kernel in JAX.
    
    Args:
        model_wavelength: Array of wavelengths (must be evenly spaced!)
        model_flux: Corresponding flux values
        sigma: Standard deviation of Gaussian kernel in wavelength units
        kernel_size: Number of elements in the kernel (odd number recommended)
        
    Returns:
        Convolved flux array
    """
    # Ensure inputs are JAX arrays
    model_wavelength = jnp.asarray(model_wavelength)
    model_flux = jnp.asarray(model_flux)
    
    # Create proper Gaussian kernel
    x = jnp.linspace(-(kernel_size//2), kernel_size//2, kernel_size)
    kernel = jnp.exp(-0.5 * (x/sigma)**2)
    kernel = kernel / jnp.sum(kernel)  # normalize
    
    # Perform convolution
    convolved = jax_convolve(model_flux, kernel, mode='same', method='fft')
    
    return convolved

@jit
def get_BTSettl_spectrum_jax(T, grid=panchromatic_btsettl_grid,**kwargs):

    gridspec = grid(jnp.array(T, dtype=jnp.float64))
    
    # Calculate the normalization factor
    sigma_sb = 5.67e-5 # erg/cm^2/s
    nf = (sigma_sb*(T)**4 ) / ( jnp.trapezoid(gridspec, x=jnp.array(panchromatic_wavelengths)*1e4) )
    re_normed_flux = gridspec * nf
    
    # gridspec = btsettl_grid(jnp.array(T, dtype=jnp.float64))
    wave_jax = jnp.array(panchromatic_wavelengths)
    flux_jax = jnp.array(re_normed_flux)
    
    return wave_jax, flux_jax

@jit
def get_binned_BTSettl_spectrum_jax(T, grid=panchromatic_btsettl_grid,data_wave=None,**kwargs):

    gridspec = grid(jnp.array(T, dtype=jnp.float64))
    
    # Calculate the normalization factor
    sigma_sb = 5.67e-5 # erg/cm^2/s
    nf = (sigma_sb*(T)**4 ) / ( jnp.trapezoid(gridspec, x=jnp.array(panchromatic_wavelengths)*1e4) )
    re_normed_flux = gridspec * nf
    
    flux_jax = jnp.array(re_normed_flux)
    
    binned_flux = shone.bin_spectrum(data_wave, panchromatic_wavelengths, flux_jax)
    
    return data_wave, binned_flux

F21_speclc_bin_edges = np.array([1.14296 ,
1.15263 ,
1.16318 ,
1.18179 ,
1.21123 ,
1.22782 ,
1.2374 ,
1.25408 ,
1.26804 ,
1.28748 ,
1.32515 ,
1.33291 ,
1.34682 ,
1.37287 ,
1.39266 ,
1.40199 ,
1.4164 ,
1.43361 ,
1.45982 ,
1.48938 ,
1.51175 ,
1.53161 ,
1.55509 ,
1.56757 ,
1.6138 ,
1.63867 ,]) * u.micron

F21_speclc_err_factor = np.array([1.00	,
1.07	,
1.41	,
2.17	,
1.30	,
1.00	,
1.28	,
1.18	,
1.66	,
1.86	,
1.00	,
1.21	,
1.98	,
1.51	,
1.00	,
1.27	,
1.43	,
1.89	,
2.01	,
1.72	,
1.71	,
1.61	,
1.30	,
2.54	,
1.69	,])


F21_SED_err_factor = np.array([11.606,10.016,3.183,1.000,2.718,3.906,6.409,
                               1.717,3.460,3.134,2.821,1.053,1.406,5.550,
                               2.273,1.000,1.000,1.073,5.025,4.385,1.159,
                               2.683,3.160,1.647,3.135,2.128,1.000,1.000,
                               1.669,2.694,3.583,1.971,1.145,3.989,3.019,
                               6.914,17.888,18.138,5.215,8.532,2.183,3.095,
                               1.195,1.271,4.225,5.362,1.000,1.000,1.430,
                               4.008,4.563,7.652,2.144,2.762,3.996,9.622,
                               1.113,1.704,4.172,1.000,2.942,3.088,3.457,
                               1.903,4.303,2.257,4.045,11.133,4.696,1.700,
                               1.226,2.184,2.651,2.481,1.283,3.845,1.202,
                               1.321,8.637,2.695,8.502,2.680,5.431,4.670,
                               1.505,1.212,1.459,1.062,3.382,6.319,2.400,
                               1.803,1.236,1.062,2.878,3.070,10.219,3.012,
                               2.082,7.564,11.554,5.829,2.976,3.876,4.837,
                               7.078,1.047,2.775,1.021,1.000,])


S22_speclc_bin_edges = np.array([0.8697709,
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

S22_speclc_err_factor = np.array([
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

S22_SED_err_factor = ([1.000, 1.000,1.000,1.000,1.000,1.000,
                       1.000,1.000,1.000,1.000,1.187,1.341,1.000,
                       1.000,1.000,1.000,1.012,1.000,1.030,1.283,
                       1.125,1.272,1.362,1.922,1.976,2.007,1.206,
                       1.387,1.357,1.392,1.798,1.202,1.000,1.000,
                       1.030,1.000,1.223,1.000,1.120,1.131,1.000,
                       1.219,1.000,1.000,1.295,1.062,1.009,1.053,
                       1.000,1.000,1.000,1.000,1.089,1.487,1.265,
                       1.016,1.056,1.034,1.364,1.005,1.000,1.000,
                       1.000,1.000,1.069,1.237,1.486,1.222,1.079,
                       1.103,1.000,1.026,1.056,1.268,1.000,1.014,
                       1.044,1.000,1.009,1.223,1.781,1.470,1.029,
                       1.000,1.000,1.185,1.762,2.281,1.247,1.102,
                       1.097,1.482,1.221,1.000,1.118,1.000,1.108,
                       1.134,1.041,1.000,1.000,1.000,1.000,1.000,
                       1.224,1.005,1.085,1.000,1.073,1.042,1.000,
                       1.026,1.085,1.000,1.013,1.013,1.194,1.791,
                       2.815,1.795,1.072,1.321,1.827,1.840,1.222,
                       1.106,1.000,1.091,1.000,1.098,1.097,1.151,
                       1.221,1.000,1.135,1.115,1.000,1.000,])