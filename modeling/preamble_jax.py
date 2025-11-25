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
# from shone.opacity.dace import download_molecule
# from shone.chemistry import FastchemWrapper
# from shone.opacity import Opacity
# from shone.transmission import de_wit_seager_2013

import fleck
from fleck.jax import ActiveStar

# from specutils.manipulation import box_smooth, gaussian_smooth, trapezoid_smooth
# from specutils.spectra import Spectrum1D, SpectralRegion

# from scipy.io import *
# from scipy.optimize import fmin_powell, curve_fit
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