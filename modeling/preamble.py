import scipy.io
from chromatic import plt, u, np
from chromatic import *
import pandas as pd
import batman
import lightkurve as lk
import lmfit
import emcee
import random
import os
import corner
from scipy.optimize import minimize

from specutils.manipulation import box_smooth, gaussian_smooth, trapezoid_smooth

from lmfit import Model, Parameters
from scipy.interpolate import interp1d

from ldtk import LDPSetCreator, BoxcarFilter
import ldtk

from svo_filters import svo

from specutils.spectra import Spectrum1D, SpectralRegion
from specutils.fitting import fit_generic_continuum

import jax
from jax import numpy as jnp
from jax import jit
from jax.scipy.signal import fftconvolve
import numpyro
from numpyro import optim, distributions as dist

from astropy.convolution import convolve, convolve_fft
from astropy.convolution import Gaussian1DKernel

from bt_settl import get_interp_stellar_spectrum

import matplotlib.animation as animation

from scipy.signal import correlate

from scipy.optimize import curve_fit

import astropy.constants as c
# from RECTE import RECTE

from matplotlib.gridspec import GridSpec
from IPython.display import display
from ipywidgets import interactive, VBox, HBox, FloatSlider

from astropy.constants import m_p

from shone.opacity import Opacity
from shone.transmission import heng_kitzmann_2017, de_wit_seager_2013
from shone.opacity.dace import download_molecule
from shone.chemistry import FastchemWrapper, species_name_to_fastchem_name

from expecto import get_spectrum
from fleck.jax import ActiveStar, bin_spectrum

from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, to_hex

G_102_back_dict = scipy.io.readsav('../data/data_from_hannah/G102/Backward_spectra.sav', verbose=False)
G_102_for_dict = scipy.io.readsav('../data/data_from_hannah/G102/Forward_spectra.sav', verbose=False)
G_141_back_dict = scipy.io.readsav('../data/data_from_hannah/G141/Backward_spectra.sav', verbose=False)
G_141_for_dict = scipy.io.readsav('../data/data_from_hannah/G141/Forward_spectra.sav', verbose=False)

model_wavelengths = np.linspace(0.7,1.7,1500) * u.micron
btsettl_grid = get_interp_stellar_spectrum(model_wavelengths.value)
btsettl_wavelengths = model_wavelengths.value[:-1]

hst_orbital_period = 95.7 # minutes
hst_per_in_days = hst_orbital_period * (1./60.) * (1./24.)

visits = {
    'F21': {
        'Grism': 'G141',
        'Forward': G_141_for_dict,
        'Backward': G_141_back_dict,
        'BJD_times': np.array(pd.read_csv('../data/F21_bjdtimes.csv')['BJD'][:]) * u.day,
        'time_lower': 2459455.708 * u.day,
        'time_upper': 2459455.738 * u.day,
        'T0 (BJD_TDB)': 2459455.9895 * u.day,
        'exp (s)': 4.9784 * u.s,
        'native resolution': 46.3 * u.angstrom
    },
    'S22': {
        'Grism': 'G102',
        'Forward': G_102_for_dict,
        'Backward': G_102_back_dict,
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
    'T_rot (d)': 4.863 * u.day,
    'inclination': 88.5,
    'eccentricity': 0.0,
    'longitude_of_periastron': 89.0
}

WFC3_Median_Spectra = {
    
    "F21" : {
        
        "Forward" : {
            "w" : None,"f" : None,"e" : None,
        },
        "Reverse" : {
            "w" : None,"f" : None,"e" : None,
        }
    },
    
    "S22" : {

        "Forward" : {
            "w" : None,"f" : None,"e" : None,
        },
        "Reverse" : {
            "w" : None,"f" : None,"e" : None,
        }
    }
}

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

# Define phoenix model wrappers:

def calculate_logg(Rstar):

    M = (0.6*u.M_sun)
    G = c.G
    R = (Rstar*u.R_sun)
    gstar = (G*M/(R**2)).decompose().to('u.cm/u.s**2')
    logg = np.log10( gstar.value )
    
    return logg

def phoenix_1T(parameters, sigma, convolution_method='astropy', kernel_type='astropy'):

    T_phot = parameters['T_phot']
    rstar = parameters['R_star']
    log_g = calculate_logg(rstar)
    
    S_phot = get_phoenix_photons(wavelength = model_wavelengths,temperature = float(T_phot), metallicity = 0.12, logg=log_g)

    E_per_photon = (3e10*6.626e-27)/(S_phot[0] * 1e-4) #ergs per photon
    _f = S_phot[1] / (1e5 * u.angstrom * u.cm * u.cm * u.s) # converted from the phoenix units of photons/nm/m^2/s
    _f = (_f * E_per_photon) * u.erg # now the model spectrum is in flux calibrated units
    exclude_nans = ~np.isnan(_f)
    model_wave = S_phot[0].value[exclude_nans]
    model_flux = _f.value[exclude_nans]
    
    # Convolve the model spectrum
    convolved = convolve_spectrum(model_wave, model_flux, sigma=sigma, method = convolution_method, kernel_type = kernel_type)
    
    return convolved

def phoenix_2T(parameters, sigma, convolution_method='astropy', kernel_type='astropy'):

    T_phot = parameters['T_phot']
    T_spot = parameters['T_spot']
    f_spot = parameters['f_spot']
    rstar = parameters['R_star']
    log_g = calculate_logg(rstar)
    
    S_spot = get_phoenix_photons(wavelength = model_wavelengths,temperature = float(T_spot), metallicity = 0.12, logg = log_g)
    S_phot = get_phoenix_photons(wavelength = model_wavelengths,temperature = float(T_phot), metallicity = 0.12, logg = log_g)

    # Calculate model spectrum
    E_per_photon = (3e10*6.626e-27)/(S_phot[0] * 1e-4) #ergs per photon
    _f = (f_spot*S_spot[1] + (1-f_spot)*S_phot[1]) / (1e5 * u.angstrom * u.cm * u.cm * u.s) # converted from the phoenix units of photons/nm/m^2/s
    _f = (_f * E_per_photon) * u.erg # now the model spectrum is in flux calibrated units
    exclude_nans = ~np.isnan(_f)
    model_wave = S_spot[0].value[exclude_nans]
    model_flux = _f.value[exclude_nans]
    
    # Convolve the model spectrum
    convolved = convolve_spectrum(model_wave, model_flux, sigma=sigma, method = convolution_method, kernel_type = kernel_type)
    
    return convolved

def phoenix_3T(parameters, sigma, convolution_method='astropy', kernel_type='astropy'):

    T_phot = parameters['T_phot']
    T_spot = parameters['T_spot']
    T_other = parameters['T_other']
    f_spot = parameters['f_spot']
    f_phot = parameters['f_phot']
    rstar = parameters['R_star']
    
    f_other = 1.0 - (f_spot + f_phot)
    log_g = calculate_logg(rstar)
    
    S_spot = get_phoenix_photons(wavelength = model_wavelengths,temperature = float(T_spot), metallicity = 0.12, logg= log_g)
    S_phot = get_phoenix_photons(wavelength = model_wavelengths,temperature = float(T_phot), metallicity = 0.12, logg= log_g)
    S_other = get_phoenix_photons(wavelength = model_wavelengths,temperature = float(T_other), metallicity = 0.12, logg = log_g)

    # Calculate model spectrum
    E_per_photon = (3e10*6.626e-27)/(S_phot[0] * 1e-4) #ergs per photon
    _f = (f_spot*S_spot[1] + f_phot*S_phot[1] + f_other * S_other[1]) / (1e5 * u.angstrom * u.cm * u.cm * u.s) # converted from the phoenix units of photons/nm/m^2/s
    _f = (_f * E_per_photon) * u.erg # now the model spectrum is in flux calibrated units
    exclude_nans = ~np.isnan(_f)
    model_wave = S_spot[0].value[exclude_nans]
    model_flux = _f.value[exclude_nans]
    
    # Convolve the model spectrum
    convolved = convolve_spectrum(model_wave, model_flux, sigma=sigma, method = convolution_method, kernel_type = kernel_type)
    
    return convolved

# Define wrapper functions for BT-SETTL Models

def btsettl_1T(parameters, sigma, convolution_method='astropy', kernel_type='astropy'):

    T_phot = parameters['T_phot']
    S_phot = btsettl_grid(float(T_phot))

    # Calculate combined spectrum
    _f = S_phot
    exclude_nans = ~np.isnan(_f)
    model_wave = btsettl_wavelengths[exclude_nans]
    model_flux = _f[exclude_nans]
    
    # Convolve the model spectrum
    convolved = convolve_spectrum(model_wave, model_flux, sigma=sigma, method = convolution_method, kernel_type = kernel_type)
    
    return convolved

def btsettl_2T(parameters, sigma, convolution_method='astropy', kernel_type='astropy'):

    f_spot = parameters['f_spot']
    T_spot = parameters['T_spot']
    T_phot = parameters['T_phot']
    S_spot = btsettl_grid(float(T_spot))
    S_phot = btsettl_grid(float(T_phot))

    # Calculate combined spectrum
    _f = (f_spot*S_spot + (1.0-f_spot)*S_phot)
    exclude_nans = ~np.isnan(_f)    
    model_wave = btsettl_wavelengths[exclude_nans]
    model_flux = _f[exclude_nans]
    
    # Convolve the model spectrum
    convolved = convolve_spectrum(model_wave, model_flux, sigma=sigma, method = convolution_method, kernel_type = kernel_type)
    
    return convolved

def btsettl_3T(parameters, sigma, convolution_method='astropy', kernel_type='astropy'):

    T_phot = parameters['T_phot']
    T_spot = parameters['T_spot']
    T_other = parameters['T_other']
    f_other = parameters['f_other']
    f_spot = parameters['f_spot']
    f_phot = 1.0 - (f_other + f_spot)
    
    S_spot = btsettl_grid(float(T_spot))
    S_phot = btsettl_grid(float(T_phot))
    S_other = btsettl_grid(float(T_other))
    
    # Calculate model spectrum
    _f = (f_spot*S_spot + f_phot*S_phot + f_other*S_other)
    exclude_nans = ~np.isnan(_f)
    model_wave = btsettl_wavelengths[exclude_nans]
    model_flux = _f[exclude_nans]
    
    # Convolve the model spectrum
    convolved = convolve_spectrum(model_wave, model_flux, sigma=sigma, method = convolution_method, kernel_type = kernel_type)
    
    return convolved

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

for visit in tqdm(['F21','S22']):
    for direction in ['Forward','Reverse']:

        # Prepare data for both directions
        exptime = visits[f'{visit}']['exp (s)']
        binwidth = visits[f'{visit}']['native resolution']
        grism = visits[f'{visit}']['Grism']
        
        trimmed_r = read_rainbow(f"../data/{visit}_{direction}_trimmed_pacman_spec.rainbow.npy")
    
        # Process forward direction data
        median_spectrum = trimmed_r.get_median_spectrum().value
        e_per_s = median_spectrum / exptime
        e_per_s_per_angstrom = e_per_s / binwidth
        _w, _s, _e = read_sensitivity_curve(grism=grism)
        binned_filter_response = bintogrid(_w.value, _s.value, newx=trimmed_r.wavelength.value)['y'] * u.cm**2 / u.erg
        calibrated_data_flux = e_per_s_per_angstrom / binned_filter_response
            
        WFC3_Median_Spectra[f'{visit}'][f'{direction}']['w'] = trimmed_r.wavelength
        WFC3_Median_Spectra[f'{visit}'][f'{direction}']['f'] = calibrated_data_flux
        WFC3_Median_Spectra[f'{visit}'][f'{direction}']['e'] = 0.005 * calibrated_data_flux

F21F_rainbow = read_rainbow(f"../data/F21_Forward_trimmed_pacman_spec.rainbow.npy")
F21R_rainbow = read_rainbow(f"../data/F21_Reverse_trimmed_pacman_spec.rainbow.npy")
S22F_rainbow = read_rainbow(f"../data/S22_Forward_trimmed_pacman_spec.rainbow.npy")
S22R_rainbow = read_rainbow(f"../data/S22_Reverse_trimmed_pacman_spec.rainbow.npy")

# File paths for Hannah's white light curves
datasets = {
    "F21_Backward": {
        "lightcurve": "../data/AUMicb_F21_Backward_lightcurve_data.txt",
    },
    "F21_Forward": {
        "lightcurve": "../data/AUMicb_F21_Forward_lightcurve_data.txt",
    },
    "S22_Backward": {
        "lightcurve": "../data/AUMicb_S22_Backward_lightcurve_data.txt",
    },
    "S22_Forward": {
        "lightcurve": "../data/AUMicb_S22_Forward_lightcurve_data.txt",
    }
}

# Function to load lightcurve data
def load_lightcurve(file_path):
    return pd.read_csv(
        file_path, 
        delim_whitespace=True, 
        comment='#', 
        names=["MJD", "Flux", "Uncertainty", "Shift"]
    )

# Example: Load all datasets
sh_data = {}
for key, paths in datasets.items():
    sh_data[key] = {
        "lightcurve": load_lightcurve(paths["lightcurve"])
    }


def ramp_model(r1=0,r2=0,r3=0,phase=None):
    
    ramp = 1. - np.exp(-r1 * phase + r2) + r3*phase

    return ramp
    
# def calculate_delta_D_spot(parameters, w):
    
#     f_spot, f_chord, T_spot, T_amb, transit_depth = parameters
    
#     S_spot = get_phoenix_photons(wavelength = w, temperature = T_spot, metallicity = 0.12, logg= 4.52)[1]
#     S_amb = get_phoenix_photons(wavelength = w, temperature = T_amb, metallicity = 0.12, logg= 4.52)[1]
    
#     flux_ratio = S_spot/S_amb
#     top = (1. - f_chord) + (f_chord * flux_ratio)
#     bottom = (1. - f_spot) + (f_spot * flux_ratio)
    
#     delta_D_spot = ((top / bottom) - 1.) * transit_depth
    
#     depth_factor = (delta_D_spot/transit_depth) + 1.

#     return delta_D_spot * 1e6