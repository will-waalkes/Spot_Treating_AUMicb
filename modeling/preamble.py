import scipy.io
from chromatic import *
from chromatic import plt, u, np
import pandas as pd
import numpy as np
import batman
import lightkurve as lk
import lmfit
import emcee
import random
import os

from specutils.manipulation import box_smooth, gaussian_smooth, trapezoid_smooth

from lmfit import Model, Parameters
from scipy.interpolate import interp1d
from RECTE import RECTE

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
from scipy.interpolate import interp1d

G_102_back_dict = scipy.io.readsav('../data/WFC3_data/G102/Backward_spectra.sav', verbose=False)
G_102_for_dict = scipy.io.readsav('../data/WFC3_data/G102/Forward_spectra.sav', verbose=False)
G_141_back_dict = scipy.io.readsav('../data/WFC3_data/G141/Backward_spectra.sav', verbose=False)
G_141_for_dict = scipy.io.readsav('../data/WFC3_data/G141/Forward_spectra.sav', verbose=False)

# g102_sigma = (210) * u.nm
# g141_sigma = (130) * u.nm
# g102_sigma = g102_sigma.to('micron')
# g141_sigma = g141_sigma.to('micron')
g102_sigma = 1.9 * u.pixel
g141_sigma = 2.1 * u.pixel

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
        'filter_response': svo.Filter('WFC3_IR.G141'),
        'filter_sigma': g141_sigma,
        'broadband_ldcs': [0.385,0.228]
    },
    'S22': {
        'Grism': 'G102',
        'Forward': G_102_for_dict,
        'Backward': G_102_back_dict,
        'BJD_times': np.array(pd.read_csv('../data/S22_bjdtimes.csv')['BJD'][:]) * u.day,
        'time_lower': 2459684.215 * u.day,
        'time_upper': 2459684.243 * u.day,
        'T0 (BJD_TDB)': 2459684.491 * u.day, # This is 27 planetary orbits after the first transit
        'exp (s)': 9.67632 * u.s, 
        'filter_response': svo.Filter('WFC3_IR.G102'),
        'filter_sigma': g102_sigma,
        'broadband_ldcs': [0.471,0.262]
    }
}

systeminfo = {
    'duration (hr)': 3.5 * u.hr,
    'T_orb (d)': 8.463 * u.day,
    'T_rot (d)': 4.863 * u.day,
    'gain': 2.5,
    'inclination': 88.5,
    'eccentricity': 0.0,
    'longitude_of_periastron': 89.0
}

def quadratic_limb_dark_model(filter_response, bandpass_of_interest, Teff=3650, logg=4.52, metallicity=0.12):

    t = filter_response.throughput[0]
    w = filter_response.wave[0].value
    
    a = (w >= bandpass_of_interest.min())
    b = (w <= bandpass_of_interest.max())
    ok_wavelengths = a + b
    throughput = t[ok_wavelengths]
    wavelengths = w[ok_wavelengths]*1000 #convert to nm

    filter = [ldtk.TabulatedFilter(f'{np.nanmedian(w)}micron', wavelengths, throughput)]
    
    sc = LDPSetCreator(teff=(Teff, 50),    # Define your star, and the code
                       logg=(logg, 0.05),    # downloads the uncached stellar
                          z=(metallicity, 0.1),    # spectra from the Husser et al.
                         filters=filter)    # FTP server automatically.
    
    ps = sc.create_profiles()                # Create the limb darkening profiles
    ldcs,ldc_err = ps.coeffs_qd(do_mc=True)         # Estimate quadratic law coefficients  

    return ldcs

def continuum_correction(wavelength, flux, err=None):

    'flux: no units'
    'wavelength: angstrom'
    data_flux = flux * u.dimensionless_unscaled
    data_wave = wavelength
    
    _m = data_flux / np.nanmedian(data_flux)
    spectrum = Spectrum1D(flux=_m, spectral_axis=data_wave)
    
    with warnings.catch_warnings():  # Ignore warnings
        warnings.simplefilter('ignore')
        g1_fit = fit_generic_continuum(spectrum)
        continuum_fit = g1_fit(data_wave)
        
    normed_flux = _m / continuum_fit
    
    if err is not None:
        normed_error = err / flux
        return normed_flux, normed_error

    else:
        return normed_flux

'example to calculate the limb darkening:'
# visit = 'S22'
# # bandpass = np.linspace(1.13,1.66,150)
# bandpass = np.linspace(0.80,1.13,150)
# this_grism = visits[f'{visit}']['filter_response']
# u1, u2 = quadratic_limb_dark_model(filter_response=this_grism, bandpass_of_interest=bandpass)[0]


def calculate_delta_D_spot(parameters, w):
    
    f_cool, f_chord, T_cool, T_amb, transit_depth = parameters
    
    S_cool = get_phoenix_photons(wavelength = w, temperature = T_cool, metallicity = 0.12, logg= 4.52)[1]
    S_amb = get_phoenix_photons(wavelength = w, temperature = T_amb, metallicity = 0.12, logg= 4.52)[1]
    
    flux_ratio = S_cool/S_amb
    top = (1. - f_chord) + (f_chord * flux_ratio)
    bottom = (1. - f_cool) + (f_cool * flux_ratio)
    
    delta_D_spot = ((top / bottom) - 1.) * transit_depth
    
    depth_factor = (delta_D_spot/transit_depth) + 1.

    return delta_D_spot * 1e6

def sinusoidal_baseline(parameters, t):

    A = parameters['A']
    B = parameters['B']
    rotation_period = systeminfo['T_rot (d)'].value
    
    # Calculate the angular frequency
    omega = 2 * np.pi / rotation_period

    baseline = A * np.sin(omega * t) + B * np.cos(omega * t) + 1
    
    return baseline

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