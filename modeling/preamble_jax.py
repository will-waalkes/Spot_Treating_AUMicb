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
# from shone.opacity.dace import download_molecule
# from shone.chemistry import FastchemWrapper
# from shone.opacity import Opacity
# from shone.transmission import de_wit_seager_2013

import fleck
from fleck.jax import ActiveStar

# from specutils.manipulation import box_smooth, gaussian_smooth, trapezoid_smooth
# from specutils.spectra import Spectrum1D, SpectralRegion

# import scipy.io
# from scipy.optimize import fmin_powell, curve_fit
# from scipy.interpolate import interp1d
# from scipy.signal import correlate

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, to_hex

# from IPython.display import display
# from ipywidgets import interactive, VBox, HBox, FloatSlider

# from astropy.convolution import convolve, convolve_fft
# from astropy.convolution import Gaussian1DKernel
import astropy.units as u
import astropy.constants as c
from astropy.constants import G, m_p
from astropy.table import Table

import pandas as pd
import random
import os
import corner
from chromatic import *
from svo_filters import svo
import numpy as np
import arviz
import arviz as az
from bt_settl import get_interp_stellar_spectrum

import pickle
from collections import defaultdict

from tqdm.auto import tqdm 
from jax import device_get, jit, lax, local_device_count, pmap, random, vmap

from functools import partial
from numpyro.infer.util import initialize_model
from numpyro.infer import MCMC, log_likelihood
from numpyro.util import fori_collect
from numpyro.infer.hmc import hmc
from datetime import datetime

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
    breathing = 1. + (b1 * phase) + (b2 * phase**2.) + (b3 * phase**3.) + (b4 * phase**4.)
    # breathing = _breathing/jnp.mean(_breathing)
    
    return jnp.array(breathing)

@jit
def ramp_model_jax(phase, r1, r2, r3):

    phase = jnp.array(phase)
    ramp = 1. - jnp.exp( (-r1 * phase) + r2) + (r3 * phase)

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

rng_seed = 0

def hstack_recursive(final_states, checkpoint_states):
    for key in final_states.keys():
        if isinstance(final_states[key], dict):
            hstack_recursive(final_states[key], checkpoint_states[key])
        else:
            final_states[key] = jnp.hstack([
                final_states[key], 
                checkpoint_states[key]
            ])

def print_big_message(big_message):
    print('\n\n')
    print('=' * len(big_message))
    print(big_message)
    print('=' * len(big_message))
    print('\n\n')

class MCMCWithCheckpoints(MCMC):
    running_states = None
    checkpoint = 0
    start_time = None
    
    def run_checkpoints(self, rng_key, *args, extra_fields=(), n_checkpoints=10, 
                        progress_bar_warmup=True, progress_bar_samples=True, 
                        init_params=None, on_checkpoint=None, **kwargs):
        """
        Run the MCMC samplers and collect samples.

        :param random.PRNGKey rng_key: Random number generator key to be used for the sampling.
            For multi-chains, a batch of `num_chains` keys can be supplied. If `rng_key`
            does not have batch_size, it will be split in to a batch of `num_chains` keys.
        :param args: Arguments to be provided to the :meth:`numpyro.infer.mcmc.MCMCKernel.init` method.
            These are typically the arguments needed by the `model`.
        :param extra_fields: Extra fields (aside from `"z"`, `"diverging"`) from the
            state object (e.g. :data:`numpyro.infer.hmc.HMCState` for HMC) to be collected
            during the MCMC run. Note that subfields can be accessed using dots, e.g.
            `"adapt_state.step_size"` can be used to collect step sizes at each step. Exclude sample sites from
            collection with "~`sampler.sample_field`.`sample_site`". e.g. "~z.a" will prevent site "a" from
            being collected if you're using the NUTS sampler. To collect samples of a site "a" in the
            unconstrained space, we can specify the variable here, e.g. `extra_fields=("z.a",)`.
        :type extra_fields: tuple or list of str
        :param init_params: Initial parameters to begin sampling. The type must be consistent
            with the input type to `potential_fn` provided to the kernel. If the kernel is
            instantiated by a numpyro model, the initial parameters here correspond to latent
            values in unconstrained space.
        :param kwargs: Keyword arguments to be provided to the :meth:`numpyro.infer.mcmc.MCMCKernel.init`
            method. These are typically the keyword arguments needed by the `model`.

        .. note:: jax allows python code to continue even when the compiled code has not finished yet.
            This can cause troubles when trying to profile the code for speed.
            See https://jax.readthedocs.io/en/latest/async_dispatch.html and
            https://jax.readthedocs.io/en/latest/profiling.html for pointers on profiling jax programs.
        """
        self.start_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
        num_warmup_total = int(self.num_warmup)
        num_samples_total = int(self.num_samples)
        
        check_point_indices = [
            jnp.arange(num_warmup_total), 
            *jnp.array_split(jnp.arange(num_samples_total), n_checkpoints)
        ]
        n_checkpoints = len(check_point_indices)
        rng_keys = random.split(rng_key, n_checkpoints)
        pbar = tqdm(enumerate(zip(rng_keys, check_point_indices)), total=n_checkpoints)
        for checkpoint, (rng_key, bounds) in pbar:
            if checkpoint == 0:
                self.progress_bar = progress_bar_warmup
                pbar.set_description('Run warmup')
                print_big_message("Begin warmup")
                self.warmup(rng_key, *args, extra_fields=extra_fields, init_params=init_params, **kwargs)
                print_big_message(f"Begin {num_samples_total} samples with {n_checkpoints} checkpoints")

            else:
                pbar.set_description(f'Run samples {bounds.min()} to {bounds.max()}')

                self.progress_bar = progress_bar_samples
                self.num_samples = bounds.size
                self.run(rng_key, *args, extra_fields=extra_fields, init_params=init_params, **kwargs)

                # add to running states:
                if self.running_states is None:
                    self.running_states = dict(self._states)
                else:
                    hstack_recursive(self.running_states, self._states)
                
                # ensure that calls to `self.get_samples` will build a new samples array
                # out of the running states:
                self._states_flat = None
                self._states = self.running_states

                if on_checkpoint is not None:
                    on_checkpoint(self, **kwargs)
                self.checkpoint += 1
        
        pbar.close()

        # reset to total number for arviz IO
        self.num_samples = num_samples_total

def post_batch_viz_save(self, **kwargs):
    """
    here we define some tasks to do after each completed checkpoint:
    """
    print(f'Corner for checkpoint {self.checkpoint}')
    samples_cumulative = self.get_samples()
    corner.corner(samples_cumulative)

    plt.suptitle(f'checkpoint {self.checkpoint}')
    plt.savefig(f'chkpt_{self.checkpoint}_corner.png',dpi=200)
    plt.show()

    with open(f'samples_cumulative_{self.start_time}_checkpoint_{self.checkpoint:04d}.pkl', 'wb') as file:
        pickle.dump(dict(samples_cumulative), file)

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
    "Rs":0.82,
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

jointvisit_speclc_bin_edges = jnp.array([0.805,
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
1.135,
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