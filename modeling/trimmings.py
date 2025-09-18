
from shone.spectrum import bin_spectrum
import lmfit
from lmfit import Model, Parameters

# G_102_back_dict = scipy.io.readsav('../data/data_from_hannah/G102/Backward_spectra.sav', verbose=False)
# G_102_for_dict = scipy.io.readsav('../data/data_from_hannah/G102/Forward_spectra.sav', verbose=False)
# G_141_back_dict = scipy.io.readsav('../data/data_from_hannah/G141/Backward_spectra.sav', verbose=False)
# G_141_for_dict = scipy.io.readsav('../data/data_from_hannah/G141/Forward_spectra.sav', verbose=False)

# WFC3_Median_Spectra = {
    
#     "F21" : {
        
#         "Forward" : {
#             "w" : None,"f" : None,"e" : None,
#         },
#         "Reverse" : {
#             "w" : None,"f" : None,"e" : None,
#         }
#     },
    
#     "S22" : {

#         "Forward" : {
#             "w" : None,"f" : None,"e" : None,
#         },
#         "Reverse" : {
#             "w" : None,"f" : None,"e" : None,
#         }
#     }
# }

# for visit in tqdm(['F21','S22']):
#     for direction in ['Forward','Reverse']:

#         # Prepare data for both directions
#         exptime = visits[f'{visit}']['exp (s)']
#         binwidth = visits[f'{visit}']['native resolution']
#         grism = visits[f'{visit}']['Grism']
        
#         trimmed_r = read_rainbow(f"../data/{visit}_{direction}_trimmed_pacman_spec.rainbow.npy")
    
#         # Process forward direction data
#         median_spectrum = trimmed_r.get_median_spectrum().value
#         e_per_s = median_spectrum / exptime
#         e_per_s_per_angstrom = e_per_s / binwidth
#         _w, _s, _e = read_sensitivity_curve(grism=grism)
#         binned_filter_response = bintogrid(_w.value, _s.value, newx=trimmed_r.wavelength.value)['y'] * u.cm**2 / u.erg
#         calibrated_data_flux = e_per_s_per_angstrom / binned_filter_response
            
#         WFC3_Median_Spectra[f'{visit}'][f'{direction}']['w'] = trimmed_r.wavelength
#         WFC3_Median_Spectra[f'{visit}'][f'{direction}']['f'] = calibrated_data_flux
#         WFC3_Median_Spectra[f'{visit}'][f'{direction}']['e'] = 0.005 * calibrated_data_flux

# F21F_rainbow = read_rainbow(f"../data/F21_Forward_trimmed_pacman_spec.rainbow.npy")
# F21R_rainbow = read_rainbow(f"../data/F21_Reverse_trimmed_pacman_spec.rainbow.npy")
# S22F_rainbow = read_rainbow(f"../data/S22_Forward_trimmed_pacman_spec.rainbow.npy")
# S22R_rainbow = read_rainbow(f"../data/S22_Reverse_trimmed_pacman_spec.rainbow.npy")

# File paths for Hannah's white light curves
# datasets = {
#     "F21_Backward": {
#         "lightcurve": "../data/AUMicb_F21_Backward_lightcurve_data.txt",
#     },
#     "F21_Forward": {
#         "lightcurve": "../data/AUMicb_F21_Forward_lightcurve_data.txt",
#     },
#     "S22_Backward": {
#         "lightcurve": "../data/AUMicb_S22_Backward_lightcurve_data.txt",
#     },
#     "S22_Forward": {
#         "lightcurve": "../data/AUMicb_S22_Forward_lightcurve_data.txt",
#     }
# }




# def phoenix_1T(parameters, sigma, convolution_method='astropy', kernel_type='astropy'):

#     T_phot = parameters['T_phot']
#     rstar = parameters['R_star']
#     log_g = calculate_logg(rstar)
    
#     S_phot = get_phoenix_photons(wavelength = model_wavelengths,temperature = float(T_phot), metallicity = 0.12, logg=log_g)

#     E_per_photon = (3e10*6.626e-27)/(S_phot[0] * 1e-4) #ergs per photon
#     _f = S_phot[1] / (1e5 * u.angstrom * u.cm * u.cm * u.s) # converted from the phoenix units of photons/nm/m^2/s
#     _f = (_f * E_per_photon) * u.erg # now the model spectrum is in flux calibrated units
#     exclude_nans = ~np.isnan(_f)
#     model_wave = S_phot[0].value[exclude_nans]
#     model_flux = _f.value[exclude_nans]
    
#     # Convolve the model spectrum
#     convolved = convolve_spectrum(model_wave, model_flux, sigma=sigma, method = convolution_method, kernel_type = kernel_type)
    
#     return convolved

# def phoenix_2T(parameters, sigma, convolution_method='astropy', kernel_type='astropy'):

#     T_phot = parameters['T_phot']
#     T_spot = parameters['T_spot']
#     f_spot = parameters['f_spot']
#     rstar = parameters['R_star']
#     log_g = calculate_logg(rstar)
    
#     S_spot = get_phoenix_photons(wavelength = model_wavelengths,temperature = float(T_spot), metallicity = 0.12, logg = log_g)
#     S_phot = get_phoenix_photons(wavelength = model_wavelengths,temperature = float(T_phot), metallicity = 0.12, logg = log_g)

#     # Calculate model spectrum
#     E_per_photon = (3e10*6.626e-27)/(S_phot[0] * 1e-4) #ergs per photon
#     _f = (f_spot*S_spot[1] + (1-f_spot)*S_phot[1]) / (1e5 * u.angstrom * u.cm * u.cm * u.s) # converted from the phoenix units of photons/nm/m^2/s
#     _f = (_f * E_per_photon) * u.erg # now the model spectrum is in flux calibrated units
#     exclude_nans = ~np.isnan(_f)
#     model_wave = S_spot[0].value[exclude_nans]
#     model_flux = _f.value[exclude_nans]
    
#     # Convolve the model spectrum
#     convolved = convolve_spectrum(model_wave, model_flux, sigma=sigma, method = convolution_method, kernel_type = kernel_type)
    
#     return convolved

# def phoenix_3T(parameters, sigma, convolution_method='astropy', kernel_type='astropy'):

#     T_phot = parameters['T_phot']
#     T_spot = parameters['T_spot']
#     T_other = parameters['T_other']
#     f_spot = parameters['f_spot']
#     f_phot = parameters['f_phot']
#     rstar = parameters['R_star']
    
#     f_other = 1.0 - (f_spot + f_phot)
#     log_g = calculate_logg(rstar)
    
#     S_spot = get_phoenix_photons(wavelength = model_wavelengths,temperature = float(T_spot), metallicity = 0.12, logg= log_g)
#     S_phot = get_phoenix_photons(wavelength = model_wavelengths,temperature = float(T_phot), metallicity = 0.12, logg= log_g)
#     S_other = get_phoenix_photons(wavelength = model_wavelengths,temperature = float(T_other), metallicity = 0.12, logg = log_g)

#     # Calculate model spectrum
#     E_per_photon = (3e10*6.626e-27)/(S_phot[0] * 1e-4) #ergs per photon
#     _f = (f_spot*S_spot[1] + f_phot*S_phot[1] + f_other * S_other[1]) / (1e5 * u.angstrom * u.cm * u.cm * u.s) # converted from the phoenix units of photons/nm/m^2/s
#     _f = (_f * E_per_photon) * u.erg # now the model spectrum is in flux calibrated units
#     exclude_nans = ~np.isnan(_f)
#     model_wave = S_spot[0].value[exclude_nans]
#     model_flux = _f.value[exclude_nans]
    
#     # Convolve the model spectrum
#     convolved = convolve_spectrum(model_wave, model_flux, sigma=sigma, method = convolution_method, kernel_type = kernel_type)
    
#     return convolved

# def btsettl_1T(parameters, sigma, convolution_method='astropy', kernel_type='astropy'):

#     T_phot = parameters['T_phot']
#     S_phot = btsettl_grid(float(T_phot))

#     # Calculate combined spectrum
#     _f = S_phot
#     exclude_nans = ~np.isnan(_f)
#     model_wave = btsettl_wavelengths[exclude_nans]
#     model_flux = _f[exclude_nans]
    
#     # Convolve the model spectrum
#     convolved = convolve_spectrum(model_wave, model_flux, sigma=sigma, method = convolution_method, kernel_type = kernel_type)
    
#     return convolved

# def btsettl_2T(parameters, sigma, convolution_method='astropy', kernel_type='astropy'):

#     f_spot = parameters['f_spot']
#     T_spot = parameters['T_spot']
#     T_phot = parameters['T_phot']
#     S_spot = btsettl_grid(float(T_spot))
#     S_phot = btsettl_grid(float(T_phot))

#     # Calculate combined spectrum
#     _f = (f_spot*S_spot + (1.0-f_spot)*S_phot)
#     exclude_nans = ~np.isnan(_f)    
#     model_wave = btsettl_wavelengths[exclude_nans]
#     model_flux = _f[exclude_nans]
    
#     # Convolve the model spectrum
#     convolved = convolve_spectrum(model_wave, model_flux, sigma=sigma, method = convolution_method, kernel_type = kernel_type)
    
#     return convolved

# def btsettl_3T(parameters, sigma, convolution_method='astropy', kernel_type='astropy'):

#     T_phot = parameters['T_phot']
#     T_spot = parameters['T_spot']
#     T_other = parameters['T_other']
#     f_other = parameters['f_other']
#     f_spot = parameters['f_spot']
#     f_phot = 1.0 - (f_other + f_spot)
    
#     S_spot = btsettl_grid(float(T_spot))
#     S_phot = btsettl_grid(float(T_phot))
#     S_other = btsettl_grid(float(T_other))
    
#     # Calculate model spectrum
#     _f = (f_spot*S_spot + f_phot*S_phot + f_other*S_other)
#     exclude_nans = ~np.isnan(_f)
#     model_wave = btsettl_wavelengths[exclude_nans]
#     model_flux = _f[exclude_nans]
    
#     # Convolve the model spectrum
#     convolved = convolve_spectrum(model_wave, model_flux, sigma=sigma, method = convolution_method, kernel_type = kernel_type)
    
#     return convolved

# # Function to load lightcurve data
# def load_lightcurve(file_path, **kwargs):
#     return pd.read_csv(
#         file_path, 
#         sep=r'\s+', 
#         comment='#', 
#         names=["MJD", "Flux", "Uncertainty", "Shift"]
#     )

# Example: Load all sh datasets 
# sh_data = {}
# for key, paths in datasets.items():
#     sh_data[key] = {
#         "lightcurve": load_lightcurve(paths["lightcurve"])
#     }

# def get_BTSettl_spectrum(T, **kwargs):

#     gridspec = btsettl_grid(jnp.array(T, dtype=jnp.float64))
#     lamb = btsettl_wavelengths * u.micron 

#     flux = gridspec * u.Unit('erg cm-2 s-1 AA-1') 

#     spec = Spectrum1D(spectral_axis=lamb, flux=flux)
    
#     return spec

# @jit
# def convolve_spectrum_jax(model_wavelength, model_flux, sigma, method='fft', kernel_type='gaussian'):
#     """
#     JAX-compatible spectral convolution with a Gaussian kernel.
    
#     Args:
#         model_wavelength: Wavelength array (must be uniformly spaced for accuracy).
#         model_flux: Flux array to convolve.
#         sigma: Standard deviation of Gaussian kernel (in pixels).
#         method: 'fft' (fast, default) or 'direct' (slower but more precise for small kernels).
#         kernel_type: 'gaussian' (default) or 'custom' (user-provided kernel).
    
#     Returns:
#         Convolved flux array (same shape as input).
#     """
#     # Ensure inputs are JAX arrays
#     model_wavelength = jnp.asarray(model_wavelength)
#     model_flux = jnp.asarray(model_flux)
#     sigma = jnp.asarray(sigma)

#     # Generate Gaussian kernel
#     if kernel_type == 'gaussian':
#         # Create kernel centered at 0 with stddev `sigma`
#         kernel_size = jnp.minimum(10 * sigma, len(model_flux))  # Auto-size kernel
#         x = jnp.arange(-kernel_size // 2, kernel_size // 2 + 1)
#         kernel = jnp.exp(-0.5 * (x / sigma)**2)
#         kernel = kernel / jnp.sum(kernel)  # Normalize
    
#     elif kernel_type == 'custom':
#         raise NotImplementedError("Custom kernels must be pre-computed and passed.")
    
#     else:
#         raise ValueError(f"Unsupported kernel_type: {kernel_type}")

#     # Perform convolution
#     if method == 'fft':
#         convolved = jax_convolve(model_flux, kernel, mode='same', method='fft')
#     elif method == 'direct':
#         convolved = jax_convolve(model_flux, kernel, mode='same', method='direct')
#     else:
#         raise ValueError(f"Unsupported method: {method}")

#     return convolved