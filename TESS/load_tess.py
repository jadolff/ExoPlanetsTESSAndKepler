### Imports and Constants ###
# Imports
import numpy as np
import pandas as pd
import lightkurve as lk
from astropy.io import fits
from typing import NamedTuple
from constants import *

# Constants
from astropy.constants import R_earth, R_sun
R_Sun_over_R_Earth = float( R_sun / R_earth )
url = 'https://mast.stsci.edu/api/v0.1/Download/file?uri='

# Transit
from planet import transit


### Loading Star Data ###
class StarInfo_tess(NamedTuple):
    kepid: int # tess id of the star - we call it kepid for legacy compatability
    q: float # related to stellar density, transit duration = q period^1/3
    err_q: float # relative error of q
    u1: float # limb darkening parameters
    u2: float
    sigma_star_flux_units: float # variance of the Gaussian component of the noise in units where star's flux is 1.
    radius: float
    radius_errp: float
    radius_errm: float

def u1u2_from_stellar_properties_tess(logg, feh, Teff):
    
    # Get u1u2 table
    df = pd.read_csv('TESS/result.csv')
    
    # convert this stellar parameters so that they fit on a latttice where the calculations are available
    # loggL, fehL, TeffL = logg_feh_Teff_lattice(logg, feh, Teff) 
    # look up the calculated values of u1 and u2
    # df = df.loc[(df['Teff'] == TeffL) & (df['logg'] == loggL) & (df['Z'] == fehL)]  

    df["dist"] = (
        (df["Teff"] - Teff).abs()
        + (df["logg"] - logg).abs()
        + (df["Z"] - feh).abs()
    )
    df = df.loc[df["dist"].idxmin()]

    if (3500.0 <= Teff <= 50000.0) and (0.0 <= logg <= 5.0) and (-5.0 <= feh <= 1.0) :
        ret = df[['aLSM', 'bLSM']].to_numpy(dtype=float)

    else :
        ret = [0, 0]
        print(logg, feh, Teff)

    return ret

def StarInfo_init_tess(tessid, name = 'star_info_withplanets'):
    
    df_all = pd.read_csv("TESS/Gaia_TESS.csv")
    data_star = df_all.loc[df_all['id_starname']==('tic' + str(tessid))]
    

    ## Get Stellar Properties
    mass = data_star['iso_mass'].item()
    mass_err = np.maximum(data_star['iso_mass_err1'].item(), 
                          -data_star['iso_mass_err2'].item()) / mass

    radius = data_star['iso_rad'].item()
    radius_err = np.maximum(data_star['iso_rad_err1'].item(),
                           -data_star['iso_rad_err2'].item()) / radius
    radius_err_high = data_star['iso_rad_err1'].item() / radius
    radius_err_low = data_star['iso_rad_err2'].item() / radius

    logg = data_star['iso_logg'].item()
    feh = data_star['iso_feh'].item()
    teff = data_star['iso_teff'].item()

    q_sun = 6.957e8 / np.cbrt(2 * np.pi * (1.989e30 * 6.673e-11) * ((3600 * 24) ** 2))
    q = q_sun * radius / np.cbrt(mass) # radius and mass are in units of sun's mass
    q_err = np.sqrt(radius_err ** 2 + (mass_err / 3.0) ** 2) # all errors are relative errors
    
    ## Get u1, u2
    u1, u2 = u1u2_from_stellar_properties_tess(logg, feh, teff)

    ## Return StarInfo
    return StarInfo_tess(tessid, 
                         q, 
                         q_err, 
                         u1, 
                         u2,
                         0.,
                         radius,
                         radius_err_high,
                        -radius_err_low,
                        )



### Loading Fluxes ###
def read_tess_light_curve(tessid, n_bins = 1, PDCSAP = True, invert=False):
  """Reads time and flux measurements for a TESS target star.
  Args:
    filenames: A list of .fits files containing time and flux measurements.
    invert: Whether to invert the flux measurements by multiplying by -1.
  Returns:
    all_time: A list of numpy arrays; the time values of the light curve.
    all_flux: A list of numpy arrays corresponding to the time arrays in
        all_time.
  """
  all_time = []
  all_flux = []
  all_flags = []

  search_result = lk.search_lightcurve('TIC ' + str(tessid), mission="TESS", 
                                       author=["SPOC", "TESS-SPOC"], exptime=120)


  for res in search_result :
    uri = res.table.as_array()['dataURI'].data[0]
    with fits.open(url + uri, mode="readonly") as hdulist:
        
        time = hdulist[1].data['TIME']
        flags = hdulist[1].data['QUALITY']

        if PDCSAP:
            flux = hdulist[1].data['PDCSAP_FLUX']
        else:
            flux = hdulist[1].data['SAP_FLUX']

    if n_bins != 1 : 
        # Average over bins of size n_bins to decrease computational cost
        n = len(time); diff = n % n_bins
        mask = np.concatenate([np.ones(n - diff, dtype=bool), np.zeros(diff, dtype=bool)])

        time = time[mask].reshape(-1, n_bins).mean(axis=1)
        flux = flux[mask].reshape(-1, n_bins).mean(axis=1)
        flags = flags[mask].reshape(-1, n_bins).mean(axis=1)

    # Remove NaN flux values.
    valid_indices = np.where(np.isfinite(flux))
    time = time[valid_indices]
    flux = flux[valid_indices]
    flags = flags[valid_indices]

    if invert:
      flux *= -1

    if time.size :

      all_time.append(time)
      all_flux.append(flux)
      all_flags.append(flags)

    # # Use only first batch    
    # break

  return all_time, all_flux, all_flags


def read_lc_tess(tessid, injected = False):
    """reads the downloaded lightcurbes.
    retrurns concatenated time, flux from all quarters, TESS flags (kind of useless) and a list of quarter borders in days"""
    
    all_time, all_flux, all_flags = read_tess_light_curve(tessid)
    for i in range(len(all_time)):
        all_flux[i] = (all_flux[i] / np.average(all_flux[i])) - 1

    Time = np.concatenate(all_time)
    Flux = np.concatenate(all_flux)
    Flags = np.concatenate(all_flags) 
    
    # [time(quarter begining) for each quarter]
    quarter_beginnings = np.array([0.5*(all_time[i-1][-1] + all_time[i][0]) - all_time[0][0] for i in range(1, len(all_time))]) 
    
    return Time, Flux, Flags, quarter_beginnings

def read_alternative(tessid, bins, injected = False):
    """reads the downloaded lightcurbes.
    retrurns concatenated time, flux from all quarters, TESS flags (kind of useless) and a list of quarter borders in days"""
    
    all_time, all_flux, all_flags = read_tess_light_curve(tessid, bins)
    for i in range(len(all_time)):
        all_flux[i] = (all_flux[i] / np.average(all_flux[i])) - 1
    
    # [time(quarter begining) for each quarter]
    quarter_beginnings = np.array([0.5*(all_time[i-1][-1] + all_time[i][0]) - all_time[0][0] for i in range(1, len(all_time))]) 
    
    return all_time, all_flux, all_flags, quarter_beginnings



### Adding Zeros ###
def add_zeros_tess(time, Flux, hyperparametrs, export_path = None, zero_paddling= 2000):
    """Finds parts of signal where there are missing measurements and adds 0 flux and infinite variance there. 
    assumes variance si 1 if we have a measurement"""
    # reads data
    # df = pd.read_csv(import_path)
    # Flux = np.array(df["Flux"])
    # time = np.array(df["Time"])

    #normalizes times and finds where there is missing data
    difference_Time = (time[1:] - time[:-1])
    difference_Time /= dt  # dimensionless t[i+1]-t[i]
    difference_Time = np.round(difference_Time, 0).astype(int)
    mask = difference_Time > 1
    indexes = (np.arange(len(difference_Time), dtype = int)+1)[mask]
    add = difference_Time[mask] - 1
    invVar = np.ones(len(Flux))

    #fills with zeros where there is no data and fixes invVar there to infinity
    indexes_all_repeated = []
    for i in range(len(indexes)):
        indexes_all_repeated = indexes_all_repeated + [indexes[i] for j in range(add[i])]

    num_points = len(Flux) + len(indexes_all_repeated)
    num_points_all = num_points + zero_paddling # we add some zero padding to prevent spectral leaking
    num_points_all += hyperparametrs.stft_sep - (num_points_all - hyperparametrs.stft_width) % hyperparametrs.stft_sep # we make sure that the final time length is compatible with STFT windows
    num_zero_padding = num_points_all - num_points
    
    indexes_all_repeated = indexes_all_repeated + [len(Flux), ] * num_zero_padding
    Flux = np.insert(Flux, indexes_all_repeated, np.zeros(len(indexes_all_repeated)))
    invVar = np.insert(invVar, indexes_all_repeated, np.zeros(len(indexes_all_repeated)))
    Time = time[0] + dt * np.arange(len(Flux))

    #prints in a file
    if export_path != None:
        d = {'Flux': Flux, 'Time': Time, 'invVar': invVar}
        df_out = pd.DataFrame(data=d)
        df_out.to_csv(export_path, index=False)

    return Time, Flux, invVar


### Load Known Planet Properties ###
def read_known_planets_tess(star, t_start, test_recovery = True):
    """Load the known data for the Kepler's confirmed planets.
        Args:
            star: pandas DataFrame, containing the properties of the star
            t_start: Time[0] before it was subtracted out
            test_recovery: if True, some known planets will be skipped, so that we can test our pipeline if it recovers them
            
        Returns:
            properties of the planets: props[i] = {'spline': spline_object, 'ttv found': 0, 'params': [period, phase, tau], 'snr': float}
    """
    
    dfp = pd.read_csv("TESS/tois.csv")
    if (np.sum(dfp['TIC ID'] == star.kepid) == 0): # there are no known planets for that star (or we do not want to eliminate them)
        return []
    
    dfp = dfp[dfp['TIC ID'] == star.kepid]
    
    dfp = dfp[dfp['TFOPWG Disposition'].isin(['PC', 'CP', 'KP'])]
    
    ## TO DO: Find such a data set for TESS
    # # do not include the candidates that are being tested
    # if test_recovery:
    #     df_tested = pd.read_csv(home + 'load/datasets/test_recovery_small.csv')
    #     #df_tested = pd.read_csv(home + 'load/datasets/test_recovery_steve.csv')
    #     dfp = dfp[~dfp['kepoi_name'].isin(df_tested['KOI'])]
    
    period, phase = dfp['Period (days)'], dfp['Epoch (BJD)'] - (t_start + 2457000)
    phase = np.mod(phase, period)
    ror = dfp['Planet Radius (R_Earth)'] / (star.radius * R_Sun_over_R_Earth) 

    ## TO DO: Figure out conversion factor ~~> why 0.5 / 24 ???
    planet_props = np.array([period, phase, dfp['Duration (hours)'] * 0.5 / 24., 
                             ror, dfp['Planet SNR']]).T

    available = np.all(np.isfinite(planet_props), axis= 1)
    planet_props = planet_props[available]

    perm = np.argsort(-planet_props[:, -1]) # sort by decreasing SNR
    planet_props = planet_props[perm, :]

    splines = [transit.prepare_shape(rr, star.u1, star.u2) for rr in planet_props[:, 3]]

    return [transit.Info(planet_props[i, -1], planet_props[i, :3], splines[i], True) for i in range(len(planet_props))]


### q TESS ###
def q_in_dt_units_tess(q):
    return q * np.power(dt, -2./3.)