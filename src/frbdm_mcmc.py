import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from multiprocessing import Pool


import emcee
from scipy.integrate import quad
from scipy.interpolate import UnivariateSpline
import jax.numpy as jnp
from jax import jit
from tqdm import tqdm
from astropy.cosmology import Planck18 as cosmo

brute_force = True

import numpy as np
from scipy.integrate import quad, dblquad
from scipy.integrate import quad_vec
from multiprocessing import Pool
import numba as nb

def generate_TNGparam_arr(zfrb):
    TNGfits = np.load('TNGparameters.npy')
    nfrb = len(zfrb)
    arr = TNGfits
    tngparams_arr = np.zeros([nfrb, 6])
    ztng = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2, 3, 4, 5]

    A = UnivariateSpline(ztng, arr[:, 0], s=0)
    dmx = UnivariateSpline(ztng, arr[:, 1], s=0)
    dmigm = UnivariateSpline(ztng, arr[:, 2], s=0)
    sigx = UnivariateSpline(ztng, arr[:, 3], s=0)
    sigigm = UnivariateSpline(ztng, arr[:, 4], s=0)
    rho = UnivariateSpline(ztng, arr[:, 5], s=0)

    for xx in range(nfrb):
        zz = zfrb[xx]
        Ai = A(zz).ravel()[0]
        dmxi = dmx(zz).ravel()[0]
        dmigmi = dmigm(zz).ravel()[0]
        sigxi = sigx(zz).ravel()[0]
        sigigmi = sigigm(zz).ravel()[0]
        rhoi = rho(zz).ravel()[0]

        tngparams_arr[xx] = np.array([Ai, dmxi, dmigmi, sigxi, sigigmi, rhoi])

    return tngparams_arr

#@njit
def pdmhost(dmhost, mu, sigma):
    # Could reasonably be: pdmhost(dmhost, 3, 1)
    p = 1/(dmhost * np.sqrt(2*np.pi) * sigma)
    p *= np.exp(-(np.log(dmhost) - mu)**2 / (2*sigma**2))
    return p

#@njit
def pdm_cosmic(dmhalo, dmigm, params, TNGparams):
    """ 2D PDF for the IGM and Halo contribution
    
    Parameters
    ----------

    params : list
        [figm, fx]
    dmhalo : array
        Halo DM
    dmigm : array
        IGM DM
    TNGparams : list
        [A, mu_x, mu_y, sigma_x, sigma_y, rho]

    Returns
    -------
    PDF : array
        2D PDF
    """
    figmTNG = 0.837
    fxTNG = 0.125
    x, y = dmhalo, dmigm
    figm, fx = params
    A, mu_x, mu_y, sigma_x, sigma_y, rho = TNGparams
    mu_y += np.log(figm / figmTNG)
    mu_x += np.log(fx / fxTNG)
    term1 = -((np.log(x) - mu_x)**2 / sigma_x**2 + (np.log(y) - mu_y)**2 / sigma_y**2)
    term2 = 2 * rho * (np.log(x) - mu_x) * (np.log(y) - mu_y) / (sigma_x * sigma_y)
    B = (2 * np.pi * sigma_x * sigma_y * x * y * np.sqrt(1 - rho**2))
    return A / B * np.exp((term1 - term2) / (2 * (1 - rho**2)))

#@njit
def pdm_product_numerical(dmhalo, dmigm, dmexgal, 
                zfrb, params, TNGparams): 
    figm, fx, mu, sigma = params
    dmhost = dmexgal - dmhalo - dmigm
    pcosmic = pdm_cosmic(dmhalo, dmigm, (figm, fx), TNGparams)
    phost = pdmhost(dmhost * (1+zfrb), mu, sigma)

    return pcosmic * phost, phost

def pdm_product(dmhalo, dmigm, dmexgal, 
                zfrb, params, TNGparams): 
    figm, fx, mu, sigma = params
    dmhost = dmexgal - dmhalo - dmigm
    pcosmic = pdm_cosmic(dmhalo, dmigm, (figm, fx), TNGparams)
    phost = pdmhost(dmhost * (1+zfrb), mu, sigma)

    return pcosmic * phost

#@njit(parallel=False)
def log_likelihood_one_source(zfrb, dmexgal, params, tngparams):

    p = dblquad(pdm_product, 0, dmexgal,
              lambda x: 0, lambda x: dmexgal-x,
               (dmexgal, zfrb, params, tngparams))[0]

    return np.log(p)


def log_likelihood(zfrb, dmexgal, params, tngparams_arr):
    logP = 0
        
    nfrb = len(zfrb)

    for ii in range(nfrb):
        tngparams = tngparams_arr[ii]

        p = dblquad(pdm_product, 0, dmexgal[ii],
              lambda x: 0, lambda x: dmexgal[ii]-x,
               (dmexgal[ii], zfrb[ii], params, tngparams))[0]

        
        logP += np.log(p)

    return logP

def log_prior(params):
    figm, fx, mu, sigma = params

    if 0.0 < figm < 1.0:
        if 0. < fx < 0.5:
            if 0 < mu < 7:
                if 0.1 < sigma < 2.:
                    return 0
    return -np.inf

def log_probability(params, zfrb, dm, tngparams_arr):
    lp = log_prior(params)
    
    if not np.isfinite(lp):
        return -np.inf

    logLike = log_likelihood(zfrb, dm, params, tngparams_arr)

    return lp + logLike

def worker(figm):
    params = [figm, 0.125, 5, 1]
    return func(params, dmhalo, dmigm, dmexgal, zex, tngparams_arr, dmex)

def log_likelihood_all(params, zfrb, dmfrb, dmhalo, dmigm, dmexgal, zex, tngparams_arr):
    nz, ndm = len(zex), len(dmhalo)
    dmex = dmexgal[0,0]
    
    P = np.empty((ndm, nz))

    for ii in range(len(zex)):        
        pp, dmhost = pdm_product_numerical(dmhalo, dmigm, dmexgal, zex[ii], params, tngparams_arr[ii])
        
        for kk, dd in enumerate(dmex):
            p, dh = pp[:, :, kk], dmhost[:, :, kk]
            P[kk, ii] = np.nansum(p[dh > 0], axis=-1)
    
    P = P / np.nansum(P, 0)
    
    nfrb = len(zfrb)
    
    logP = 0
    
    for nn in range(nfrb):
        ll = np.argmin(np.abs(zfrb[nn] - zex))
        kk = np.argmin(np.abs(dmfrb[nn] - dmex))
        lp = np.log(P[kk, ll])
        
        if np.isnan(lp):
            return -np.inf
        else:
            logP += lp

    return P, logP

def log_prior(params):
    figm, fx, mu, sigma = params

    if 0.0 < figm < 1.2:
        if 0. < fx < 0.25:
            if 0 < mu < 7:
                if 0.01 < sigma < 2.:
                    return 0
            
    return -np.inf

def log_posterior(params, zfrb, dmfrb, dmhalo, dmigm, dmexgal, zex, tngparams_arr):
    
    log_pri = log_prior(params)
    
    if not np.isfinite(log_pri):
        return -np.inf
    
    log_like = log_likelihood_all(params, zfrb, dmfrb, dmhalo, dmigm, dmexgal, zex, tngparams_arr)[1]

    return log_pri + log_like 

def main(data, param_dict):
    zfrb, dmfrb = data

    dmmin, dmmax, ndm = param_dict['dmmin'], param_dict['dmmax'], param_dict['ndm']
    zmin, zmax, nz = param_dict['zmin'], param_dict['zmax'], param_dict['nz']
    dmexmin, dmexmax, ndmex = param_dict['dmexmin'], param_dict['dmexmax'], param_dict['ndmex']
    nmcmc_steps = param_dict['nmcmc_steps']
    nwalkers = param_dict['nwalkers']
    ndim = param_dict['ndim']
    pguess = param_dict['pguess']

    # Generate the figm values
    dmi = np.linspace(dmmin, dmmax, ndm)
    dmh = np.linspace(dmmin, dmmax, ndm)
    dmex = np.linspace(dmexmin, dmexmax, ndm)

    zex = np.linspace(zmin, zmax, nz)

    tngparams_arr = generate_TNGparam_arr(zex)

    dmhalo, dmigm, dmexgal = np.meshgrid(dmh, dmi, dmex)

    pguess = (0.8, 0.15, 5, 1)

    nwalkers = 32
    ndim = 4
    nsamp = 1500
    tngparams_arr = generate_TNGparam_arr(zex)
    pos = pguess + 1e-3 * np.random.randn(nwalkers, ndim)

    with Pool(32) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior,
                                        args=(zfrb, dmfrb, dmhalo,
                                             dmigm, dmexgal, zex, tngparams_arr), pool=pool)
        sampler.run_mcmc(pos, nsamp, progress=True)
        
    flat_samples = sampler.get_chain(discard=0, thin=1, flat=True)
    np.save('flat_samples_figm_%dsteps.npy' % len(flat_samples), flat_samples)

    return flat_samples 

if __name__ == '__main__':
    df = pd.read_csv('../data/dsafrbsnov23.csv', delim_whitespace=False)
    zdsa = df['redshift'].values
    dmdsa = df['dm_exgal'].values
    ind = np.where((zdsa != -1) & (dmdsa > 0) & (np.abs(zdsa) > 0.05))[0]
    zdsa = np.abs(zdsa[ind])
    dmdsa = dmdsa[ind]

    # Start parameters for MCMC chain 
    figm_start, fX_start, mu_start, sigma_start = 0.8, 0.15, 5, 1

    param_dict = {'dmmin': 0, 
                  'dmmax': 5000, 
                  'ndm': 100,
                  'zmin': 0, 
                  'zmax': 2, 
                  'nz': 100,
                  'dmexmin': 0, 
                  'dmexmax': 5000, 
                  'ndmex': 100,
                  'nmcmc_steps' : 1500,
                  'nwalkers' : 32,
                  'ndim' : 4,
                  'pguess' : (figm_start, fX_start, mu_start, sigma_start),                
                  }
    
    data = (zdsa, dmdsa - 30.)

    flat_samples = main(data, param_dict=param_dict)