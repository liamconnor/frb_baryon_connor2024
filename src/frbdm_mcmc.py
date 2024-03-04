import os
import h5py

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
from scipy.integrate import dblquad
from multiprocessing import Pool
import numba as nb

def generate_TNGparam_arr(zfrb):
    """ Read in TNG300 parameter fits from Walker et al. 2023 
    and interpolate to the redshifts of the FRBs. These 
    parameters fit a 2D logNormal distribution in 
    the IGM and halo contributions to the FRB DM.
    """
    TNGfits = np.load('/home/connor/TNG300-1/TNGparameters.npy')
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
    """ Log-normal PDF for the host DM."""
    prob = 1/(dmhost * np.sqrt(2*np.pi) * sigma)
    prob *= np.exp(-(np.log(dmhost) - mu)**2 / (2*sigma**2))
    return prob

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
    """ Product of the cosmic and host DM PDFs
    """
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
    """ Log likelihood for a single FRB, but compute using 
    the 2D integration function dblquad rather than 
    summing over a meshgrid of DM values."""
    p = dblquad(pdm_product, 0, dmexgal,
              lambda x: 0, lambda x: dmexgal-x,
               (dmexgal, zfrb, params, tngparams))[0]

    return np.log(p)

def log_likelihood_all_dblquad(zfrb, dmexgal, params, tngparams_arr):
    logP = 0
        
    nfrb = len(zfrb)

    for ii in range(nfrb):
        tngparams = tngparams_arr[ii]

        p = dblquad(pdm_product, 0, dmexgal[ii],
              lambda x: 0, lambda x: dmexgal[ii]-x,
               (dmexgal[ii], zfrb[ii], params, tngparams))[0]

        
        logP += np.log(p)

    return logP

def log_likelihood_all(params, zfrb, dmfrb, dmhalo, 
                       dmigm, dmexgal, zex, tngparams_arr):
    """ Log likelihood for all FRBs
    """
    nz, ndm = len(zex), len(dmhalo)
    dmex = dmexgal[0,0]
    
    P = np.empty((ndm, nz))

    # Iterate over FRB redshifts and compute the likelihood
    # in DM at that redshift for those baryon parameters
    for ii in range(len(zex)):        
        pp, dmhost = pdm_product_numerical(dmhalo, dmigm, dmexgal, 
                                           zex[ii], params, tngparams_arr[ii])
        
        for kk, dd in enumerate(dmex):
            p, dh = pp[:, :, kk], dmhost[:, :, kk]
            P[kk, ii] = np.nansum(p[dh > 0], axis=-1)
    
    # Normalize the likelihoods on a per-redshift basis
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
    """ Logarithmic priors on the baryon parameters

    Parameters
    ----------
    params : list
        [figm, fx, mu, sigma]
        figm - fraction of baryons in IGM
        fx - fraction of baryons in halos
        mu, sigma - parameters for the log-normal 
                    distribution of the host DM
    
    Returns
    -------
    0 if the parameters are within the prior range
    -np.inf if the parameters are outside the prior range
    """
    figm, fx, mu, sigma = params

    if 0.0 < figm < 1.1:
        if 0. < fx < 1:
            if 0 < mu < 9:
                if 0.01 < sigma < 2.5:
                    if figm + fx < 1.1:
                        return 0
            
    return -np.inf

def log_posterior(params, zfrb, dmfrb, dmhalo, dmigm,
                  dmexgal, zex, tngparams_arr):
    """ Log posterior for the baryon parameters. First 
    check if the parameters are within the prior range, 
    then compute the log likelihood."""
    log_pri = log_prior(params)
    
    if not np.isfinite(log_pri):
        return -np.inf
    
    log_like = log_likelihood_all(params, zfrb, dmfrb, dmhalo, 
                                  dmigm, dmexgal, zex, 
                                  tngparams_arr)[1]
    if not np.isfinite(log_like):
        return -np.inf
    
    return log_pri + log_like 

def pdmigm_mcquinn(dm, figm, C0, sigma, dmigm_allbaryons,
           A=1., alpha=3., beta=3.):
    """
    PDF(Delta) following the McQuinn formalism describing the DM_cosmic PDF

    See Macquart+2020 for details

    Args:
        Delta (float or np.ndarray):
            DM / averageDM values
        C0 (float):
            parameter
        sigma (float):
        A (float, optional):
        alpha (float, optional):
        beta (float, optional):

    Returns:
        float or np.ndarray:

    """
    dmmean = figm * dmigm_allbaryons
    Delta = dm / dmmean
    p = A * np.exp(-(Delta**(-alpha) - C0) ** 2 / (
            2 * alpha**2 * sigma**2)) * Delta**(-beta)

    return p

def get_params_zhang(zfrb):
    A = A_spl(zfrb)
    C0 = C0_spl(zfrb)
    sigma = sigmaDM_spl(zfrb)
    return A, C0, sigma

def pdm_product(dmhalo, dmigm, dmexgal, 
                zfrb, params, TNGparams): 
    figm, fx, mu, sigma = params
    dmhost = dmexgal - dmhalo - dmigm
    pcosmic = pdm_cosmic(dmhalo, dmigm, (figm, fx), TNGparams)
    phost = pdmhost(dmhost * (1+zfrb), mu, sigma)

    return pcosmic * phost

def prod_prob_mcquinn(dmigm, zfrb, dm, params, tngparams, dmigm_allbaryons):
    A, C0, sigmaigm = tngparams
    figm, F, mu_h, sigma_h = params
    dmhost = dm - dmigm
    sigma_igm = F / np.sqrt(zfrb)
    prod = pdmhost(dmhost * (1+zfrb), mu_h, sigma_h) * pdmigm_mcquinn(dm, figm, C0, sigma_igm, dmigm_allbaryons)
    return prod

def log_likelihood_all_mcquinn(params, zfrb, dmexgal, 
                   IGMparams, dmigm_allbaryons_arr):

    A, C0, sigmaigm = IGMparams
    
    if type(zfrb)==np.float64:
        p = quad(prod_prob_mcquinn, 0, dmexgal, (zfrb, dmexgal, 
                                        (A, C0, sigmaigm), 
                                        dmigm_allbaryons_arr,
                                             params))[0]
        return np.log(p)
    
    nfrb = len(zfrb)
    logP = 0
    
    for kk in range(nfrb):
#         p = dblquad(prod_prob_mcquinn, 0, dmexgal[kk], 
#                     lambda x: 0, lambda x: dmexgal[kk] - x, 
#                     (zfrb[kk], dmexgal[kk], 
#                     (A[kk], C0[kk], sigmaigm[kk]), 
#                     dmigm_allbaryons_arr[kk],
#                     params))[0]
        
        p = quad(prod_prob_mcquinn, 0, dmexgal[kk], (zfrb[kk], dmexgal[kk], 
                                                     params,
                                        (A[kk], C0[kk], sigmaigm[kk]), 
                                        dmigm_allbaryons_arr[kk],
                                             ))[0]
        
        logP += np.log(p)
    
    return logP

def log_prior_mcquinn(params):
    figm, F, mu_h, sigma_h = params

    if 0 < figm < 2:
        if 0.1 < mu_h < 9:
            if 0.1 < sigma_h < 2.:
                if 0.0 < F < 2.:
                    return 0
            
    return -np.inf

def log_posterior_mcquinn(params, zfrb, dmex, IGMparams, 
                          dmigm_allbaryons_arr):
    lp = log_prior_mcquinn(params)
    
    if not np.isfinite(lp):
        return -np.inf
    
    log_like = log_likelihood_all_mcquinn(params, zfrb, dmex, 
                               IGMparams, dmigm_allbaryons_arr)

    return lp + log_like

def main_mcquinn(data, mcmc_filename='test.h5'):
    # Start parameters for MCMC chain 
    figm_start, F_start, mu_start, sigma_start = 1.0, 0.5, 5, 1

    param_dict = {'dmmin': 0, 
                  'dmmax': 1700, 
                  'ndm': 100,
                  'zmin': 0, 
                  'zmax': 1.5, 
                  'nz': 100,
                  'dmexmin': 0, 
                  'dmexmax': 1700, 
                  'ndmex': 100,
                  'nmcmc_steps' : 5000,
                  'nwalkers' : 32,
                  'ndim' : 4,
                  'pguess' : (figm_start, F_start, mu_start, sigma_start),                
                  }

    zfrb, dmfrb = data

    nmcmc_steps = param_dict['nmcmc_steps']
    nwalkers = param_dict['nwalkers']
    ndim = param_dict['ndim']
    pguess = param_dict['pguess']

    # Generate the array of parameters from TNG FRB simulations
    IGMparams = get_params(zfrb)
    dmigm_allbaryons_arr = np.array([get_dmigm(zfrb[xx]) for xx in range(len(zfrb))])

    nsamp = nmcmc_steps
    pos = pguess + 1e-3 * np.random.randn(nwalkers, ndim)

    if os.path.exists(mcmc_filename):
        print("Picking up %s where it left off \n" % mcmc_filename)        
        backend = emcee.backends.HDFBackend(mcmc_filename)
    else:
        print("Starting %s from scratch \n" % mcmc_filename)
        backend = emcee.backends.HDFBackend(mcmc_filename)
        backend.reset(nwalkers, ndim)

    with Pool(32) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior_mcquinn,
                                        args=(zfrb, dmfrb, IGMparams, 
                                              dmigm_allbaryons_arr), 
                                             pool=pool, backend=backend)
        
        sampler.run_mcmc(pos, nsamp, progress=True)        

def main(data, param_dict, mcmc_filename='test.h5'):
    
    zfrb, dmfrb = data

    dmmin, dmmax, ndm = param_dict['dmmin'], param_dict['dmmax'], param_dict['ndm']
    zmin, zmax, nz = param_dict['zmin'], param_dict['zmax'], param_dict['nz']
    dmexmin, dmexmax, ndmex = param_dict['dmexmin'], param_dict['dmexmax'], param_dict['ndmex']
    nmcmc_steps = param_dict['nmcmc_steps']
    nwalkers = param_dict['nwalkers']
    ndim = param_dict['ndim']
    pguess = param_dict['pguess']

    # Generate the IGM DM values
    dmi = np.linspace(dmmin, dmmax, ndm)
    # Generate halo DM values
    dmh = np.linspace(dmmin, dmmax, ndm)
    # Generate the total exgal DM values
    dmex = np.linspace(dmexmin, dmexmax, ndm)

    zex = np.linspace(zmin, zmax, nz)

    # Generate the array of parameters from TNG FRB simulations
    tngparams_arr = generate_TNGparam_arr(zex)

    # Generate the meshgrid of halo, IGM, and total exgal DM values
    dmhalo, dmigm, dmexgal = np.meshgrid(dmh, dmi, dmex)

    nsamp = nmcmc_steps
    pos = pguess + 1e-3 * np.random.randn(nwalkers, ndim)

    if os.path.exists(mcmc_filename):
        print("Picking up %s where it left off \n" % mcmc_filename)        
        backend = emcee.backends.HDFBackend(mcmc_filename)
    else:
        print("Starting %s from scratch \n" % mcmc_filename)
        backend = emcee.backends.HDFBackend(mcmc_filename)
        backend.reset(nwalkers, ndim)

    with Pool(64) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior,
                                        args=(zfrb, dmfrb, dmhalo,
                                             dmigm, dmexgal, zex, 
                                             tngparams_arr), 
                                             pool=pool, backend=backend)
        sampler.run_mcmc(pos, nsamp, progress=True)
        
    flat_samples = sampler.get_chain(discard=0, thin=1, flat=True)
    np.save('flat_samples_figm_%dsteps.npy' % len(flat_samples), flat_samples)

    return flat_samples 

if __name__ == '__main__':
    datadir = '/home/connor/software/baryon_paper/data/'
    fn_dsa = '/home/connor/data/dsafrbs_feb2024.csv'
    df = pd.read_csv(fn_dsa, delim_whitespace=False)
    zdsa = df['redshift'].values
    dmdsa = df['dm_exgal'].values
    ind = np.where((zdsa != -1) & (dmdsa > 0) & (np.abs(zdsa) > 0.01))[0]
    zdsa = np.abs(zdsa[ind])
    dmdsa = dmdsa[ind]

    # Start parameters for MCMC chain 
    figm_start, fX_start, mu_start, sigma_start = 0.8, 0.15, 5, 1

    param_dict = {'dmmin': 0, 
                  'dmmax': 1700, 
                  'ndm': 100,
                  'zmin': 0, 
                  'zmax': 1.5, 
                  'nz': 100,
                  'dmexmin': 0, 
                  'dmexmax': 1700, 
                  'ndmex': 100,
                  'nmcmc_steps' : 5000,
                  'nwalkers' : 32,
                  'ndim' : 4,
                  'pguess' : (figm_start, fX_start, mu_start, sigma_start),                
                  }
    
    data = (zdsa, dmdsa - 30.)

    dmall_sub = np.load(datadir + 'dmall_sub.npy')
    zall_sub = np.load(datadir +'zall_sub.npy')
    
    data = (zall_sub, dmall_sub - 30.)

    print("Analyzing %d FRBs" % len(zall_sub))
    
#    dmmac = np.load('dmmacquart20.npy')
#    zmac = np.load('zmacquart20.npy')

#    data = (zmac, dmmac)
#    dm = np.load('/home/connor/dmsim.npy')

#    ztng, dmtng = dm[0], dm[1]

#    indnr = np.where(dm[0] < 2.0)[0]

#    ztng = ztng[indnr]
#    dmtng = dmtng[indnr]

#    induse = np.random.randint(0, len(ztng), 50)

#    data = (ztng[induse], dmtng[induse])
    
    ftoken = 'figm_all_march2_nomark_noada_figmpfxmax.h5'
    mcmc_filename = datadir + "emceechain_%s" % ftoken
    data_filename = datadir + "data_%s" % ftoken

    g = h5py.File(data_filename, 'w')
    g.create_dataset('zfrb', data=data[0])
    g.create_dataset('dmfrb', data=data[1])
    g.close()

    #main(data, param_dict=param_dict, 
    #                    mcmc_filename=mcmc_filename)

    main_mcquinn(data, mcmc_filename=mcmc_filename)
    
