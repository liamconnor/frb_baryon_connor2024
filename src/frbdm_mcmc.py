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
from astropy.cosmology import Planck18 as P
from scipy.integrate import dblquad, quad
from astropy import constants as con, units as u

from reader import read_frb_catalog

def dmigm_integrand(z, figm=1, fe=7/8., alpha=0.0):
    figm = figm * (1 + alpha*z)
    y = (1+z)*figm*fe / (P.H(z)/P.H0)
    return y

def get_dmigm(zfrb, figm=1):
    A = 3 * con.c * P.Ob(0) * P.H0 / (8 * np.pi * con.G* con.m_p)
    val = (A * quad(dmigm_integrand, 0, zfrb, args=(figm))[0]).to(u.pc * u.cm**-3).value
    return val

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
    figmTNG = 0.797
    fxTNG = 0.131
#    figmTNG, fxTNG = 0.76, 0.20
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

    if 0.0 < figm < 1.25:
        if 0. < fx < 1.25:
            if 0 < mu < 9:
                if 0.01 < sigma < 2.5:
                    if figm + fx < 1.5:
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

def zhang_params():
    """ The fit parameters from Zhang et al. 2020 assuming the mcquinn PDF model
    https://iopscience.iop.org/article/10.3847/1538-4357/abceb9
    """
    dzhang = np.array([[0.1, 0.04721, -13.17, 2.554],
                        [0.2, 0.005693, -1.008, 1.118],
                        [0.3, 0.003584, 0.596, 0.7043],
                        [0.4, 0.002876, 1.010, 0.5158],
                        [0.5, 0.002423, 1.127, 0.4306],
                        [0.7, 0.001880, 1.170, 0.3595],
                        [1, 0.001456, 1.189, 0.3044],
                        [1.5, 0.001098, 1.163, 0.2609],
                        [2, 0.0009672, 1.162, 0.2160],
                        [2.4, 0.0009220, 1.142, 0.1857],
                        [3, 0.0008968, 1.119, 0.1566],
                        [3.5, 0.0008862, 1.104, 0.1385],
                        [4, 0.0008826, 1.092, 0.1233],
                        [4.4, 0.0008827, 1.084, 0.1134],
                        [5, 0.0008834, 1.076, 0.1029],
                        [6.5, 0.0008881, 1.066, 0.08971]])

    zarr, A, C0, sigmaDM = dzhang[:,0], dzhang[:,1], dzhang[:,2], dzhang[:,3]

    A_spl = UnivariateSpline(zarr, A, s=0)
    C0_spl = UnivariateSpline(zarr, C0, s=0)
    sigmaDM_spl = UnivariateSpline(zarr, sigmaDM, s=0)

    return A_spl, C0_spl, sigmaDM_spl

def get_params_zhang(zfrb, A_spl, C0_spl, sigmaDM_spl):
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

    if 0 < figm < 4:
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
        

    return flat_samples 

if __name__ == '__main__':
    datadir = '/home/connor/software/baryon_paper/data/'
    fnfrb = datadir + 'allfrbs_13march24y.csv'
    ftoken_output = 'figm_dsaonly_march20b_allFRBs_dmmax1500.h5'
    zmin_sample = 0.01
    zmax_sample = np.inf
    telecopes = 'DSA-110'
    max_fractional_MWDM = 0.4
    dmhalo = 30.
    exclude_frbs = ['ada', 'FRB20190520B']
    nmcmc_steps = 2000
    
    frb_catalog = read_frb_catalog(fnfrb, zmin=zmin_sample, zmax=zmax_sample, 
                                   telescope=telecopes, secure_host=True,
                                   max_fractional_MWDM=max_fractional_MWDM,
                                   exclude_names=exclude_frbs)

    zfrb = frb_catalog['redshift'].values
    dmfrb = frb_catalog['dm_exgal'].values - dmhalo
    dmmax = frb_catalog['dmmax'].values
    
    data = (zfrb, dmfrb)
    
    # Start parameters for MCMC chain 
    figm_start, fX_start, mu_start, sigma_start = 0.8, 0.15, 5, 1

    param_dict = {'dmmin': 0, 
                  'dmmax': 1500., 
                  'ndm': 100,
                  'zmin': 0, 
                  'zmax': 1.5, 
                  'nz': 100,
                  'dmexmin': 0, 
                  'dmexmax': 1500, 
                  'ndmex': 100,
                  'nmcmc_steps' : nmcmc_steps,
                  'nwalkers' : 32,
                  'ndim' : 4,
                  'pguess' : (figm_start, fX_start, mu_start, sigma_start),                
                  }


#    data = np.load('/home/connor/TNG300_Total_DMvsZ.pdf.npy')
#    ztng = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1., 1.5, 2, 3, 4, 5])
#    dmfrb = data[:8, np.random.randint(0, 5000, 5)].flatten() 
#    zfrb = (ztng[:8, None] * np.ones([1, 5])).flatten()
#    dmhost = np.random.lognormal(4.5, 0.8, len(zfrb)) * (1+zfrb)**-1

#    data = (zfrb, dmfrb + dmhost)
    
    mcmc_filename = datadir + "emceechain_%s" % ftoken_output
    data_filename = datadir + "data_%s" % ftoken_output

    # Save the data itself to an h5 file
    g = h5py.File(data_filename, 'w')
    g.create_dataset('zfrb', data=data[0])
    g.create_dataset('dmfrb', data=data[1])
    g.close()

    main(data, param_dict=param_dict, 
                        mcmc_filename=mcmc_filename)

    #main_mcquinn(data, mcmc_filename=mcmc_filename)
    
