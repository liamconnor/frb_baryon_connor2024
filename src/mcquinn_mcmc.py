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


