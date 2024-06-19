""" 
MCMC code to fit the baryon parameters to the FRB DM data. This 
is a JAX-compiled MCMC code, but is still far too slow. Please 
feel free to submit a PR to speed things up! I didn't manage to 
get it running on GPU yet.

Author: Liam Connor
Email: liam.dean.connor@gmail.com
Data: 2024-06-19
"""

import os

import numpy as np
import h5py
import argparse
import pandas as pd
from multiprocessing import Pool
import warnings
import emcee
from scipy.integrate import quad
from scipy.interpolate import UnivariateSpline
from tqdm import tqdm
from astropy.cosmology import Planck18 as P
from scipy.integrate import quad
from astropy import constants as con, units as u
import jax.numpy as jnp
from jax import jit, vmap
import jax

from reader import read_frb_catalog

warnings.filterwarnings("ignore", category=RuntimeWarning)

def dmigm_integrand(z, fd=1, fe=7/8., alpha=0.0):
    """ Integrand for the IGM contribution to the FRB DM.

    Parameters:
    -----------
    z : float
        Redshift
    fd : float
        The fraction of baryons that are diffuse gas in the IGM/halos
    fe : float
        The average number of electrons per baryon
    alpha : float
        The redshift dependence of the diffuse baryon fraction

    Returns:
    --------
    float : The integrand for the cosmic contribution to the FRB DM
    """
    fd = fd * (1 + alpha*z)
    y = (1+z)*figm*fe / (P.H(z)/P.H0)
    return y

def get_dmigm(zfrb, fd=1):
    """ Compute the cosmic gas contribution to the FRB DM.
    
    Parameters:
    -----------
    zfrb : float
        Redshift of the FRB
    fd : float
        The fraction of baryons that are diffuse gas in the IGM/halos
    
    Returns:
    --------
    float : The cosmic gas contribution to the FRB DM
    """
    A = 3 * con.c * P.Ob(0) * P.H0 / (8 * np.pi * con.G * con.m_p)
    dm = (A * quad(dmigm_integrand, 0, zfrb, args=(fd))[0]).to(u.pc * u.cm**-3).value
    return dm

def generate_TNGparam_arr(zfrb):
    """ Read in TNG300 parameter fits based on Walker et al. 2023 
    and interpolate to the redshifts of the FRBs. These 
    parameters fit a 2D multivariate logNormal distribution in 
    the IGM and halo contributions to the FRB DM.

    The parameter at each redshift is a 6D list:
    [A, mu_dmx, mu_dmigm, sigma_dmx, sigma_dmigm, rho] which corresponds to the 
    parameters of the logNormal distribution in DM halo and DM IGM.
    """
    TNGfits = np.load('/home/connor/software/baryon_paper/src/tng_params_new.npy')
    nfrb = len(zfrb)
    arr = TNGfits
    tngparams_arr = np.zeros([nfrb, 6])

    # The redshift snapshots at which the TNG parameters were fit
    ztng = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2, 3, 4, 5]

    A = UnivariateSpline(ztng, arr[:, 0], s=0)
    mu_dmx = UnivariateSpline(ztng, arr[:, 1], s=0)
    mu_dmigm = UnivariateSpline(ztng, arr[:, 2], s=0)
    sigx = UnivariateSpline(ztng, arr[:, 3], s=0)
    sigigm = UnivariateSpline(ztng, arr[:, 4], s=0)
    rho = UnivariateSpline(ztng, arr[:, 5], s=0)

    for ii in range(nfrb):
        zz = zfrb[ii]
        Ai = A(zz).ravel()[0]
        mu_dmxi = mu_dmx(zz).ravel()[0]
        mu_dmigmi = mu_dmigm(zz).ravel()[0]
        sigxi = sigx(zz).ravel()[0]
        sigigmi = sigigm(zz).ravel()[0]
        rhoi = rho(zz).ravel()[0]

        tngparams_arr[ii] = np.array([Ai, mu_dmxi, mu_dmigmi, sigxi, sigigmi, rhoi])

    return tngparams_arr

@jit
def pdm_cosmic(dmhalo, dmigm, params, TNGparams, 
               figm_baseline=0.797, fx_baseline=0.131):
    """ 2D Multivariate Log-normal PDF for the cosmic DM contribution

    Parameters:
    -----------
    dmhalo : float, array or meshgrid
        The DM contribution from intervening halos
    dmigm : float, array or meshgrid
        The DM contribution from the IGM
    params : list
        The cosmic gas parameters [figm, fx]
    TNGparams : array
        The TNG300 parameters for the logNormal distribution in DMhalo and DMigm
    figm_baseline : float
        The fiducial value of the IGM baryon fraction from the baseline sim
    fx_baseline : float
        The fiducial value of the halo baryon fraction from the baseline sim
    
    Returns:
    --------
    The probability of the cosmic DM contribution for that DMhalo and DMigm and set of params
    """
    x, y = dmhalo, dmigm
    figm, fx = params
    A, mu_x, mu_y, sigma_x, sigma_y, rho = TNGparams
    mu_y += jnp.log(figm / figm_baseline)
    mu_x += jnp.log(fx / fx_baseline)
    term1 = -((jnp.log(x) - mu_x)**2 / sigma_x**2 + (jnp.log(y) - mu_y)**2 / sigma_y**2)
    term2 = 2 * rho * (jnp.log(x) - mu_x) * (jnp.log(y) - mu_y) / (sigma_x * sigma_y)
    B = (2 * jnp.pi * sigma_x * sigma_y * x * y * jnp.sqrt(1 - rho**2))
    return A / B * jnp.exp((term1 - term2) / (2 * (1 - rho**2)))

@jit
def pdm_product_numerical(dmhalo, dmigm, dmexgal, zfrb, params, TNGparams):
    """ Compute the product of the cosmic and host DM PDFs. 
    This will serve the integrand for the likelihood function.

    Parameters:
    -----------
    dmhalo : meshgrid
        Values of DM halo over which to compute the PDF
    dmigm : meshgrid
        Values of DM IGM over which to compute the PDF
    dmexgal : meshgrid
        Values of total exgalactic DM over which to compute the PDF
    zfrb : float
        Redshift of the FRB
    params : list
        [figm, fx, mu, sigma]
        figm - fraction of baryons in IGM
        fx - fraction of baryons in halos
        mu, sigma - parameters for the log-normal host distribution 
    TNGparams : array
        The TNG300 parameters for the logNormal distribution in DMhalo and DMigm

    Returns:
    --------
    The product of the cosmic and host DM PDFs
    """
    figm, fx, mu, sigma = params
    dmhost = dmexgal - dmhalo - dmigm
    pcosmic = vmap(pdm_cosmic, 
                   in_axes=(0, 0, None, None))(dmhalo, dmigm, (figm, fx), 
                                               TNGparams)
    phost = vmap(pdmhost, in_axes=(0, None, None))(dmhost * (1+zfrb), mu, sigma)
    return pcosmic * phost, dmhost

@jit
def pdm_product_numerical_cosmic(dmigm, dmexgal, zfrb, params, TNGparams):
    """ Compute the PDF integral of the cosmic DM.
    """
    figm, fx = params
    dmhalo = dmexgal - dmigm
    pcosmic = vmap(pdm_cosmic, in_axes=(0, 0, None, None))(dmhalo, dmigm, (figm, fx), TNGparams)
    return pcosmic

@jit
def pdmhost(dmhost, mu, sigma):
    """ Log-normal PDF for the host DM."""
    prob = 1/(dmhost * jnp.sqrt(2*np.pi) * sigma)
    prob *= jnp.exp(-(jnp.log(dmhost) - mu)**2 / (2*sigma**2))
    return prob

def log_likelihood_all(params, zfrb, dmfrb, dmhalo, dmmax_survey, 
                       dmigm, dmexgal, zex, tngparams_arr):
    """ Log likelihood summed over all FRBs in the dataset.

    Parameters:
    -----------
    params : list
        [figm, fx, mu, sigma]
        figm - fraction of baryons in IGM
        fx - fraction of baryons in halos
        mu, sigma - parameters for the log-normal host distribution
    zfrb : array
        Redshifts of the FRBs in sample
    dmfrb : array
        DM of the FRBs in sample
    dmhalo : meshgrid
        Values of DM halo over which to compute the PDF
    dmmax_survey : array
        Maximum DM of the survey that detected the FRB
    dmigm : meshgrid
        Values of DM IGM over which to compute the PDF
    dmexgal : meshgrid
        Values of total exgalactic DM over which to compute the PDF
    zex : array
        Redshifts at which to compute the PDF
    tngparams_arr : array

    Returns:
    --------
    float : The log likelihood of the baryon parameters
    """
    nz, ndm = len(zex), len(dmhalo)
    dmex = dmexgal[0,0]
    
    P = np.empty((ndm, nz))

    # Iterate over all redshifts and compute the likelihood
    # in DM at that redshift for those baryon parameters.
    # This method avoids computing the likelihood multiple times 
    # for each FRB. 
    for ii in range(len(zex)):
        pp, dmhost = pdm_product_numerical(dmhalo, dmigm, dmexgal, 
                                           zex[ii], params, 
                                           tngparams_arr[ii])
        # Convert back to numpy arrays
        pp = np.array(pp)
        dmhost = np.array(dmhost)
        
        # Sum the likelihoods over the allowed values of host DM 
        # (DMexgal - DMhalo - DMigm)
        for kk, dd in enumerate(dmex):
            p, dh = pp[:, :, kk], dmhost[:, :, kk]
            P[kk, ii] = np.nansum(p[dh > 0], axis=-1)

    nfrb = len(zfrb)
    
    logP = 0

    # Step through each FRB in the dataset and
    # find the Likelihood bin that is nearest
    # to the measured DM and z.
    for nn in range(nfrb):
        # Find bin with DMmax of the survey that detected FRB nn
        dmmax_bin = np.argmin(np.abs(dmex - dmmax_survey[nn]))
        
        # Nearest redshift bin
        ll = np.argmin(np.abs(zfrb[nn] - zex))
        # Nearest DM bin
        kk = np.argmin(np.abs(dmfrb[nn] - dmex))
        # Normalize the likelihoods on a per-redshift basis        
        Prob_normalized = P[:, ll] / np.nansum(P[:dmmax_bin+1, ll])
        # Loglikelihood in that bin
        lp = np.log(Prob_normalized[kk])
        
        # If any FRB produces a zero likelihood, return -inf
        # as these parameters are now excluded.
        if np.isnan(lp):
            return -np.inf
        else:
            logP += lp

    return logP

def log_prior(params, prior_dict):
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
    if len(params) != 4:
        print("Wrong number of parameters in log_prior. Exiting.")
        return

    if prior_dict is None:
        return 0

    figm, fx, mu, sigma = params
    
    figmmin, figmmax = prior_dict['figmmin'], prior_dict['figmmax']
    fxmin, fxmax = prior_dict['fxmin'], prior_dict['fxmax']
    mumin, mumax = prior_dict['mumin'], prior_dict['mumax']
    sigmin, sigmax = prior_dict['sigmin'], prior_dict['sigmax']
    fcos_total_max = prior_dict['fcos_total_max']

    if figmmin < figm < figmmax and fxmin < fx < fxmax \
        and mumin < mu < mumax and sigmin < sigma < sigmax \
        and figm + fx < fcos_total_max:

        return 0

    return -np.inf

def log_posterior(params, zfrb, dmfrb, dmmax_survey, dmhalo, dmigm,
                  dmexgal, zex, tngparams_arr, prior_dict):
    """ Log posterior for the baryon parameters. First 
    check if the parameters are within the prior range, 
    then compute the log likelihood."""
    log_pri = log_prior(params, prior_dict)

    if not np.isfinite(log_pri):
        return -np.inf

    log_like = log_likelihood_all(params, zfrb, dmfrb, dmhalo, dmmax_survey,
                                  dmigm, dmexgal, zex, 
                                  tngparams_arr)
    if not np.isfinite(log_like):
        return -np.inf
    
    return log_pri + log_like 


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

def main(data, param_dict, mcmc_filename='test.h5'):
    """
    Main function to run the MCMC chain. This function
    will run the MCMC chain and save the results to an
    HDF5 file.

    Parameters:
    -----------
    data : tuple
        Tuple containing the redshifts and DMs of the FRBs
    param_dict : dict
        Dictionary containing the parameters for the MCMC chain
    mcmc_filename : str
        Filename to save the MCMC chain to

    Returns:
    --------
    None
    """
    
    zfrb, dmfrb = data
    prior_dict = param_dict['prior_dict']
    dmmin, dmmax, ndm = param_dict['dmmin'], param_dict['dmmax'], param_dict['ndm']
    zmin, zmax, nz = param_dict['zmin'], param_dict['zmax'], param_dict['nz']
    dmexmin, dmexmax, ndmex = param_dict['dmexmin'], param_dict['dmexmax'], param_dict['ndmex']
    nmcmc_steps = param_dict['nmcmc_steps']
    nwalkers = param_dict['nwalkers']
    ndim = param_dict['ndim']
    pguess = param_dict['pguess']
    dmmax_survey = param_dict['dmmax_survey']

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
    pos = pguess + 1e-2 * np.random.randn(nwalkers, ndim)

    if os.path.exists(mcmc_filename):
        print("Picking up %s where it left off \n" % mcmc_filename)        
        backend = emcee.backends.HDFBackend(mcmc_filename)
        pos = None
    else:
        print("Starting %s from scratch \n" % mcmc_filename)
        backend = emcee.backends.HDFBackend(mcmc_filename)
        backend.reset(nwalkers, ndim)

    with Pool(32) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior,
                                        args=(zfrb, dmfrb, dmmax_survey, dmhalo,
                                             dmigm, dmexgal, zex, 
                                             tngparams_arr, prior_dict), 
                                             pool=pool, backend=backend)
        sampler.run_mcmc(pos, nsamp, progress=True)
        

def parse_arguments():
    """
    Parses command-line arguments using argparse.

    Returns:
        Namespace: An object containing parsed arguments.
    """

    parser = argparse.ArgumentParser(description="Your script description")

    # Optional arguments with short and long options, type casting
    parser.add_argument("-t", "--telescope", type=str,
                        help="Name of the telescope to use",
                        default='all')
    parser.add_argument("--zmin", type=float, default=0.0, help="Minimum redshift value")
    parser.add_argument("--zmax", type=float, default=np.inf, help="Maximum redshift value")
    parser.add_argument("--dmmw", type=float, default=0.4, help="Maximum fractional MW DM")
    parser.add_argument("--exclude", type=str, default=None, help="Exclude FRB names")  
    parser.add_argument("--nmcmc", type=int, default=2000, help="Number of MCMC steps")
    parser.add_argument("--dmhalo", type=float, default=30., help="Halo DM value")
    parser.add_argument("--dmmax", type=float, default=1500., help="Maximum DM value")
    parser.add_argument("--fnout", type=str, default='', help="Output file string")

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()

    datadir = '../data/'
    fnfrb = datadir + 'naturesample_april2024.csv'
    zmin_sample = args.zmin
    zmax_sample = args.zmax
    telecopes = args.telescope
    max_fractional_MWDM = args.dmmw
    dmhalo = args.dmhalo
    exclude_frbs = ['ada', 'FRB20190520B', 'mayra', 'wilhelm']

    if args.exclude is not None:
        exclude_frbs = exclude_frbs + [args.exclude]
        print(exclude_frbs)

    nmcmc_steps = args.nmcmc
    ftoken_output = args.fnout + 'june19_zmin%0.2f_zmax%0.2f_tel%s_exclude%s.h5' % \
                        (zmin_sample, zmax_sample, telecopes, args.exclude)

    frb_catalog = read_frb_catalog(fnfrb, zmin=zmin_sample, zmax=zmax_sample, 
                                   telescope=telecopes, secure_host=True,
                                   max_fractional_MWDM=max_fractional_MWDM,
                                   exclude_names=exclude_frbs)

    zfrb = np.abs(frb_catalog['redshift'].values)
    dmfrb = frb_catalog['dm_exgal'].values - dmhalo
    dmmax_survey = frb_catalog['dmmax'].values
    
    data = (zfrb, dmfrb)
    print(len(frb_catalog))
    
    # Start parameters for MCMC chain 
    figm_start, fX_start, mu_start, sigma_start = 0.75, 0.10, 4.25, 0.5

    # Bayesian priors on the baryon parameters
    prior_dict = {'figmmin': 0.0,
                'figmmax': 1.0,
                'fxmin': 0.0,
                'fxmax': 1.0,
                'mumin': 0.0,
                'mumax': 7.0,
                'sigmin': 0.1,
                'sigmax': 3.0,
                'fcos_total_max': 1.2}
    
    # Parameters for the MCMC chain
    param_dict = {'dmmin': 0, 
                  'dmmax': 2000., 
                  'ndm': 100,
                  'zmin': 0, 
                  'zmax': 1.5, 
                  'nz': 100,
                  'dmexmin': 0, 
                  'dmexmax': 2000, 
                  'ndmex': 100,
                  'nmcmc_steps' : nmcmc_steps,
                  'nwalkers' : 32,
                  'ndim' : 4,
                  'dmmax_survey' : dmmax_survey,
                  'pguess' : (figm_start, fX_start, mu_start, sigma_start),
                  'prior_dict' : prior_dict}
    
    mcmc_filename = datadir + "emceechain_%s" % ftoken_output
    data_filename = datadir + "data_%s" % ftoken_output

    # Save the data itself to an h5 file
    g = h5py.File(data_filename, 'w')
    g.create_dataset('zfrb', data=data[0])
    g.create_dataset('dmfrb', data=data[1])
    g.close()

    main(data, param_dict=param_dict, 
                        mcmc_filename=mcmc_filename)

    
