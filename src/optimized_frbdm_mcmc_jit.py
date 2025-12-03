"""
GPU-accelerated optimization for frbdm_mcmc_jit.py
Focuses on speeding up the 2D integral calculation and enabling proper GPU acceleration

Author: Kshitij Duraphe
Based on original by: Liam Connor
"""

import os
import numpy as np
import h5py
import argparse
import pandas as pd
import warnings
import emcee
from tqdm import tqdm
from astropy.cosmology import Planck18 as P
import jax
import jax.numpy as jnp
from jax import jit, vmap
from jax.scipy.integrate import trapezoid
from functools import partial
from typing import List, Optional

# Enable JAX 64-bit precision for numerical stability in integration
jax.config.update("jax_enable_x64", True)

# Try to use GPU if available
try:
    # Check if GPU is available and print device info
    if jax.devices('gpu'):
        print(f"Using GPU: {jax.devices('gpu')[0]}")
    else:
        print("No GPU found, using CPU")
except Exception:
    print("Error detecting devices, defaulting to CPU")

# Constants
figm: float = 0.84  # cosmic baryon fraction in the IGM
c: float = 299792.458  # speed of light in km/s
H0: float = 67.66  # Hubble constant in km/s/Mpc
omega_b: float = 0.02242 / 0.6766**2  # baryon density
pc_to_cm: float = 3.086e18  # parsec to cm conversion

# Pre-compute z grid for integration
Z_GRID: jnp.ndarray = jnp.linspace(0, 10, 500)
Z_GRID_FINE: jnp.ndarray = jnp.linspace(0, 10, 2000)  # Finer grid for final results

@jit
def hz_func(z: float) -> float:
    """
    Compute the dimensionless Hubble parameter at redshift z.

    Args:
        z (float): Redshift value.

    Returns:
        float: The dimensionless Hubble parameter, H(z)/H0.
    """
    return jnp.sqrt(P.Om0 * (1 + z) ** 3 + P.Ode0)

# Pre-compute h(z) on the grid for better performance
HZ_GRID: jnp.ndarray = jit(vmap(hz_func))(Z_GRID)
HZ_GRID_FINE: jnp.ndarray = jit(vmap(hz_func))(Z_GRID_FINE)

@jit
def dmigm_integrand(z: float, fd: float, fe: float, alpha: float) -> float:
    """
    Compute the integrand for the IGM contribution to the dispersion measure (DM).

    DM_IGM = integral( integrand dz ), from z=0 to z=zfrb

    Args:
        z (float): Redshift value.
        fd (float): Fraction of free electrons not in collapsed structures, 
                    possibly evolving with redshift.
        fe (float): Free electron fraction (e.g., helium correction).
        alpha (float): Linear evolution parameter for fd with redshift (fd_z = fd * (1 + alpha*z)).

    Returns:
        float: Value of the integrand at redshift z.
    """
    fd_z = fd * (1 + alpha * z)
    return (1 + z) * figm * fe * fd_z / hz_func(z)

@jit
def compute_dm_grid(fd: float, fe: float, alpha: float) -> jnp.ndarray:
    """
    Pre-compute the DM integrand across the entire Z_GRID for given parameters.

    Args:
        fd (float): Fraction of free electrons not in collapsed structures.
        fe (float): Free electron fraction (e.g., helium correction).
        alpha (float): Linear evolution parameter for fd.

    Returns:
        jnp.ndarray: DM integrand evaluated on Z_GRID.
    """
    return vmap(lambda z: dmigm_integrand(z, fd, fe, alpha))(Z_GRID)

@jit
def get_dmigm(zfrb: float, integrand_grid: jnp.ndarray) -> float:
    """
    Compute the IGM DM for a single FRB redshift zfrb using a pre-computed integrand grid.

    We avoid dynamic slicing by zeroing out integrand values above zfrb, then
    integrating over the entire Z_GRID.

    Args:
        zfrb (float): Redshift of the FRB.
        integrand_grid (jnp.ndarray): Pre-computed values of the integrand for all z in Z_GRID.

    Returns:
        float: IGM dispersion measure contribution in pc/cm^3.
    """
    # Where Z_GRID > zfrb, set integrand to 0.
    integrand_selected = jnp.where(Z_GRID <= zfrb, integrand_grid, 0.0)
    # Integrate over the entire (static) Z_GRID
    return 855.7 * trapezoid(integrand_selected, Z_GRID)

# Vectorize get_dmigm to process multiple FRBs at once
batch_get_dmigm = jit(vmap(get_dmigm, in_axes=(0, None)))

@jit
def log_likelihood(
    theta: jnp.ndarray, 
    zfrb_array: jnp.ndarray, 
    dmobs_array: jnp.ndarray, 
    dmhost_array: jnp.ndarray, 
    dmmw_array: jnp.ndarray
) -> float:
    """
    Compute the log-likelihood of the data given the model parameters.

    Args:
        theta (jnp.ndarray): Model parameters [fd, log_sig_host, alpha, fe].
        zfrb_array (jnp.ndarray): Array of FRB redshifts.
        dmobs_array (jnp.ndarray): Observed DM values for each FRB.
        dmhost_array (jnp.ndarray): Host DM contribution for each FRB (if any).
        dmmw_array (jnp.ndarray): Milky Way DM contribution for each FRB.

    Returns:
        float: Log-likelihood value. If parameters are invalid (e.g., out of bounds),
               returns -jnp.inf.
    """
    fd, log_sig_host, alpha, fe = theta
    
    # Construct a validity mask using JAX-friendly comparisons
    valid_params = (
        (fd > 0) & (fd <= 1) &
        (fe > 0) & (fe <= 1) &
        (log_sig_host >= -3) & (log_sig_host <= 3)
    )
    
    # Pre-compute integrand grid once for all redshifts
    integrand_grid = compute_dm_grid(fd, fe, alpha)
    
    # Batch compute all DM values at once using vectorization
    dm_igm_array = batch_get_dmigm(zfrb_array, integrand_grid)
    
    # Calculate model and residual for all FRBs at once
    dm_model = dm_igm_array + dmhost_array
    dm_resid = dmobs_array - dmmw_array - dm_model
    
    # Calculate log likelihood with total variance
    sig_host = 10 ** log_sig_host
    sig2_tot = sig_host ** 2 + 10.0 ** 2  # e.g., 10 pc/cm^3 observational error
    
    # Vectorized log likelihood calculation
    log_like = -0.5 * jnp.sum(
        dm_resid ** 2 / sig2_tot + jnp.log(2 * jnp.pi * sig2_tot)
    )
    
    # Return -inf for invalid parameters, otherwise return log_likelihood
    return jnp.where(valid_params, log_like, -jnp.inf)

def read_frb_catalog(
    fnfrb: str, 
    zmin: float = 0.0, 
    zmax: float = np.inf, 
    telescope: str = 'all', 
    secure_host: bool = True,
    max_fractional_MWDM: float = np.inf, 
    exclude_names: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Read FRB catalog from a CSV file and apply various filters.

    Args:
        fnfrb (str): Filename (path) to the CSV file containing the FRB catalog.
        zmin (float, optional): Minimum redshift cutoff. Defaults to 0.0.
        zmax (float, optional): Maximum redshift cutoff. Defaults to np.inf.
        telescope (str, optional): Filter by telescope name or "all". Defaults to 'all'.
        secure_host (bool, optional): If True, only include FRBs with secure host associations. 
                                      Defaults to True.
        max_fractional_MWDM (float, optional): Maximum fraction of Milky Way DM allowed 
                                               (NE2001 / dm_opt). Defaults to np.inf.
        exclude_names (List[str], optional): List of FRB names to exclude. Defaults to None.

    Returns:
        pd.DataFrame: Filtered FRB catalog.
    """
    frb_catalog = pd.read_csv(fnfrb, sep=',')
    zfrb = frb_catalog['redshift'].values
    redshift_type = frb_catalog['redshift_type'].values

    ind = np.where((redshift_type != 'none') & (np.abs(zfrb) > zmin) &
                   (np.abs(zfrb) < zmax))[0]

    frb_catalog = frb_catalog.iloc[ind]

    if secure_host:
        ind = np.where(frb_catalog['secure_host'] == 'yes')[0]
        frb_catalog = frb_catalog.iloc[ind]

    if telescope != 'all':
        ind = np.where(frb_catalog['survey'] == telescope.upper())[0]
        if len(ind) == 0:
            print("Are you sure you have the right telescope name?")
        frb_catalog = frb_catalog.iloc[ind]

    if max_fractional_MWDM < np.inf:
        frac_mw = frb_catalog['ne2001'].values / frb_catalog['dm_opt'].values
        ind = np.where(frac_mw < max_fractional_MWDM)[0]
        frb_catalog = frb_catalog.iloc[ind]

    if exclude_names is not None:
        if not isinstance(exclude_names, list):
            exclude_names = [exclude_names]
        for name in exclude_names:
            indexNames = frb_catalog[frb_catalog['name'] == name].index
            frb_catalog.drop(indexNames, inplace=True)

    return frb_catalog

def run_mcmc(
    zfrb_array: np.ndarray, 
    dmobs_array: np.ndarray, 
    dmhost_array: np.ndarray, 
    dmmw_array: np.ndarray,
    nwalkers: int = 32, 
    nsteps: int = 2000, 
    progress: bool = True
) -> emcee.EnsembleSampler:
    """
    Run MCMC sampling to estimate model parameters for the FRB DM model.

    Args:
        zfrb_array (np.ndarray): Array of FRB redshifts.
        dmobs_array (np.ndarray): Array of observed DMs for each FRB.
        dmhost_array (np.ndarray): Array of host DM contributions for each FRB.
        dmmw_array (np.ndarray): Array of Milky Way DM contributions for each FRB.
        nwalkers (int, optional): Number of MCMC walkers. Defaults to 32.
        nsteps (int, optional): Number of MCMC steps. Defaults to 2000.
        progress (bool, optional): If True, display a progress bar. Defaults to True.

    Returns:
        emcee.EnsembleSampler: The MCMC sampler after running the chain.
    """
    # Convert numpy arrays to JAX arrays for GPU acceleration
    zfrb_array_jax = jnp.array(zfrb_array)
    dmobs_array_jax = jnp.array(dmobs_array)
    dmhost_array_jax = jnp.array(dmhost_array)
    dmmw_array_jax = jnp.array(dmmw_array)
    
    # Define the number of parameters (fd, log_sig_host, alpha, fe)
    ndim = 4
    
    # Create log probability function with fixed data parameters
    def log_prob(theta: jnp.ndarray) -> float:
        return log_likelihood(theta, zfrb_array_jax, dmobs_array_jax, dmhost_array_jax, dmmw_array_jax)
    
    # Set up initial MCMC positions
    fd_init, sig_host_init, alpha_init, fe_init = 0.8, 1.7, 0.0, 7/8.
    pos = np.array([
        [fd_init, sig_host_init, alpha_init, fe_init] +
        1e-3 * np.random.randn(ndim) for _ in range(nwalkers)
    ])
    
    # Compile the log_prob function once before MCMC
    print("Compiling likelihood function...")
    _ = log_prob(jnp.array([0.8, 1.7, 0.0, 7/8.]))
    print("Compilation complete, starting MCMC...")
    
    # Initialize and run emcee sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)
    sampler.run_mcmc(pos, nsteps, progress=progress)
    
    return sampler

def main() -> None:
    """
    Main function to parse arguments, filter the FRB catalog, run the MCMC, and save results.
    """
    parser = argparse.ArgumentParser(description="Optimized MCMC analysis of FRB DMs")
    parser.add_argument('--zmin', type=float, default=0.01, help='Minimum redshift')
    parser.add_argument('--zmax', type=float, default=np.inf, help='Maximum redshift')
    parser.add_argument('--tel', type=str, default='all', help='Telescope (e.g., dsa-110)')
    parser.add_argument('--exclude', type=str, default=None, help='Name of FRB to exclude')
    parser.add_argument('--nmcmc', type=int, default=2000, help='Number of MCMC steps')
    parser.add_argument('--nwalkers', type=int, default=32, help='Number of walkers')
    args = parser.parse_args()
    
    # Read FRB catalog
    frb_catalog = read_frb_catalog(
        '../data/frbsample_connor0924.csv',
        zmin=args.zmin,
        zmax=args.zmax,
        telescope=args.tel,
        exclude_names=args.exclude
    )
    
    # Extract relevant data
    zfrb_array = frb_catalog['redshift'].values
    dmobs_array = frb_catalog['dm_opt'].values
    dmmw_array = frb_catalog['ne2001'].values
    # We'll assume zero host contribution in this example
    dmhost_array = np.zeros_like(zfrb_array)
    
    print(f"Found {len(zfrb_array)} FRBs within redshift range")
    
    # Run optimized MCMC
    sampler = run_mcmc(
        zfrb_array, dmobs_array, dmhost_array, dmmw_array,
        nwalkers=args.nwalkers, nsteps=args.nmcmc
    )
    
    # Save results
    outdir = 'data'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    # Create output filename
    outstr = f"_zmin{args.zmin:.2f}_zmax{args.zmax if args.zmax != np.inf else 'inf'}"
    outstr += f"_tel{args.tel}_exclude{args.exclude if args.exclude else 'None'}"
    
    # Save MCMC chain
    fn_chain = os.path.join(outdir, f"optimized_emceechain_results{outstr}.h5")
    with h5py.File(fn_chain, 'w') as f:
        f.create_dataset('chain', data=np.array(sampler.chain))
        f.create_dataset('lnprobability', data=np.array(sampler.lnprobability))
    
    # Save data
    fn_data = os.path.join(outdir, f"optimized_data_results{outstr}.h5")
    with h5py.File(fn_data, 'w') as f:
        f.create_dataset('zfrb', data=zfrb_array)
        f.create_dataset('dmobs', data=dmobs_array)
        f.create_dataset('dmmw', data=dmmw_array)
    
    print(f"Results saved to {fn_chain} and {fn_data}")
    
    # Compute parameter estimates from the MCMC chain
    burnin = args.nmcmc // 2
    flat_samples = sampler.chain[:, burnin:, :].reshape(-1, 4)
    
    fd_mcmc = np.percentile(flat_samples[:, 0], [16, 50, 84])
    sig_host_mcmc = np.percentile(10 ** flat_samples[:, 1], [16, 50, 84])
    alpha_mcmc = np.percentile(flat_samples[:, 2], [16, 50, 84])
    fe_mcmc = np.percentile(flat_samples[:, 3], [16, 50, 84])
    
    print("\nParameter estimates (16th, 50th, 84th percentiles):")
    print(f"fd = {fd_mcmc[1]:.3f} ({fd_mcmc[0]:.3f}, {fd_mcmc[2]:.3f})")
    print(f"σ_host = {sig_host_mcmc[1]:.1f} "
          f"({sig_host_mcmc[0]:.1f}, {sig_host_mcmc[2]:.1f}) pc/cm³")
    print(f"alpha = {alpha_mcmc[1]:.3f} "
          f"({alpha_mcmc[0]:.3f}, {alpha_mcmc[2]:.3f})")
    print(f"fe = {fe_mcmc[1]:.3f} "
          f"({fe_mcmc[0]:.3f}, {fe_mcmc[2]:.3f})")

if __name__ == "__main__":
    main()
