import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.coordinates import match_coordinates_sky

# Define a function to cross-match FRB sources with a galaxy cluster catalog
def cross_match_frb_with_clusters(frb_sources, cluster_catalog, thresh_arcmin=1.0):
    # Create SkyCoord objects for FRB sources and cluster catalog
    frb_coords = SkyCoord(ra=frb_sources['ra'], dec=frb_sources['dec'], unit=(u.deg, u.deg))
    cluster_coords = SkyCoord(ra=cluster_catalog['RA'], dec=cluster_catalog['Dec'], unit=(u.deg, u.deg))

    # Perform the cross-match
    idx, sep, _ = match_coordinates_sky(frb_coords, cluster_coords)

    # Filter clusters that are within a certain angular separation threshold (e.g., 1 arcmin)
    matching_clusters = cluster_catalog[idx[sep < thresh_arcmin*u.arcmin]]

    return matching_clusters

def read_legacy():
    pass

def cross_match_Legacy(frb_sources, fn_legacy):
    pass 

def read_PSZ2():
    pass

def cross_match_PSZ2(frb_sources, fn_PSZ2):
    pass

def read_WHY_clustercat(fndir):

    fn_1 = fndir + '/table1.dat'
    fn_2 = fndir + '/table2.dat'
    fn_3 = fndir + '/table3.dat'

    f1 = open(fn_1, 'r')
    f2 = open(fn_2, 'r')
    f3 = open(fn_3, 'r')

    lines1 = f1.readlines()
    lines2 = f2.readlines()
    lines3 = f3.readlines()

    arr_1 = []
    [arr_1.append(np.array(lines1[ii].split())) for ii in range(len(lines1))]
    arr_1 = np.concatenate(arr_1).reshape(-1, 9)

    arr_2 = []
    [arr_2.append(np.array(lines2[ii].split())) for ii in range(len(lines2))]
    arr_2 = np.concatenate(arr_2).reshape(-1, 9)

    arr_3 = []
    [arr_3.append(np.array(lines3[ii].split())) for ii in range(len(lines3))]
    arr_3 = np.concatenate(arr_3).reshape(-1, 9)

    data_1 = {
        'name': arr_1[:, 0],
        'ra': arr_1[:, 1].astype(float),
        'dec': arr_1[:, 2].astype(float),
        'redshift': arr_1[:, 3].astype(float),
    }

    df_1 = pd.DataFrame(data_1)

    data_2 = {
        'name': arr_2[:, 1],
        'ra': arr_2[:, 2].astype(float),
        'dec': arr_2[:, 3].astype(float),
        'redshift': arr_2[:, 4].astype(float),
    }

    df_2 = pd.DataFrame(data_2)

    data_3 = {
        'name': arr_3[:, 1],
        'ra': arr_3[:, 2].astype(float),
        'dec': arr_3[:, 3].astype(float),
        'redshift': arr_3[:, 4].astype(float),
    }

    df_3 = pd.DataFrame(data_3)

    return df_1, df_2, df_3

def read_ROSAT(fn_ROSAT='table_rxgcc.fits'):
    # https://www.aanda.org/articles/aa/abs/2022/02/aa40908-21/aa40908-21.html
    f = fits.open(fn_ROSAT)
    data = f[1].data

    ra, dec, redshift = [], [], []

    for i in range(len(data)):
        ra.append(data[i][1])
        dec.append(data[i][2])
        redshift.append(data[i][5])

    data = { 
        'ra': ra,
        'dec': dec,
        'redshift': redshift
    }

    df = pd.DataFrame(data)

    return df

def cross_match_ROSAT(frb_sources, fn_ROSAT):
    pass

def read_frb_catalog(fn_frb):
    # Index(['name', 'mjd', 'snr_heim', 'dm_heim', 'dm_opt', 'dm_exgal', 'ibox', 'redshift', 'ra', 'dec']
    frb_sources = pd.read_csv(fn_frb, delim_whitespace=False)

    return frb_sources