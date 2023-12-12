import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.coordinates import match_coordinates_sky
from astropy.io import fits
from astropy.cosmology import Planck18 as cosmo

# Define a function to cross-match FRB sources with a galaxy cluster catalog
def cross_match_frb_with_clusters(frb_sources, cluster_catalog, thresh_bperp_mpc=2.0):
    if cluster_catalog is None:
        return None
    # Create SkyCoord objects for FRB sources and cluster catalog
    frb_coords = SkyCoord(ra=frb_sources['ra'], dec=frb_sources['dec'], unit=(u.deg, u.deg))
    cluster_coords = SkyCoord(ra=cluster_catalog['ra'], dec=cluster_catalog['dec'], unit=(u.deg, u.deg))

    # Calculate the angular diameter distance to each FRB source and cluster
    D_A_clust = cosmo.angular_diameter_distance(cluster_catalog['redshift']).values
    D_A_frb = cosmo.angular_diameter_distance(np.abs(frb_sources['redshift'])).values
    clust_ind_match = []
    frb_ind_match = []
    bperp_match_arr = []

    for ii in range(len(frb_coords)):
        sep_rad = frb_coords[ii].separation(cluster_coords).rad
        bperp = sep_rad * D_A_clust
        
        # Find indexes that are within 2 Mpc of an FRB sightline
        clust_ind_match_ii = np.where((bperp.value < thresh_bperp_mpc) & (D_A_clust < 1.1 * D_A_frb[ii]))[0]

        if len(clust_ind_match_ii) > 0:
            clust_ind_match.append(clust_ind_match_ii)
            frb_ind_match.append(ii)
            bperp_match_arr.append(bperp[clust_ind_match_ii].value)

    return clust_ind_match, frb_ind_match, bperp_match_arr

def read_legacy():
    pass

def cross_match_Legacy(frb_sources, fn_legacy):
    pass 

def read_PSZ2(fn_PSZ2):
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

    name_3, ra_3, dec_3, z_3 = [], [], [], []

    for line in lines3:
        if line[0] == '#':
            continue

        ll = line.split()
        if ll[0]=='WHY':
            ll = np.roll(ll, -1)

        name_3.append(ll[0])
        ra_3.append(ll[2])
        dec_3.append(ll[1])
        z_3.append(ll[3])

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

    data_3 = {'name': np.array(name_3),
        'ra': np.array(ra_3).astype(float),
        'dec': np.array(dec_3).astype(float),
        'redshift': np.array(z_3).astype(float),
    }

    df_3 = pd.DataFrame(data_3)

    return df_1, df_2, df_3

def read_xclass(fn_xclass='./Xclass_cat.fit'):
    f = fits.open(fn_xclass)
    data = f[1].data

    ra, dec, redshift = [], [], []

    for ii in range(len(data)):
        # Example line: 'names':['XClass','RAJ2000','DEJ2000','RAmdeg','DEmdeg','z','f_z','MLdet']
        # (20, 193.438, 10.1954, 193.438, 10.1951, 0.654, '  confirmed', 360.664)
        ra.append(data[ii][3])
        dec.append(data[ii][4])
        redshift.append(data[ii][5])

    data = {
        'ra': np.array(ra).astype(float),
        'dec': np.array(dec).astype(float),
        'redshift': np.array(redshift).astype(float)
    }

    df = pd.DataFrame(data)

    return df

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

def cross_match_all(fn_frb):
    fn_frb='/Users/liamconnor/Desktop/dsafrbsnov23.csv'

    frb_sources = read_frb_catalog(fn_frb)

    # PSZ2
    fn_PSZ2 = '/Users/kim/Research/FRB/FRB_catalogs/PSZ2/PSZ2v1.fits'
    PSZ2_cat = read_PSZ2(fn_PSZ2)
    matching_clusters = cross_match_frb_with_clusters(frb_sources, PSZ2_cat)
    print('Number of FRB sources matched with PSZ2 clusters: {}'.format(len(matching_clusters)))

    # WHY
    fndir = '/Users/liamconnor/work/projects/baryons/data/WHY_cluster_cat/'
    WHY_cat_1, WHY_cat_2, WHY_cat_3 = read_WHY_clustercat(fndir)
    clust_ind_match, frb_ind_match, bperp_match_arr = cross_match_frb_with_clusters(frb_sources, WHY_cat_1)
    print('FRB sources matched with WHY cat 1: {}'.format(len(matching_clusters)))
    clust_ind_match_WHY_1, frb_ind_match, bperp_match_arr = cross_match_frb_with_clusters(frb_sources, WHY_cat_2)
    print(frb_sources.iloc[frb_ind_match])
#    matching_clusters = cross_match_frb_with_clusters(frb_sources, WHY_cat_3)
#    print('Number of FRB sources matched with WHY clusters: {}'.format(len(matching_clusters)))



    # ROSAT
    fn_ROSAT = '/Users/liamconnor/work/projects/baryons/data/RXGCC_cluster_cat/table_rxgcc.fits'
    ROSAT_clusters = read_ROSAT(fn_ROSAT)
    clust_ind_match_ROSAT, frb_ind_match, bperp_match_arr = cross_match_frb_with_clusters(frb_sources, ROSAT_clusters)
    print(frb_sources.iloc[frb_ind_match])

    # Legacy
    fn_legacy = '/Users/kim/Research/FRB/FRB_catalogs/legacy/legacy_cluster_catalog.csv'
    legacy_clusters = read_legacy(fn_legacy)
    matching_clusters = cross_match_frb_with_clusters(frb_sources, legacy_clusters)
    print('Number of FRB sources matched with Legacy clusters: {}'.format(len(matching_clusters)))