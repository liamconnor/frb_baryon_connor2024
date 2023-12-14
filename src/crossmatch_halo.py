import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.io import fits
from astropy.cosmology import Planck18 as cosmo


# Define a function to cross-match FRB sources with a galaxy cluster catalog
def cross_match_frb_with_clusters(frb_sources, cluster_catalog, thresh_bperp_mpc=2.0):
    if cluster_catalog is None:
        return None

    D_A_frb = cosmo.angular_diameter_distance(np.abs(frb_sources['redshift'].values)).value

    clust_ind_match = []
    frb_ind_match = []
    bperp_match_arr = []
    in_footprint = []

    for ii in range(len(frb_sources)):
        dra = np.abs(frb_sources['ra'].iloc[ii] - cluster_catalog['ra'])
        ddec = np.abs(frb_sources['dec'].iloc[ii] - cluster_catalog['dec'])
        sep_deg = np.sqrt(dra**2 * np.cos(np.deg2rad(frb_sources['dec'].iloc[ii]))**2 + ddec**2)

        # Get cluster indexes that are within 5 deg of the FRB sightline
        ind_close = np.where(sep_deg < 5.0)[0]

        if np.min(sep_deg) > 2.0:
            continue
        else:
            in_footprint.append(ii)

        # Calculate the angular diameter distance to each FRB source and cluster
        D_A_clust = cosmo.angular_diameter_distance(cluster_catalog['redshift'].iloc[ind_close].values).value

        bperp = np.pi / 180. * sep_deg[ind_close].values * D_A_clust
        
        # Find indexes that are within 2 Mpc of an FRB sightline
        clust_ind_match_ii = np.where((bperp < thresh_bperp_mpc) & (D_A_clust < 1.1 * D_A_frb[ii]))[0]

        if len(clust_ind_match_ii) == 0:
            continue
        elif len(clust_ind_match_ii) == 1:
            clust_ind_match.append(ind_close[clust_ind_match_ii[0]])
            frb_ind_match.append(ii)
            bperp_match_arr.append(bperp[clust_ind_match_ii[0]])
        elif len(clust_ind_match_ii) > 1:
            for nn in range(len(clust_ind_match_ii)):
                clust_ind_match.append(ind_close[clust_ind_match_ii[nn]])
                frb_ind_match.append(ii)
                bperp_match_arr.append(bperp[clust_ind_match_ii[nn]])

    return clust_ind_match, frb_ind_match, bperp_match_arr, in_footprint

def read_legacy(fn_legacy, logM_min=13.5):
    d = np.load(fn_legacy)
    ind_logM = np.where(d[5] > logM_min)[0]
    group_idx, richness, ragr, decgr, zgr, logMgr, Lgr = d[0, ind_logM],\
                                                         d[1, ind_logM],\
                                                         d[2, ind_logM],\
                                                         d[3, ind_logM],\
                                                         d[4, ind_logM],\
                                                         d[5, ind_logM],\
                                                         d[6, ind_logM]

    data = {
        'name': group_idx,
        'ra': ragr,
        'dec': decgr,
        'redshift': zgr,
        'm500': 10**logMgr,
        'richness': richness,
            }

    df = pd.DataFrame(data)

    return df

def cross_match_Legacy(frb_sources, fn_legacy):
    pass 

def read_PSZ2(fn_PSZ2):
    """ Read in the PSZ2 catalog fits file 
    and return a pandas DataFrame
    """
    f = fits.open(fn_PSZ2)

    data = f[1].data

    ra, dec, redshift, m500, r500, y5r500 = [], [], [], [], [], []
    names = []

    for i in range(len(data)):
        if data[i][-4]==0:
            continue
        names.append(data[i][1])
        ra.append(data[i][2])
        dec.append(data[i][3])
        redshift.append(data[i][-4])
        m500.append(data[i][-2])
        y5r500.append(data[i][-5])

    data = {
        'name' : names, 
        'ra': ra,
        'dec': dec,
        'redshift': redshift,
        'm500': m500,
        'Y5R500' : y5r500,        
    }   

    df = pd.DataFrame(data)

    return df

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
    """ Read in the XClass catalog and return a pandas DataFrame
    
    Catalog paper: https://www.aanda.org/articles/aa/full_html/2021/08/aa40566-21/aa40566-21.html

    """
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
    ind_keep = np.where(df['redshift'] != 0)[0]

    return df.iloc[ind_keep]

def read_MCXC(fn_mcxc='mcxc.fits'):
    """ Read in the MCXC catalog and return a pandas DataFrame
    """
    f = fits.open(fn_mcxc)
    data = f[1].data

    ra, dec, redshift, names, m500, r500 = [], [], [], [], [], []

    for i in range(len(data)):
        ra_str = data[i][2]
        dec_str = data[i][3]

        ra_deg = 15 * (float(ra_str[0:2]) + float(ra_str[3:5])/60. + float(ra_str[6:])/3600.)
        dec_deg = np.sign(float(dec_str[0:3])) * (np.abs(float(dec_str[0:3])) + 
                                                  float(dec_str[4:6])/60. + 
                                                  float(dec_str[7:])/3600.)

        ra.append(ra_deg)
        dec.append(dec_deg)
        redshift.append(data[i][4])
        names.append(data[i][0])
        m500.append(data[i][6]*1e14)
        r500.append(data[i][7])

    data = {
        'name' : names,
        'ra': ra,
        'dec': dec,
        'redshift': redshift,
        'm500': m500,
        'r500_mpc': r500,
    }

    df = pd.DataFrame(data)

    return df

def read_ROSAT(fn_ROSAT='table_rxgcc.fits'):
    """ Read in the ROSAT catalog and return a pandas DataFrame
    https://www.aanda.org/articles/aa/abs/2022/02/aa40908-21/aa40908-21.html
    """
    f = fits.open(fn_ROSAT)
    data = f[1].data

    ra, dec, redshift, m500, r500 = [], [], [], [], []

    for i in range(len(data)):
        ra.append(data[i][1])
        dec.append(data[i][2])
        redshift.append(data[i][5])
        m500.append(data[i][-13]*1e14)
        r500.append(data[i][14])

    data = { 
        'ra': ra,
        'dec': dec,
        'redshift': redshift,
        'm500': m500,
        'r500_mpc': r500,
    }

    df = pd.DataFrame(data)

    return df

def cross_match_ROSAT(frb_sources, fn_ROSAT):
    pass

def read_frb_catalog(fn_frb):
    # Index(['name', 'mjd', 'snr_heim', 'dm_heim', 'dm_opt', 'dm_exgal', 'ibox', 'redshift', 'ra', 'dec']
    frb_sources = pd.read_csv(fn_frb, delim_whitespace=False)

    return frb_sources

def create_frbcluster_dataframe(frb_sources_match, 
                                cluster_sources_match, 
                                bperp_match_arr):
    
    if len(frb_sources_match) != len(cluster_sources_match):
        print('Error: FRB and cluster catalogs are not the same length')
    elif len(frb_sources_match) == 0:
        print('Error: No FRB sources matched with clusters')
        return None

    datax = {
        'frb_name': frb_sources_match['name'].values,
        'frb_ra': frb_sources_match['ra'].values,
        'frb_dec': frb_sources_match['dec'].values,   
        'frb_redshift': frb_sources_match['redshift'].values,
        'frb_dm_exgal': frb_sources_match['dm_exgal'].values,
        'cluster_name': cluster_sources_match['name'].values,
        'cluster_ra': cluster_sources_match['ra'].values,
        'cluster_dec': cluster_sources_match['dec'].values,
        'cluster_redshift': cluster_sources_match['redshift'].values,
        'cluster_m500': cluster_sources_match['m500'].values,
        'b_perp_mpc': np.array(bperp_match_arr),
    }

    df = pd.DataFrame(datax)

    return df 

def cross_match_all(fn_frb):
    fn_frb_dsa='/Users/liamconnor/Desktop/dsafrbsnov23.csv'
    fn_frb_nondsa='/Users/liamconnor/work/projects/baryons/data/frbdata/nondsa_frbs_nov2023.csv'

    frb_sources_dsa = read_frb_catalog(fn_frb_dsa)
    frb_sources = read_frb_catalog(fn_frb_nondsa)

    # PSZ2
    fn_PSZ2 = '/Users/liamconnor/work/projects/baryons/data/PSZ2_cat.fits'
    PSZ2_cat = read_PSZ2(fn_PSZ2)
    clust_ind_match_PSZ2, frb_ind_match, bperp_match_arr, in_footprint = cross_match_frb_with_clusters(frb_sources, 
                                                                                                       PSZ2_cat)
    match_dataframe = create_frbcluster_dataframe(frb_sources.iloc[frb_ind_match], 
                                PSZ2_cat.iloc[clust_ind_match_PSZ2], 
                                bperp_match_arr)

    # WHY
    fndir = '/Users/liamconnor/work/projects/baryons/data/WHY_cluster_cat/'
    WHY_cat_1, WHY_cat_2, WHY_cat_3 = read_WHY_clustercat(fndir)
    clust_ind_match_WHY_1, frb_ind_match, bperp_match_arr = cross_match_frb_with_clusters(frb_sources, WHY_cat_1)
    print(frb_sources.iloc[frb_ind_match])
    print(WHY_cat_1.iloc[clust_ind_match_WHY_1])

    clust_ind_match_WHY_2, frb_ind_match, bperp_match_arr = cross_match_frb_with_clusters(frb_sources, WHY_cat_2)
    print(frb_sources.iloc[frb_ind_match])
#    matching_clusters = cross_match_frb_with_clusters(frb_sources, WHY_cat_3)
#    print('Number of FRB sources matched with WHY clusters: {}'.format(len(matching_clusters)))

    # MCXC
    fn_mcxc = '/Users/liamconnor/work/projects/baryons/data/MCXC/mcxc.fits'
    MCXC_clusters = read_MCXC(fn_mcxc)
    clust_ind_match_MCXC, frb_ind_match, bperp_match_arr, in_footprint = cross_match_frb_with_clusters(frb_sources, MCXC_clusters)
    print(frb_sources.iloc[frb_ind_match])
    print(MCXC_clusters.iloc[clust_ind_match_MCXC])
    print(bperp_match_arr)
    match_dataframe = create_frbcluster_dataframe(frb_sources.iloc[frb_ind_match], 
                                MCXC_clusters.iloc[clust_ind_match_MCXC], 
                                bperp_match_arr)

    # ROSAT
    fn_ROSAT = '/Users/liamconnor/work/projects/baryons/data/RXGCC_cluster_cat/table_rxgcc.fits'
    ROSAT_clusters = read_ROSAT(fn_ROSAT)
    clust_ind_match_ROSAT, frb_ind_match, bperp_match_arr, in_footprint = cross_match_frb_with_clusters(frb_sources, 
                                                                                                        ROSAT_clusters)
    print(frb_sources.iloc[frb_ind_match])
    print(ROSAT_clusters.iloc[clust_ind_match_ROSAT])
    print(bperp_match_arr)
    match_dataframe = create_frbcluster_dataframe(frb_sources.iloc[frb_ind_match], 
                                ROSAT_clusters.iloc[clust_ind_match_ROSAT], 
                                bperp_match_arr)

    # XClass
    fn_xclass = '/Users/liamconnor/work/projects/baryons/data/xclass/Xclass_cat.fit'
    xclass_clusters = read_xclass(fn_xclass)
    clust_ind_match_xclass, frb_ind_match, bperp_match_arr, in_footprint = cross_match_frb_with_clusters(frb_sources, 
                                                                                                         xclass_clusters)
    print(frb_sources.iloc[frb_ind_match])
    match_dataframe = create_frbcluster_dataframe(frb_sources.iloc[frb_ind_match], 
                                xclass_clusters.iloc[clust_ind_match_xclass], 
                                bperp_match_arr)

    # Legacy 'DESIDR9_NGC_group_12p5Msun.npy'
    fn_legacy = '/Users/liamconnor/work/projects/baryons/data/DESIDR9/DESIDR9_allsky_group_12p5Msun.npy'
    legacy_clusters = read_legacy(fn_legacy, logM_min=13.5)
    clust_ind_match_legacy, frb_ind_match, bperp_match_arr, in_footprint = cross_match_frb_with_clusters(frb_sources, 
                                                                                                         legacy_clusters)
    match_dataframe = create_frbcluster_dataframe(frb_sources.iloc[frb_ind_match], 
                                legacy_clusters.iloc[clust_ind_match_legacy], 
                                bperp_match_arr)
