import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.io import fits
from astropy.cosmology import Planck18 as cosmo

import reader

# Define a function to cross-match FRB sources with a galaxy cluster catalog
def cross_match_frb_with_clusters(frb_sources, cluster_catalog,
                                  thresh_bperp_mpc=1.5, 
                                  cluster_zmax=None):
    """ Cross match FRB sources with a galaxy cluster catalog
    
    Parameters
    ----------
    frb_sources : pandas DataFrame
        DataFrame containing FRB sources
    cluster_catalog : pandas DataFrame
        DataFrame containing galaxy cluster catalog
    thresh_bperp_mpc : float
        Maximum impact parameter in Mpc
    cluster_zmax : float or None
        Maximum redshift of clusters to consider

    Returns
    -------
    cluster_catalog.iloc[clust_ind_match] : pandas DataFrame
        DataFrame containing matched clusters
    frb_ind_match : list
        List of FRB indexes that match with clusters
    bperp_match_arr : list
        List of impact parameters for each FRB-cluster match
    in_footprint : list
        List of FRB indexes that are in the footprint of the cluster catalog
    """
    if cluster_catalog is None:
        return None
    
    if cluster_zmax is not None:
        ind_zmax = np.where(cluster_catalog['redshift'] < cluster_zmax)[0]
        cluster_catalog = cluster_catalog.iloc[ind_zmax]

    D_A_frb = cosmo.angular_diameter_distance(\
                    np.abs(frb_sources['redshift'].values)).value

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
        D_A_clust = cosmo.angular_diameter_distance(\
                        cluster_catalog['redshift'].iloc[ind_close].values).value

        bperp = np.pi / 180. * sep_deg.iloc[ind_close].values * D_A_clust
        
        # Find indexes that are within 2 Mpc of an FRB sightline
        clust_ind_match_ii = np.where((bperp < thresh_bperp_mpc)\
                                    & (D_A_clust < 1.1 * D_A_frb[ii]))[0]

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

    return cluster_catalog.iloc[clust_ind_match], frb_ind_match, bperp_match_arr, in_footprint

def read_legacy(fn_legacy, logM_min=13.5):
    """ Read in the Legacy catalog and return a pandas DataFrame

    Parameters
    ----------
    fn_legacy : str
        Filename of the Legacy catalog
    logM_min : float
        Minimum logM500 mass to consider
    
    Returns
    -------
    df : pandas DataFrame
        DataFrame containing Legacy clusters
    """
    d = np.load(fn_legacy)
    ind_logM = np.where(d[5] > logM_min)[0]
    group_idx, richness, ragr, decgr, zgr, logMgr, Lgr = d[0, ind_logM],\
                                                         d[1, ind_logM],\
                                                         d[2, ind_logM],\
                                                         d[3, ind_logM],\
                                                         d[4, ind_logM],\
                                                         d[5, ind_logM],\
                                                         d[6, ind_logM]

    group_idx_str = ['DESIDR9_' + s for s in group_idx.astype(str)]

    data = {
        'name': group_idx_str,
        'ra': ragr,
        'dec': decgr,
        'redshift': zgr,
        'm500': 10**logMgr,
        'richness': richness,
            }

    df = pd.DataFrame(data)

    return df

def read_PSZ2(fn_PSZ2):
    """ Read in the PSZ2 catalog fits file 
    and return a pandas DataFrame

    Parameters
    ----------
    fn_PSZ2 : str
        Filename of the PSZ2 catalog
    
    Returns
    -------
    df : pandas DataFrame
        DataFrame containing PSZ2 clusters
    """
    f = fits.open(fn_PSZ2)

    data = f[1].data

    ra, dec, redshift, m500, y5r500 = [], [], [], [], []
    names = []

    for i in range(len(data)):
        if data[i][-4]==0:
            continue
        names.append(data[i][1])
        ra.append(data[i][2])
        dec.append(data[i][3])
        redshift.append(data[i][-4])
        m500.append(data[i][-2] * 1e14) # SZ mass estimate
        y5r500.append(data[i][-5]) # The mean marginal Y5R500 for 
                                   # the SZ source detection as determined 
                                   # by the reference pipeline, in units of arcmin2

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
    """ Read in the WHY cluster catalog and return a pandas DataFrame

    Parameters
    ----------
    fndir : str
        Directory containing the WHY cluster catalog
    
    Returns
    -------
    df_1 : pandas DataFrame
        DataFrame containing WHY cluster catalog 1
    """
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

    Parameters
    ----------
    fn_xclass : str
        Filename of the XClass catalog
    
    Returns
    -------
    df : pandas DataFrame
        DataFrame containing XClass clusters
    """
    f = fits.open(fn_xclass)
    data = f[1].data

    name, ra, dec, redshift, m500 = [], [], [], [], []

    for ii in range(len(data)):
        # Example line: 'names':['XClass','RAJ2000','DEJ2000','RAmdeg','DEmdeg','z','f_z','MLdet']
        # (20, 193.438, 10.1954, 193.438, 10.1951, 0.654, '  confirmed', 360.664)
        name.append('xclass_' + str(data[ii][0]))
        ra.append(data[ii][3])
        dec.append(data[ii][4])
        redshift.append(data[ii][5])
        m500.append(-1)

    data = {
        'name': name,
        'ra': np.array(ra).astype(float),
        'dec': np.array(dec).astype(float),
        'redshift': np.array(redshift).astype(float),
        'm500': np.array(m500),
    }

    df = pd.DataFrame(data)
    ind_keep = np.where(df['redshift'] != 0)[0]

    return df.iloc[ind_keep]

def read_MCXC(fn_mcxc='mcxc.fits'):
    """ Read in the MCXC catalog and return a pandas DataFrame

    Parameters
    ----------
    fn_mcxc : str
        Filename of the MCXC catalog
    
    Returns
    -------
    df : pandas DataFrame
        DataFrame containing MCXC clusters
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

    Parameters
    ----------
    fn_ROSAT : str
        Filename of the ROSAT catalog

    Returns
    -------
    df : pandas DataFrame
        DataFrame containing ROSAT clusters
    """
    f = fits.open(fn_ROSAT)
    data = f[1].data

    name, ra, dec, redshift, m500, r500 = [], [], [], [], [], []

    for i in range(len(data)):
        name.append('ROSAT_' + data[i][0])
        ra.append(data[i][1])
        dec.append(data[i][2])
        redshift.append(data[i][5])
        m500.append(data[i][-13]*1e14)
        r500.append(data[i][14])

    data = { 
        'name': name,
        'ra': ra,
        'dec': dec,
        'redshift': redshift,
        'm500': m500,
        'r500_mpc': r500,
    }

    df = pd.DataFrame(data)

    return df

def read_frb_catalog(fn_frb):
    """ Read in the FRB catalog and return a pandas DataFrame
    """
    # Index(['name', 'mjd', 'snr_heim', 'dm_heim', 'dm_opt', 'dm_exgal', 'ibox', 'redshift', 'ra', 'dec']
    frb_sources = pd.read_csv(fn_frb, delim_whitespace=False)

    return frb_sources

def create_frbcluster_dataframe(frb_sources_match, 
                                cluster_sources_match, 
                                bperp_match_arr):
    """ Create a pandas DataFrame containing matched FRB-cluster pairs

    Parameters
    ----------
    frb_sources_match : pandas DataFrame
        DataFrame containing matched FRB sources
    cluster_sources_match : pandas DataFrame
        DataFrame containing matched cluster sources
    bperp_match_arr : list
        List of impact parameters for each FRB-cluster match (Mpc)

    Returns
    -------
    df : pandas DataFrame
        DataFrame containing matched FRB-cluster pairs
    """
    
    if len(frb_sources_match) != len(cluster_sources_match):
        print('Error: FRB and cluster catalogs are not the same length')
    elif len(frb_sources_match) == 0:
        print('Error: No FRB sources matched with clusters')
        return None

    datax = {
        'frb_name': frb_sources_match['name'].values,
        'frb_dm_exgal': frb_sources_match['dm_exgal'].values,
        'frb_ra': frb_sources_match['ra'].values,
        'frb_dec': frb_sources_match['dec'].values,   
        'cluster_ra': cluster_sources_match['ra'].values,
        'cluster_dec': cluster_sources_match['dec'].values,
        'frb_redshift': frb_sources_match['redshift'].values,
        'cluster_redshift': cluster_sources_match['redshift'].values,
        'cluster_name': cluster_sources_match['name'].values,
        'cluster_m500': cluster_sources_match['m500'].values,
        'b_perp_mpc': np.array(bperp_match_arr),
    }

    df = pd.DataFrame(datax)

    return df 

def read_CHIME(fn_CHIME):
    """ Read in the CHIME catalog and return a pandas DataFrame

    Parameters
    ----------
    fn_CHIME : str
        Filename of the CHIME catalog

    Returns
    -------
    df : pandas DataFrame
        DataFrame containing CHIME FRB sources
    """
    datapd = pd.read_csv(fn_CHIME)
    ind = range(len(datapd))#np.where(datapd['repeater_name']=='-9999')[0]
    ra_ch, dec_ch = datapd['ra'][ind].values, datapd['dec'][ind].values
    ra_ch_err, dec_ch_err = datapd['ra_err'][ind].values, datapd['dec_err'][ind].values
    try:
        dm_ch = datapd['dm_exc_ne2001'][ind].values
    except:
        dm_ch = datapd['dm'][ind].values
    names_ch = datapd['tns_name'][ind].values

    redshift_estimate = (dm_ch - 30) / 850

    for jj in range(len(redshift_estimate)):
        if redshift_estimate[jj] < 0:
            redshift_estimate[jj] = 0.

    data = {
        'name': names_ch,
        'dm_exgal' : dm_ch,
        'ra': ra_ch,
        'dec': dec_ch,
        'ra_err': ra_ch_err,
        'dec_err': dec_ch_err,
        'redshift': redshift_estimate
    }

    df = pd.DataFrame(data)

    return df

def cross_match_all(frb_sources, thresh_bperp_mpc=1.5, 
                    cluster_zmax=None, min_mass_legacy=14.):
    """ Cross match FRB sources with all cluster catalogs

    Parameters
    ----------
    frb_sources : pandas DataFrame
        DataFrame containing FRB sources
    thresh_bperp_mpc : float
        Maximum impact parameter to be considered a match in Mpc
    cluster_zmax : float or None
        Maximum redshift of clusters to consider

    Returns
    -------
    match_dataframe : pandas DataFrame
        DataFrame containing matched FRB-cluster pairs
    """
    match_dataframe = pd.DataFrame()

    in_footprint_total = []

    print("Trying PSZ2")
    # PSZ2
    fn_PSZ2 = '/Users/liamconnor/work/projects/baryons/data/PSZ2_cat.fits'
    PSZ2_cat = read_PSZ2(fn_PSZ2)
    _ = cross_match_frb_with_clusters(frb_sources, PSZ2_cat, 
                                      thresh_bperp_mpc=thresh_bperp_mpc, 
                                      cluster_zmax=cluster_zmax)

    clust_match_PSZ2, frb_ind_match, bperp_match_arr, in_footprint = _

    match_dataframe_PSZ2 = create_frbcluster_dataframe(frb_sources.iloc[frb_ind_match], 
                                                    clust_match_PSZ2, 
                                                    bperp_match_arr)
    match_dataframe = pd.concat([match_dataframe, match_dataframe_PSZ2], 
                                ignore_index=True)
    
    # Keep track of which FRBs are in the footprint of each catalog
    if len(in_footprint):
        in_footprint_total.append(in_footprint)

    try:
        # WHY
        fndir = '/Users/liamconnor/work/projects/baryons/data/WHY_cluster_cat/'
        WHY_cat_1, WHY_cat_2, WHY_cat_3 = read_WHY_clustercat(fndir)
        clust_ind_match_WHY_1, frb_ind_match, bperp_match_arr = cross_match_frb_with_clusters(frb_sources, WHY_cat_1)
        clust_ined_match_WHY_2, frb_ind_match, bperp_match_arr = cross_match_frb_with_clusters(frb_sources, WHY_cat_2)
    except:
        pass

    print("Trying MCXC Cluster cat")
    # MCXC
    fn_mcxc = '/Users/liamconnor/work/projects/baryons/data/MCXC/mcxc.fits'
    MCXC_clusters = read_MCXC(fn_mcxc)
    _ = cross_match_frb_with_clusters(frb_sources, MCXC_clusters, 
                                      thresh_bperp_mpc, cluster_zmax)
    clust_match_MCXC, frb_ind_match, bperp_match_arr, in_footprint = _

    match_dataframe_mcxc = create_frbcluster_dataframe(frb_sources.iloc[frb_ind_match], 
                                                        clust_match_MCXC, 
                                                        bperp_match_arr)
    
    match_dataframe = pd.concat([match_dataframe, match_dataframe_mcxc], 
                                ignore_index=True)
    
    # Keep track of which FRBs are in the footprint of each catalog
    if len(in_footprint):
        in_footprint_total.append(in_footprint)

    # ROSAT
    fn_ROSAT = '/Users/liamconnor/work/projects/baryons/data/RXGCC_cluster_cat/table_rxgcc.fits'
    ROSAT_clusters = read_ROSAT(fn_ROSAT)

    print("Trying ROSAT RXGCC")
    _ = cross_match_frb_with_clusters(frb_sources, 
                                      ROSAT_clusters, 
                                      thresh_bperp_mpc=thresh_bperp_mpc,
                                      cluster_zmax=cluster_zmax)
    clust_match_ROSAT, frb_ind_match, bperp_match_arr, in_footprint = _
    
    match_dataframe_rosat = create_frbcluster_dataframe(frb_sources.iloc[frb_ind_match], 
                                                        clust_match_ROSAT, 
                                                        bperp_match_arr)
    
    match_dataframe = pd.concat([match_dataframe, match_dataframe_rosat], 
                                ignore_index=True)
    
    # Keep track of which FRBs are in the footprint of each catalog
    if len(in_footprint):
        in_footprint_total.append(in_footprint)

    print("Trying Xclass now")
    # XClass
    fn_xclass = '/Users/liamconnor/work/projects/baryons/data/xclass/Xclass_cat.fit'
    xclass_clusters = read_xclass(fn_xclass)
    _ = cross_match_frb_with_clusters(frb_sources, 
                                      xclass_clusters,
                                      thresh_bperp_mpc=thresh_bperp_mpc,
                                      cluster_zmax=cluster_zmax)
    
    clust_match_xclass, frb_ind_match, bperp_match_arr, in_footprint = _

    match_dataframe_xclass = create_frbcluster_dataframe(frb_sources.iloc[frb_ind_match], 
                                                        clust_match_xclass, 
                                                        bperp_match_arr)
    
    match_dataframe = pd.concat([match_dataframe, 
                                 match_dataframe_xclass], 
                                ignore_index=True)

    # Keep track of which FRBs are in the footprint of each catalog
    if len(in_footprint):
        in_footprint_total.append(in_footprint)

    print("Trying Legacy now")
    # Legacy 'DESIDR9_NGC_group_12p5Msun.npy'
    fn_legacy = '/Users/liamconnor/work/projects/baryons/data/DESIDR9/DESIDR9_NGC_group_12p5Msun.npy'
    legacy_clusters = read_legacy(fn_legacy, logM_min=min_mass_legacy)
    _ = cross_match_frb_with_clusters(frb_sources, legacy_clusters,
                                      thresh_bperp_mpc=thresh_bperp_mpc,
                                      cluster_zmax=cluster_zmax)

    clust_match_legacy, frb_ind_match, bperp_match_arr, in_footprint = _
    
    match_dataframe_legacy = create_frbcluster_dataframe(frb_sources.iloc[frb_ind_match], 
                                                        clust_match_legacy, 
                                                        bperp_match_arr)
    
    match_dataframe = pd.concat([match_dataframe, 
                                 match_dataframe_legacy], 
                                ignore_index=True)
    
    # Keep track of which FRBs are in the footprint of each catalog
    if len(in_footprint):
        in_footprint_total.append(in_footprint)

    return match_dataframe, in_footprint

fn_frb_dsa='/Users/liamconnor/Desktop/allfrbs_13march24y.csv'
fn_frb_dsa='../data/allfrbs_naturesample.csv'
#fn_frb_nondsa='/Users/liamconnor/work/projects/baryons/data/frbdata/nondsa_frbs_nov2023.csv'
#fn_frb_dsa = '/Users/liamconnor/Desktop/dsa110_frbs_dec23.csv'
#fn_CHIME='/Users/liamconnor/work/projects/frb/chime_cgm/data/chimefrbcat1.csv'
#fn_CHIME = 'CHIME/chime_basecat1_catalog.csv'

#frb_sources = read_frb_catalog(fn_frb_dsa)
frb_sources = reader.read_frb_catalog(fn_frb_dsa)

#frb_sources = read_CHIME(fn_CHIME)
#frb_sources = frb_sources[frb_sources['redshift'] > 0.5]
match_dataframe, in_footprint = cross_match_all(frb_sources,
                                                thresh_bperp_mpc=1.5,
                                                cluster_zmax=None,
                                                min_mass_legacy=12.5)

# scatter(ROSAT_clusters['ra'], ROSAT_clusters['dec'], s=1, color='C0', label='ROSAT')
# scatter(MCXC_clusters['ra'], MCXC_clusters['dec'], s=1, color='C1', label='MCXC')
# scatter(PSZ2_cat['ra'], PSZ2_cat['dec'], s=1, color='k', alpha=0.2, label='PSZ2')
# scatter(legacy_clusters['ra'], legacy_clusters['dec'], s=1, color='C2', label='Legacy')

