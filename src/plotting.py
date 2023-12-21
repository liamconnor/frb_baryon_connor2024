import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from astropy import constants as con
from astropy import units as u


fn_frb_dsa='/Users/liamconnor/Desktop/dsafrbsnov23.csv'
fn_frb_nondsa='/Users/liamconnor/work/projects/baryons/data/nondsa_frbs_nov2023.csv'
fn_frb_dsa = '/Users/liamconnor/Desktop/dsa110_frbs_dec23.csv'

frb_sources_dsa = read_frb_catalog(fn_frb_dsa)
#frb_sources_nondsa = read_frb_catalog(fn_frb_nondsa)

def spline_zhangdm():
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

def get_1d_values(p, x):
    pcum = np.cumsum(p)
    pmedian = x[np.argmin(np.abs(pcum - 0.5))]
    p_p1sig = x[np.argmin(np.abs(pcum - 0.84))]
    p_n1sig = x[np.argmin(np.abs(pcum - 0.16))]
    p_p2sig = x[np.argmin(np.abs(pcum - 0.95))]
    p_n2sig = x[np.argmin(np.abs(pcum - 0.05))]
    pmean = np.nansum(x * p)
        
    return pmedian, p_p1sig, p_n1sig, p_n2sig, p_p2sig, pmean

def get_contours(p, x):
    nz = p.shape[1]
    arr = np.zeros([nz, 6])
    
    for ii in range(nz):
        pmedian, p_p1sig, p_n1sig, p_n2sig, p_p2sig, pmean = get_1d_values(p[:, ii], x)
        
        arr[ii, 0] = pmedian
        arr[ii, 1] = p_p1sig
        arr[ii, 2] = p_n1sig
        arr[ii, 3] = p_n2sig
        arr[ii, 4] = p_p2sig
        arr[ii, 5] = pmean
    
    return arr

def pdmigm(zfrb, dm, figm, C0, sigma, A=1., alpha=3., beta=3.):
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
    Kcon = 3 * con.c * cosmo.Ob(0) * cosmo.H0 / (8 * np.pi * con.G* con.m_p)
    K = Kcon.to(u.pc*u.cm**-3).value
    dmmean = K * figm * zfrb
    Delta = dm / dmmean
    p = A * np.exp(-(Delta**(-alpha) - C0) ** 2 / (
            2 * alpha**2 * sigma**2)) * Delta**(-beta)

    return p / np.nansum(p)

def create_cosmic_contour(figm = 0.83):
    zex = np.linspace(0.1, 2.0, 1000)
    dms = np.linspace(0.1, 2000, 1000)

    A_spl, C0_spl, sigmaDM_spl = spline_zhangdm()
    p = np.zeros([len(dms), len(zex)])

    for ii in range(len(zex)):
        z = zex[ii]
        C0_at_z, sigma_at_z, A_at_z = C0_spl(z), sigmaDM_spl(z), A_spl(z)
        p[:, ii] = pdmigm(z, dms, figm, C0_at_z, sigma_at_z, A=A_at_z, alpha=3., beta=3.)

    arr = get_contours(p, dms)    

    return arr

def plot_dmexcess_halos_dsa(frb_sources):

    zdsa = frb_sources['redshift'].values
    dmdsa = frb_sources['dm_exgal'].values

    zmod = np.linspace(0, 1.5, 100)

    ind = np.where((zdsa != -1) & (dmdsa > 0) & (np.abs(zdsa) > 0.01))[0]
    frb_sources = frb_sources.iloc[ind]

    zdsa = np.abs(zdsa[ind])
    dmdsa = dmdsa[ind]
    dmhalo = 30.

#    arr = create_cosmic_contour(figm = 0.83) 
    arr = np.load('contours_zdm_plot.npy')
    zex = np.load('./zex_contourplot.npy')

    dmmean = arr[:, 5]#(coeff * zdsa)
    
    dmmean_mod = []
    [dmmean_mod.append(dmmean[np.argmin(np.abs(zex - z))]) for z in zdsa]
    dmexcess_frac = (dmdsa - dmhalo)/dmmean_mod

    zmod = zex
    dmexcess_frac_model1 = arr[:, 0] / arr[:, 5]
    dmexcess_frac_model2 = arr[:, 1] / arr[:, 5]
    dmexcess_frac_model3 = arr[:, 2] / arr[:, 5]
    dmexcess_frac_model4 = arr[:, 3] / arr[:, 5]
    dmexcess_frac_model5 = arr[:, 4] / arr[:, 5]

    ind_host_names = ['ansel', 
                      'etienne', 
                      'ada', 
                      'ayo', 
                      'leonidas', 
                      'ferb', 
                      'phineas', 
                      'fatima', 
                      'martha', 
                      'alex']
    

    ind_host_names = list(frb_sources_dsa[np.abs(frb_sources_dsa['rm']) > 400.]['name'].values)

    ind_halo_names = ['fen', 'nihari', 'elektra', 'mifanshan', 'jackie', 'gemechu']

    ind_hostdm = np.where(frb_sources['name'].isin(ind_host_names))[0]
    ind_halodm = np.where(frb_sources['name'].isin(ind_halo_names))[0]
    ind_neither = range(len(zdsa))

    c1, c2, c3 = 'lavender', 'lightgreen', 'lightsalmon' # lemonchiffon
    c1, c2, c3 = 'lightblue', 'orange', '#89A54E'

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.plot(zmod, dmexcess_frac_model1, color='k', linestyle='--', linewidth=1.5, zorder=1)
    ax.plot(zmod, dmexcess_frac_model2, color='k', linestyle='--', linewidth=1.5, zorder=2)
    ax.plot(zmod, dmexcess_frac_model3, color='k', linestyle='--', linewidth=1.5, zorder=4)
    ax.plot(zmod, dmexcess_frac_model4, color='k', linestyle='--', linewidth=1.5, zorder=5)
    ax.plot(zmod, dmexcess_frac_model5, color='k', linestyle='--', linewidth=1.5, zorder=6)
    ax.plot(zmod, np.ones_like(zmod), color='k', linestyle='--', linewidth=1, zorder=7)

    ax.scatter(zdsa[ind_neither], dmexcess_frac[ind_neither], zorder=7,
               s=100, color=c1, edgecolors='k', label='DSA-110 FRBs', alpha=1)
    ax.scatter(zdsa[ind_hostdm], dmexcess_frac[ind_hostdm], zorder=8, c=c2,
               s=100, edgecolors='k', label='Large Rotation Measure', alpha=1,)
    ax.scatter(zdsa[ind_halodm], dmexcess_frac[ind_halodm], zorder=9, alpha=1,
               s=100, color=c3, edgecolors='k', label='Massive group / Cluster')
    ax.legend()
    ax.grid('on', alpha=0.5, linestyle='--')
    ax.set_xlabel('Redshift', fontsize=18)
    ax.set_ylabel(r'$\Delta_{\rm DM}$', fontsize=18)

    for nameii in ind_halo_names:
        ind_nameii = np.where(frb_sources['name'] == nameii)[0]
        zii = np.abs(frb_sources['redshift'].values[ind_nameii])
        text = ax.text(zii, dmexcess_frac[ind_nameii], nameii, 
                       fontsize=12, color='k', zorder=10)

def plot_dmexcess_halos_nondsa(frb_sources, in_footprint):
    frb_sources = read_frb_catalog(fn_frb_nondsa)

    zfrb = frb_sources['redshift'].values
    dmfrb = frb_sources['dm_exgal'].values

#    frb_sources = frb_sources.iloc[np.array(in_footprint)]
    dmhalo = 30.

    arr = np.load('contours_zdm_plot.npy')
    zex = np.load('./zex_contourplot.npy')

    dmmean = arr[:, 5]#(coeff * zdsa)
    
    dmmean_mod = []
    [dmmean_mod.append(dmmean[np.argmin(np.abs(zex - z))]) for z in zfrb]
    dmexcess_frac = (dmfrb - dmhalo)/dmmean_mod

    zmod = zex
    dmexcess_frac_model1 = arr[:, 0] / arr[:, 5]
    dmexcess_frac_model2 = arr[:, 1] / arr[:, 5]
    dmexcess_frac_model3 = arr[:, 2] / arr[:, 5]
    dmexcess_frac_model4 = arr[:, 3] / arr[:, 5]
    dmexcess_frac_model5 = arr[:, 4] / arr[:, 5]

    ind_host_names = ['FRB20190520B', 'FRB20121102A', 'FRB20201124A']
    ind_halo_names = ['FRB20190520B', 'FRB20220610A', 'FRB20211127I', 'FRB20200906A']
    ind_hostdm = np.where(frb_sources['name'].isin(ind_host_names))[0]
    ind_halodm = np.where(frb_sources['name'].isin(ind_halo_names))[0]
    ind_neither = range(len(zfrb))

    c1, c2, c3 = 'white', 'lightblue', 'lightsalmon' # lemonchiffon

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
#    ax.plot(zmod, np.ones_like(zmod), color='k', linestyle='--', linewidth=1, zorder=1)
#    ax.plot(zmod, 1 + 5 * np.exp(-2*zmod), color='k', linestyle='--', linewidth=1, zorder=2)
#    ax.plot(zmod, 1 - 1.5 * np.exp(-2*zmod), color='k', linestyle='--', linewidth=1, zorder=3)
    ax.scatter(zfrb[ind_neither], dmexcess_frac[ind_neither], zorder=4,
               s=100, color=c1, edgecolors='k', label='DSA-110 FRBs', alpha=1)
    ax.scatter(zfrb[ind_hostdm], dmexcess_frac[ind_hostdm], zorder=5,
               s=100, color=c2, edgecolors='k', label='Large host RM', alpha=1)
    ax.scatter(zfrb[ind_halodm], dmexcess_frac[ind_halodm], zorder=6,
               s=100, color=c3, edgecolors='k', label='Massive group / Cluster')
    ax.legend()
    ax.grid('on', alpha=0.5, linestyle='--')
    ax.set_xlabel('Redshift', fontsize=18)
    ax.set_ylabel(r'$\Delta_{\rm DM}$', fontsize=18)
