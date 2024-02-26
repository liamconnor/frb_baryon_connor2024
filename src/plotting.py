import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from astropy import constants as con
from astropy import units as u

plt.rcParams.update({
                    'font.size': 12,
                    'font.family': 'serif',
                    'axes.labelsize': 14,
                    'axes.titlesize': 15,
                    'xtick.labelsize': 12,
                    'ytick.labelsize': 12,
                    'xtick.direction': 'in',
                    'ytick.direction': 'in',
                    'xtick.top': True,
                    'ytick.right': True,
                    'lines.linewidth': 0.5,
                    'lines.markersize': 5,
                     'legend.borderaxespad': 1,
                     'legend.frameon': True,
                     'legend.loc': 'lower right'})

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
    msize = 60
    c1, c2, c3 = 'lightblue', 'orange', 'red'
    fn_frb_dsa = '/Users/liamconnor/Desktop/dsa110_frbs_dec23.csv'
    marker1='o'
    marker2='s'

    frb_sources_dsa = read_frb_catalog(fn_frb_dsa)

    zdsa = frb_sources_dsa['redshift'].values
    dmdsa = frb_sources_dsa['dm_exgal'].values

    zmod = np.linspace(0, 1.5, 100)

    ind = np.where((zdsa != -1) & (dmdsa > 0) & (np.abs(zdsa) > 0.025))[0]
    frb_sources_dsa = frb_sources_dsa.iloc[ind]

    zdsa = np.abs(zdsa[ind])
    dmdsa = dmdsa[ind]
    dmhalo = 30.
    
    arr = np.load('contours_zdm_plot.npy')
    zex = np.load('./zex_contourplot.npy')

    for ii in range(6):
        f = UnivariateSpline(zex,arr[:, ii])
        arr[:, ii] = f(zex)

    dmmean = arr[:, 0]#(coeff * zdsa)
    
    dmmean_mod = []
    [dmmean_mod.append(dmmean[np.argmin(np.abs(zex - z))]) for z in zdsa]
    dmexcess_frac = (dmdsa - dmhalo)/dmmean_mod

    zmod = zex
    dmexcess_frac_model1 = arr[:, 5] / dmmean
    dmexcess_frac_model2 = arr[:, 1] / dmmean
    dmexcess_frac_model3 = arr[:, 2] / dmmean
    dmexcess_frac_model4 = arr[:, 3] / dmmean
    dmexcess_frac_model5 = arr[:, 4] / dmmean

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

    rm_host_frame = np.abs(frb_sources_dsa['rm'].values) *\
                 (np.abs(frb_sources_dsa['redshift']) + 1)**2
    ind_host_names = frb_sources_dsa[rm_host_frame > 750]['name'].values

    ind_halo_names = ['fen', 'nihari', 'elektra', 'mifanshan', 'jackie', 'gemechu']

    ind_hostdm = np.where(frb_sources_dsa['name'].isin(ind_host_names))[0]
    ind_halodm = np.where(frb_sources_dsa['name'].isin(ind_halo_names))[0]
    ind_neither = range(len(zdsa))

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.plot(zmod, dmexcess_frac_model1, color='k', linestyle='--', 
            linewidth=1., zorder=1, alpha=0.35)
    ax.plot(zmod, dmexcess_frac_model2, color='k', linestyle='--', 
            linewidth=1., zorder=2, alpha=0.35)
    ax.plot(zmod, dmexcess_frac_model3, color='k', linestyle='--', 
            linewidth=1., zorder=4, alpha=0.35)
    ax.plot(zmod, dmexcess_frac_model4, color='k', linestyle='--', 
            linewidth=1., zorder=5, alpha=0.35)
    ax.plot(zmod, dmexcess_frac_model5, color='k', linestyle='--', 
            linewidth=1., zorder=6, alpha=0.35)
    ax.plot(zmod, np.ones_like(zmod), color='red', linestyle='--', 
            linewidth=0.5, zorder=7, alpha=0.5)

    ax.scatter(zdsa[ind_neither], dmexcess_frac[ind_neither], zorder=11,
               marker=marker1, linewidth=1.5,
               s=msize, color=c1, edgecolors='k', label='DSA-110 FRBs', alpha=1)
    ax.scatter(zdsa[ind_hostdm], dmexcess_frac[ind_hostdm], zorder=12, c=c2,
               marker=marker1, linewidth=1.5,
               s=msize, edgecolors='k', label='Large Rotation Measure', alpha=1,)
    ax.scatter(zdsa[ind_halodm], dmexcess_frac[ind_halodm], zorder=13, alpha=1,
               marker=marker1, linewidth=1.5,
               s=msize, color=c3, edgecolors='k', 
               label='Massive group / Cluster')
    ax.set_xlabel('Redshift', fontsize=18)
    ax.set_ylabel(r'$\Delta_{\rm DM}$', fontsize=18)

    # for nameii in ind_halo_names:
    #     ind_nameii = np.where(frb_sources['name'] == nameii)[0]
    #     zii = np.abs(frb_sources['redshift'].values[ind_nameii])
    #     text = ax.text(zii, dmexcess_frac[ind_nameii], nameii, 
    #                    fontsize=12, color='k', zorder=10)

    fn_frb_nondsa = '/Users/liamconnor/work/projects/baryons/data/frbdata/nondsa_frbs_nov2023.csv'
    frb_sources_nondsa = read_frb_catalog(fn_frb_nondsa)

    match_dataframe, in_footprint = cross_match_all(frb_sources_nondsa,
                                                thresh_bperp_mpc=1.5,
                                                cluster_zmax=None,
                                                min_mass_legacy=13.5)
    
    frb_sources_footprint = frb_sources_nondsa.iloc[np.array(in_footprint)]

    zfrb = frb_sources_nondsa['redshift'].values
    dmfrb = frb_sources_nondsa['dm_exgal'].values

    zfrb_footprint = frb_sources_footprint['redshift'].values
    dmfrb_footprint = frb_sources_footprint['dm_exgal'].values

    arr = np.load('contours_zdm_plot.npy')
    zex = np.load('./zex_contourplot.npy')
    
    dmmean_mod = []
    [dmmean_mod.append(dmmean[np.argmin(np.abs(zex - z))]) for z in zfrb]
    dmexcess_frac = (dmfrb - dmhalo)/dmmean_mod

    dmmean_mod = []
    [dmmean_mod.append(dmmean[np.argmin(np.abs(zex - z))]) for z in zfrb_footprint]
    dmexcess_frac_footprint = (dmfrb_footprint - dmhalo)/dmmean_mod

    ind_host_names = ['FRB20190520B', 'FRB20121102A', 'FRB20201124A']
    ind_halo_names = ['FRB20190520B', 'FRB20220610A', 'FRB20211127I', 'FRB20200906A']
    ind_hostdm = np.where(frb_sources_nondsa['name'].isin(ind_host_names))[0]
    ind_halodm = np.where(frb_sources_nondsa['name'].isin(ind_halo_names))[0]
    ind_neither = range(len(zfrb))

#    fig = plt.figure(figsize=(8, 6))
#    ax = fig.add_subplot(111)
#    ax.plot(zmod, np.ones_like(zmod), color='k', linestyle='--', linewidth=1, zorder=1)
#    ax.plot(zmod, 1 + 5 * np.exp(-2*zmod), color='k', linestyle='--', linewidth=1, zorder=2)
#    ax.plot(zmod, 1 - 1.5 * np.exp(-2*zmod), color='k', linestyle='--', linewidth=1, zorder=3)
    ax.scatter(zfrb_footprint[zfrb_footprint > 0.025], 
               dmexcess_frac_footprint[zfrb_footprint>0.025], 
               zorder=8, marker=marker2, linewidth=1,
               s=msize*0.75, color=c1, edgecolors='grey', 
               label='non-DSA FRBs', alpha=1)
    ax.scatter(zfrb[ind_hostdm], dmexcess_frac[ind_hostdm], zorder=9, 
               linewidth=1, marker=marker2,
               s=msize*0.75, color=c2, edgecolors='grey', label='', alpha=1)
    ax.scatter(zfrb[ind_halodm], dmexcess_frac[ind_halodm], zorder=10, 
               linewidth=1, marker=marker2,
               s=msize*0.75, color=c3, edgecolors='grey', label='')
    ax.legend(loc=0)
    ax.grid('on', alpha=0.25, linestyle='--')
    ax.set_xlabel('Redshift', fontsize=18)
#    ax.set_ylabel('DM surpluss$)')
    ax.set_ylabel(r'$\Delta_{\rm DM}$', fontsize=18)

def plot_dmex(frb_sources):
    msize = 60
    c1, c2, c3 = 'lightblue', 'orange', 'red'
    fn_frb_dsa = '/Users/liamconnor/Desktop/dsa110_frbs_dec23.csv'
    marker1='o'
    marker2='s'

    frb_sources_dsa = read_frb_catalog(fn_frb_dsa)

    zdsa = frb_sources_dsa['redshift'].values
    dmdsa = frb_sources_dsa['dm_exgal'].values

    zmod = np.linspace(0, 1.5, 100)

    ind = np.where((zdsa != -1) & (dmdsa > 0) & (np.abs(zdsa) > 0.025))[0]
    frb_sources_dsa = frb_sources_dsa.iloc[ind]

    zdsa = np.abs(zdsa[ind])
    dmdsa = dmdsa[ind]
    dmhalo = 30.
    
    arr = np.load('contours_zdm_plot.npy')
    zex = np.load('./zex_contourplot.npy')

    for ii in range(6):
        f = UnivariateSpline(zex,arr[:, ii])
        arr[:, ii] = f(zex)

    dmmean = arr[:, 0]#(coeff * zdsa)
    
    dmmean_mod = []
    [dmmean_mod.append(dmmean[np.argmin(np.abs(zex - z))]) for z in zdsa]
    dmexcess_frac = (dmdsa - dmhalo)#/dmmean_mod

    zmod = zex
    dmexcess_frac_model1 = arr[:, 5] #/ dmmean
    dmexcess_frac_model2 = arr[:, 1] #/ dmmean
    dmexcess_frac_model3 = arr[:, 2] #/ dmmean
    dmexcess_frac_model4 = arr[:, 3] #/ dmmean
    dmexcess_frac_model5 = arr[:, 4] #/ dmmean

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

    rm_host_frame = np.abs(frb_sources_dsa['rm'].values) *\
                 (np.abs(frb_sources_dsa['redshift']) + 1)**2
    ind_host_names = frb_sources_dsa[rm_host_frame > 750]['name'].values

    ind_halo_names = ['fen', 'nihari', 'elektra', 'mifanshan', 'jackie', 'gemechu']

    ind_hostdm = np.where(frb_sources_dsa['name'].isin(ind_host_names))[0]
    ind_halodm = np.where(frb_sources_dsa['name'].isin(ind_halo_names))[0]
    ind_neither = range(len(zdsa))

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.plot(zmod, dmexcess_frac_model1, color='k', linestyle='--', 
            linewidth=1., zorder=1, alpha=0.35)
    ax.plot(zmod, dmexcess_frac_model2, color='k', linestyle='--', 
            linewidth=1., zorder=2, alpha=0.35)
    ax.plot(zmod, dmexcess_frac_model3, color='k', linestyle='--', 
            linewidth=1., zorder=4, alpha=0.35)
    ax.plot(zmod, dmexcess_frac_model4, color='k', linestyle='--', 
            linewidth=1., zorder=5, alpha=0.35)
    ax.plot(zmod, dmexcess_frac_model5, color='k', linestyle='--', 
            linewidth=1., zorder=6, alpha=0.35)
    ax.plot(zmod, dmmean, color='red', linestyle='--', 
            linewidth=0.5, zorder=7, alpha=0.5)

    ax.scatter(zdsa[ind_neither], dmexcess_frac[ind_neither], zorder=11,
               marker=marker1, linewidth=1.5,
               s=msize, color=c1, edgecolors='k', label='DSA-110 FRBs', alpha=1)
    ax.scatter(zdsa[ind_hostdm], dmexcess_frac[ind_hostdm], zorder=12, c=c2,
               marker=marker1, linewidth=1.5,
               s=msize, edgecolors='k', label='Large Rotation Measure', alpha=1,)
    ax.scatter(zdsa[ind_halodm], dmexcess_frac[ind_halodm], zorder=13, alpha=1,
               marker=marker1, linewidth=1.5,
               s=msize, color=c3, edgecolors='k', 
               label='Massive group / Cluster')

    fn_frb_nondsa = '/Users/liamconnor/work/projects/baryons/data/frbdata/nondsa_frbs_nov2023.csv'
    frb_sources_nondsa = read_frb_catalog(fn_frb_nondsa)

    match_dataframe, in_footprint = cross_match_all(frb_sources_nondsa,
                                                thresh_bperp_mpc=1.5,
                                                cluster_zmax=None,
                                                min_mass_legacy=13.5)
    
    frb_sources_footprint = frb_sources_nondsa.iloc[np.array(in_footprint)]

    zfrb = frb_sources_nondsa['redshift'].values
    dmfrb = frb_sources_nondsa['dm_exgal'].values

    zfrb_footprint = frb_sources_footprint['redshift'].values
    dmfrb_footprint = frb_sources_footprint['dm_exgal'].values

    arr = np.load('contours_zdm_plot.npy')
    zex = np.load('./zex_contourplot.npy')
    
    dmmean_mod = []
    [dmmean_mod.append(dmmean[np.argmin(np.abs(zex - z))]) for z in zfrb]
    dmexcess_frac = (dmfrb - dmhalo)#/dmmean_mod

    dmmean_mod = []
    [dmmean_mod.append(dmmean[np.argmin(np.abs(zex - z))]) for z in zfrb_footprint]
    dmexcess_frac_footprint = (dmfrb_footprint - dmhalo)#/dmmean_mod

    ind_host_names = ['FRB20190520B', 'FRB20121102A', 'FRB20201124A']
    ind_halo_names = ['FRB20190520B', 'FRB20220610A', 'FRB20211127I', 'FRB20200906A']
    ind_hostdm = np.where(frb_sources_nondsa['name'].isin(ind_host_names))[0]
    ind_halodm = np.where(frb_sources_nondsa['name'].isin(ind_halo_names))[0]
    ind_neither = range(len(zfrb))

#    fig = plt.figure(figsize=(8, 6))
#    ax = fig.add_subplot(111)
#    ax.plot(zmod, np.ones_like(zmod), color='k', linestyle='--', linewidth=1, zorder=1)
#    ax.plot(zmod, 1 + 5 * np.exp(-2*zmod), color='k', linestyle='--', linewidth=1, zorder=2)
#    ax.plot(zmod, 1 - 1.5 * np.exp(-2*zmod), color='k', linestyle='--', linewidth=1, zorder=3)
    ax.scatter(zfrb_footprint[zfrb_footprint > 0.025], 
               dmexcess_frac_footprint[zfrb_footprint>0.025], 
               zorder=8, marker=marker2, linewidth=1.5,
               s=msize*0.75, color=c1, edgecolors='k', 
               label='non-DSA FRBs', alpha=1)
    
    ax.scatter(zfrb[ind_hostdm], dmexcess_frac[ind_hostdm], zorder=9, 
               linewidth=1.5, marker=marker2,
               s=msize*0.75, color=c2, edgecolors='k', label='', alpha=1)
    
    ax.scatter(zfrb[ind_halodm], dmexcess_frac[ind_halodm], zorder=10, 
               linewidth=1.5, marker=marker2,
               s=msize*0.75, color=c3, edgecolors='k', label='')
    
    ax.legend(loc=4)
    ax.grid('on', alpha=0.25, linestyle='--')
    ax.set_xlabel('Redshift', fontsize=18)
    ax.set_ylabel(r'DM$_{\rm ex}$ (pc cm$^{-3}$)', fontsize=18)

def recreate_macquart20(plot_dsa=False):
    """ Recreate the Macquart relation plot 
    from the 2020 paper
    """

    if plot_dsa:
        fig = plt.figure(figsize=(8.5 * 1.65, 5.8 * 1.65))
    else:
        fig = plt.figure(figsize=(8.5, 5.8))

    ax = fig.add_subplot(111)

    msize = 35

    A_med = 1000
    A_low = 700
    A_up = 1600

    zline = np.linspace(0, 2, 1000)
    dmcos_line = A_med * zline
    ax.plot(zline, dmcos_line, color='k', linewidth=1, 
            zorder=2, label=r'DM$_{cosmic}$(z) Planck15')

    if plot_dsa:
        frb_sources_dsa = read_frb_catalog(fn_frb_dsa)
        zdsa = frb_sources_dsa['redshift'].values
        dmdsa = frb_sources_dsa['dm_exgal'].values
        ind = np.where((zdsa != -1) & (dmdsa > 0) & (np.abs(zdsa) > 0.0))[0]
        frb_sources_dsa = frb_sources_dsa.iloc[ind]
        zdsa = np.abs(zdsa[ind])
        dmdsa = dmdsa[ind]
        dmhalo = 30.
        ax.scatter(zdsa, dmdsa - dmhalo - 80 / (zdsa+1), zorder=3, 
                   marker='o', linewidth=1, s=60, color='lightblue', 
                   edgecolors='blue', alpha=1, label='DSA-110 FRBs')
        alpha=0.85
        msize = 35
        ax.set_xlim(0, 1.6)
        ax.set_ylim(0, 2285)
    else:
        ax.set_xlim(0, 0.7)
        ax.set_ylim(0, 1000)
        alpha=1.

    ax.fill_between(zline, A_low * zline, A_up * zline, color='Grey', 
                    alpha=0.3, zorder=1)
    ax.scatter(0.11778, 1600*0.11778, c='red', marker='s', s=msize, alpha=alpha, zorder=4)
    ax.scatter(0.29, 225, c='green', marker='s', s=msize, alpha=alpha, zorder=4)
    ax.scatter(0.32, 245, c='blue', marker='s', s=msize, alpha=alpha, zorder=4)
    ax.scatter(0.4755, 450, c='C1', marker='s', s=msize, alpha=alpha, zorder=4)
    ax.scatter(0.52172, 430, c='purple', marker='s', s=msize, alpha=alpha, zorder=4)

    ax.scatter(0.375, 180, c='grey', marker='s', s=msize, alpha=alpha)
    ax.scatter(0.195, 1400 * 0.195, c='grey', marker='*', s=msize, alpha=alpha)
    ax.scatter(0.66000, 950 * 0.66, c='grey', marker='o', s=msize, alpha=alpha)

    ax.set_xlabel(r'$z_{FRB}$', fontsize=24)
    ax.set_ylabel(r'DM$_{\rm cosmic}$ (pc cm$^{-3}$)', fontsize=24)

    if plot_dsa:
        ax.legend(loc=2, fontsize=20)
        plt.savefig('macquart20_dsa.pdf', bbox_inches='tight')
    else:
        ax.legend(loc=2, fontsize=16)
        plt.savefig('macquart20.pdf', bbox_inches='tight')



def plot_corner_gtc():
    import pygtc

    flat_samples1 = np.load('./flat_samples_11200steps_newcut.npy')
    flat_samples2 = np.load('./flat_samples_25600steps_DSAonly.npy')

    # List of parameter names, supports latex
    # NOTE: For capital greek letters in latex mode, use \mathsf{}
    names = [r'$f_{igm}$',r'$f_X$',r'$\mu$',r'$\sigma$']
    #names = ['1', '2', '3', '4']


    # Labels for the different chains
    chainLabels = ["All FRBs", "DSA-110 only"]

    # List of Gaussian curves to plot
    #(to represent priors): mean, width
    # Empty () or None if no prior to plot
    priors = ((),
            (),
            (),
            (),)

    # List of truth values, to mark best-fit or input values
    # NOT a python array because of different lengths
    # Here we choose two sets of truth values
    truths = ((4, .5, None, .1, 0, None, None, 0),
            (None, None, .3, 1, None, None, None, None))

    # Labels for the different truths
    truthLabels = ( 'the truth',
                'also true')

    # Do the 
    GTC = pygtc.plotGTC(chains=[flat_samples2[5000::3], flat_samples2[5000::3]],
                        paramNames=names,
                        chainLabels=chainLabels,
    #                    truths=truths,
    #                    truthLabels=truthLabels,
                        priors=priors,
                    customLabelFont={'size':18})

for i in range(4):
    mcmc = np.percentile(flat_samples2[1000::10, i], [16, 50, 84])
    q = np.diff(mcmc)
    txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
    if i==2:
        mu_h, mu_h_low, mu_h_up = mcmc[1], q[0], q[1]
        txt = txt.format(mcmc[1], q[0], q[1], labels[i])
    elif i==3:
        sig_h = mcmc[1]
        txt = txt.format(mcmc[1], q[0], q[1], labels[i])
    else:
        txt = txt.format(mcmc[1], q[0], q[1], labels[i])
    display(Math(txt))
    print(txt)