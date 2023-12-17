import matplotlib.pyplot as plt

fn_frb_dsa='/Users/liamconnor/Desktop/dsafrbsnov23.csv'
fn_frb_nondsa='/Users/liamconnor/work/projects/baryons/data/nondsa_frbs_nov2023.csv'

frb_sources_dsa = read_frb_catalog(fn_frb_dsa)
frb_sources_nondsa = read_frb_catalog(fn_frb_nondsa)

def plot_dmexcess_halos_dsa(frb_sources):

    zdsa = frb_sources['redshift'].values
    dmdsa = frb_sources['dm_exgal'].values

    zmod = np.linspace(0, 1.5, 100)

    ind = np.where((zdsa != -1) & (dmdsa > 0) & (np.abs(zdsa) > 0.01))[0]
    zdsa = np.abs(zdsa[ind])
    dmdsa = dmdsa[ind]
    coeff = 850.
    dmhost = 150.
    dmhalo = 20.
    dmmean = (coeff * zdsa)
    dmexcess_frac = (dmdsa - dmhalo - dmhost * (zdsa + 1)**-1)/dmmean

    zmod = zex
    dmexcess_frac_model1 = (arr[:, 0] - dmhalo - dmhost * (zmod + 1)**-1)/(coeff * zex)
    dmexcess_frac_model2 = (arr[:, 1] - dmhalo - dmhost * (zmod + 1)**-1)/(coeff * zex)
    dmexcess_frac_model3 = (arr[:, 2] - dmhalo - dmhost * (zmod + 1)**-1)/(coeff * zex)
    dmexcess_frac_model4 = (arr[:, 3] - dmhalo - dmhost * (zmod + 1)**-1)/(coeff * zex)
    dmexcess_frac_model5 = (arr[:, 4] - dmhalo - dmhost * (zmod + 1)**-1)/(coeff * zex)

    frb_sources = frb_sources.iloc[ind]

    ind_host_names = ['ansel', 'etienne', 'ada', 
                      'ayo', 'leonidas', 'ferb', 'phineas', 
                      'fatima', 'martha', 'alex']
    
    ind_halo_names = ['fen', 'nihari', 'elektra', 'mifanshan', 'jackie']

    ind_hostdm = np.where(frb_sources['name'].isin(ind_host_names))[0]
    ind_halodm = np.where(frb_sources['name'].isin(ind_halo_names))[0]
    ind_neither = range(len(zdsa))

#    c1, c2, c3 = 'lavender', 'palegoldenrod', 'lightsalmon' # lemonchiffon
    c1, c2, c3 = 'white', 'gold', 'lightsalmon' # lemonchiffon

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.plot(zmod, dmexcess_frac_model1, color='k', linestyle='--', linewidth=2, zorder=1)
    ax.plot(zmod, dmexcess_frac_model2, color='k', linestyle='--', linewidth=3, zorder=2)
    ax.plot(zmod, dmexcess_frac_model3, color='k', linestyle='--', linewidth=3, zorder=4)
    ax.plot(zmod, dmexcess_frac_model4, color='k', linestyle='--', linewidth=3, zorder=5)
    ax.plot(zmod, dmexcess_frac_model5, color='k', linestyle='--', linewidth=3, zorder=6)

    ax.scatter(zdsa[ind_neither], dmexcess_frac[ind_neither], zorder=7,
               s=100, color=c1, edgecolors='k', label='DSA-110 FRBs', alpha=1)
    ax.scatter(zdsa[ind_hostdm], dmexcess_frac[ind_hostdm], zorder=8,
               s=100, color=c2, edgecolors='k', label='Large host RM', alpha=1)
    ax.scatter(zdsa[ind_halodm], dmexcess_frac[ind_halodm], zorder=9,
               s=100, color=c3, edgecolors='k', label='Massive group / Cluster')
    ax.legend()
    ax.grid('on', alpha=0.5, linestyle='--')
    ax.set_xlabel('Redshift', fontsize=18)
    ax.set_ylabel(r'$\Delta_{\rm DM}$', fontsize=18)

def plot_dmexcess_halos_nondsa(frb_sources):

    zdsa = frb_sources['redshift'].values
    dmdsa = frb_sources['dm_exgal'].values

    zmod = np.linspace(0, 1.5, 100)

    ind = np.where((zdsa != -1) & (dmdsa > 0) & (np.abs(zdsa) > 0.01))[0]
    zdsa = np.abs(zdsa[ind])
    dmdsa = dmdsa[ind]
    coeff = 850.
    dmhost = 125.
    dmhalo = 20.
    dmmean = (coeff * zdsa)
    dmexcess_frac = (dmdsa - dmhalo - dmhost * (zdsa + 1)**-1)/dmmean
    frb_sources = frb_sources.iloc[ind]

    ind_host_names = ['FRB20190520B', 'FRB20121102A', 'FRB20201124A']
    ind_halo_names = ['FRB20190520B', 'FRB20220610A', 'FRB20211127I', 'FRB20200906A']
    ind_hostdm = np.where(frb_sources['name'].isin(ind_host_names))[0]
    ind_halodm = np.where(frb_sources['name'].isin(ind_halo_names))[0]
    ind_neither = range(len(zdsa))

    c1, c2, c3 = 'white', 'lightblue', 'lightsalmon' # lemonchiffon

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.plot(zmod, np.ones_like(zmod), color='k', linestyle='--', linewidth=2, zorder=1)
    ax.plot(zmod, 1 + 5 * np.exp(-2*zmod), color='k', linestyle='--', linewidth=3, zorder=2)
    ax.plot(zmod, 1 - 1.5 * np.exp(-2*zmod), color='k', linestyle='--', linewidth=3, zorder=3)
    ax.scatter(zdsa[ind_neither], dmexcess_frac[:], zorder=4,
               s=100, color=c1, edgecolors='k', label='DSA-110 FRBs', alpha=1)
    ax.scatter(zdsa[ind_hostdm], dmexcess_frac[ind_hostdm], zorder=5,
               s=100, color=c2, edgecolors='k', label='Large host RM', alpha=1)
    ax.scatter(zdsa[ind_halodm], dmexcess_frac[ind_halodm], zorder=6,
               s=100, color=c3, edgecolors='k', label='Massive group / Cluster')
    ax.legend()
    ax.grid('on', alpha=0.5, linestyle='--')
    ax.set_xlabel('Redshift', fontsize=18)
    ax.set_ylabel(r'$\Delta_{\rm DM}$', fontsize=18)
