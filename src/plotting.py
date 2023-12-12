import matplotlib.pyplot as plt


def plot_dmexcess_halos(frb_sources):
    fn_frb='/Users/liamconnor/Desktop/dsafrbsnov23.csv'
    frb_sources = read_frb_catalog(fn_frb)

    zdsa = frb_sources['redshift'].values
    dmdsa = frb_sources['dm_exgal'].values

    zmod = np.linspace(0, 1.5, 100)

    ind = np.where((zdsa != -1) & (dmdsa > 0) & (np.abs(zdsa) > 0.01))[0]
    zdsa = np.abs(zdsa[ind])
    dmdsa = dmdsa[ind]
    coeff = 850.
    dmhost = 150.
    dmhalo = 30.
    dmmean = (coeff * zdsa)
    dmexcess_frac = (dmdsa - dmhalo - dmhost * (zdsa + 1)**-1)/dmmean
    frb_sources = frb_sources.iloc[ind]

    ind_host_names = ['zach', 'ansel', 'etienne', 'ada', 'ayo', 'leonidas']
    ind_halo_names = ['elektra', 'jackie', 'mifanshan']
    ind_hostdm = np.where(frb_sources['name'].isin(ind_host_names))[0]
    ind_halodm = np.where(frb_sources['name'].isin(ind_halo_names))[0]
    ind_neither = range(len(zdsa))

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.plot(zmod, np.ones_like(zmod), color='k', linestyle='--', linewidth=2, zorder=1)
    ax.plot(zmod, 1 + 5 * np.exp(-2*zmod), color='k', linestyle='--', linewidth=3, zorder=2)
    ax.plot(zmod, 1 - 1.5 * np.exp(-2*zmod), color='k', linestyle='--', linewidth=3, zorder=3)
    ax.scatter(zdsa[ind_neither], dmexcess_frac[ind_neither], zorder=4,
               s=100, color='lavender', edgecolors='k', label='DSA-110 FRBs', alpha=1)
    ax.scatter(zdsa[ind_hostdm], dmexcess_frac[ind_hostdm], zorder=5,
               s=100, color='lemonchiffon', edgecolors='k', label='Large host RM', alpha=1)
    ax.scatter(zdsa[ind_halodm], dmexcess_frac[ind_halodm], zorder=6,
               s=100, color='lightsalmon', edgecolors='k', label='Massive halo DM')
    ax.legend()
    ax.grid('on', alpha=0.5, linestyle='--')
    ax.set_xlabel('Redshift', fontsize=18)
    ax.set_ylabel(r'$\Delta_{\rm DM}$', fontsize=18)

