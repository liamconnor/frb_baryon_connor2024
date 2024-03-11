import numpy as np 
import pandas as pd

def read_frb_catalog(fnfrb, zmin=0.0, zmax=np.inf, 
                     telescope='all', secure_host=True,
                     max_fractional_MWDM=np.inf):
    # Read the FRB catalog from a csv file
    frb_catalog = pd.read_csv(fnfrb, delim_whitespace=False)
    zfrb = frb_catalog['redshift'].values
    redshift_type = frb_catalog['redshift_type'].values

    ind = np.where((redshift_type != 'none') & (np.abs(zfrb) > zmin) &\
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

    return frb_catalog