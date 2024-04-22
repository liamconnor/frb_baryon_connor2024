import numpy as np 
import pandas as pd

def read_frb_catalog(fnfrb, zmin=0.0, zmax=np.inf, 
                     telescope='all', secure_host=True,
                     max_fractional_MWDM=np.inf, exclude_names=None):
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

    if exclude_names is not None:
        if type(exclude_names) != list:
            exclude_names = [exclude_names]
        for name in exclude_names:
            indexNames = frb_catalog[ frb_catalog['name'] == name].index
            frb_catalog.drop(indexNames, inplace=True)

    return frb_catalog

def print_to_latex():
    print("This is a function that prints to latex")

    data = read_frb_catalog('frb_catalog.csv')

    include_cols = ['name','dm_exgal','redshift','survey','ne2001']
    data['dm_exgal'].values = np.round(data['dm_exgal'].values, 1)

    print(data[include_cols].to_latex(caption='This is the caption'))