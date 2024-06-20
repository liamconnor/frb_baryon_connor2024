# Code to prepare data from a mock FRB survey 
# in IllustrisTNG TNG300-1 (Walker et al. (2023))
# Liam Connor

import numpy as np
import glob

snap_to_z = {'99':0,
             '91':0.1,
             '84':0.2,
             '78':0.3,
             '72':0.4,
             '67':0.5,
             '59':0.7,
             '50':1.0,
             '40':1.5,
             '33':2.0,
             '25':3.0,
             '21':4.0,
             '17':5.0,}

REDSHIFTS = np.array(list(snap_to_z.values()))
gas_types = ['Halos','Filaments','Voids','Total']

def file_list():
    simdir = '/Users/liamconnor/Downloads/for_Liam_Connor/'
    fl = glob.glob(simdir + 'Sim_TNG300-1_Snap_*.npy')
    fl.sort()
    fl = fl[::-1]

    return fl

def read_simdata():
    data_full = []

    for ii,fn in enumerate(fl[:]):
        print(fn, REDSHIFTS[ii])
        data = np.load(fn, allow_pickle=True).item()
        data_full.append(data)

    return data_full 

def read_sightlines_allz(data_full, gas_type):
    """ Make a Macquart relation for a gas type, 
    continuous in redshift.
    """

    DM=[]
    DMnew=0
    zarr = np.zeros([5125, 13])

    if gas_type=='Total':
        gas_type=''

    for ii,fn in enumerate(fl[:-1]):
        snap_no1 = fn.split('_')[-4]
        z1 = snap_to_z[snap_no1]

        snap_no2 = fl[ii+1].split('_')[-4]
        z2 = snap_to_z[snap_no2]

        if gas_type in ('igm', 'IGM'):
            dDM_z = np.array(data_full[ii]['dDMdzFilament_Pakmor']) +\
                    np.array(data_full[ii]['dDMdzVoid_Pakmor'])
            dDM_zp1 = np.array(data_full[ii+1]['dDMdzFilament_Pakmor']) +\
                    np.array(data_full[ii+1]['dDMdzVoid_Pakmor'])
        else:
            dDM_z = np.array(data_full[ii][f'dDMdz{gas_type}_Pakmor'])
            dDM_zp1 = np.array(data_full[ii+1][f'dDMdz{gas_type}_Pakmor'])

        delta_z = (np.array([z2 - z1])).repeat(len(dDM_z))
        frac_z = np.random.uniform(0,1,len(dDM_z))
        print(len(delta_z))

        DMnew = DMnew + (frac_z * dDM_z + (1-frac_z)*dDM_zp1) * delta_z * frac_z
        DM.append(DMnew)
        zarr[:, ii+1] = zarr[:, ii] + delta_z * frac_z
    
    redshift, dm = zarr[:, 1:].T.flatten(), array(DM).flatten()

    return redshift, dm

def read_sightlines_allz(data_full):
    """ Make a Macquart relation for a gas type, 
    continuous in redshift.
    """

    DM=[]
    DMnewHalo=0
    DMnewIGM=0
    DMnewTotal=0
    nz_per_snap = 5125
    zarr = np.zeros([nz_per_snap, 13])

    gas_types = ['Halo','IGM','']

    DM = np.zeros([12 * nz_per_snap, 3])

    for ii,fn in enumerate(fl[:-1]):
        snap_no1 = fn.split('_')[-4]
        z1 = snap_to_z[snap_no1]

        dDM_z_len = len(np.array(data_full[ii][f'dDMdz_Pakmor']))
        snap_no2 = fl[ii+1].split('_')[-4]
        z2 = snap_to_z[snap_no2]
        delta_z = (np.array([z2 - z1])).repeat(dDM_z_len)
        frac_z = np.random.uniform(0,1,len(delta_z))
        zarr[:, ii+1] = zarr[:, ii] + delta_z * frac_z

        for jj, gas_type in enumerate(gas_types[:]):
            if gas_type in ('igm', 'IGM'):
                dDM_z = np.array(data_full[ii]['dDMdzFilament_Pakmor']) +\
                        np.array(data_full[ii]['dDMdzVoid_Pakmor'])
                dDM_zp1 = np.array(data_full[ii+1]['dDMdzFilament_Pakmor']) +\
                        np.array(data_full[ii+1]['dDMdzVoid_Pakmor'])
                DMnewIGM = DMnewIGM + (frac_z * dDM_z + (1-frac_z)*dDM_zp1) * delta_z * frac_z
                DM[nz_per_snap*ii:nz_per_snap*(ii+1), 1] = DMnewIGM
            elif gas_type=='Halo':
                dDM_z = np.array(data_full[ii][f'dDMdz{gas_type}_Pakmor'])
                dDM_zp1 = np.array(data_full[ii+1][f'dDMdz{gas_type}_Pakmor'])
                DMnewHalo = DMnewHalo + (frac_z * dDM_z + (1-frac_z)*dDM_zp1) * delta_z * frac_z
                DM[nz_per_snap*ii:nz_per_snap*(ii+1), 0] = DMnewHalo
            else:
                dDM_z = np.array(data_full[ii][f'dDMdz{gas_type}_Pakmor'])
                dDM_zp1 = np.array(data_full[ii+1][f'dDMdz{gas_type}_Pakmor'])
                DMnewTotal = DMnewTotal + (frac_z * dDM_z + (1-frac_z)*dDM_zp1) * delta_z * frac_z
                DM[nz_per_snap*ii:nz_per_snap*(ii+1), 2] = DMnewTotal
            #DM.append(DMnew)
    
    redshift = zarr[:, 1:].T.flatten()
#    dm = zarr[:, 1:].T.flatten(), array(DM).flatten()

    return redshift, dm

def read_sightlines_snaponly(data_full, gas_type):

    DM=[]
    DMnew=0
    zarr = np.zeros([5125, 13])

    if gas_type=='Total':
        gas_type=''

    for ii,fn in enumerate(fl[:-1]):
        snap_no1 = fn.split('_')[-4]
        z1 = snap_to_z[snap_no1]

        snap_no2 = fl[ii+1].split('_')[-4]
        z2 = snap_to_z[snap_no2]

        if gas_type in ('igm', 'IGM'):
            dDM_z = np.array(data_full[ii]['dDMdzFilament_Pakmor']) +\
                    np.array(data_full[ii]['dDMdzVoid_Pakmor'])
            dDM_zp1 = np.array(data_full[ii+1]['dDMdzFilament_Pakmor']) +\
                    np.array(data_full[ii+1]['dDMdzVoid_Pakmor'])
        else:
            dDM_z = np.array(data_full[ii][f'dDMdz{gas_type}_Pakmor'])
            dDM_zp1 = np.array(data_full[ii+1][f'dDMdz{gas_type}_Pakmor'])

        delta_z = (np.array([z2 - z1])).repeat(len(dDM_z))

        DMnew = DMnew + 0.5 * (dDM_z + dDM_zp1) * delta_z
        DM.append(DMnew)
        zarr[:, ii+1] = zarr[:, ii] + delta_z 
    
    redshift, dm = zarr[:, 1:].T.flatten(), array(DM).flatten()

    return redshift, dm

def dmz_by_gastype():
    """ Create and save arrays of redshift and 
    DM with shape (nsightlines, 3). The three 
    columns are Halos, IGM, and Total. 
    """
    redshift_arr = []
    dmarr = []

    data_full = read_simdata()

    for gas_type in ['Halo', 'IGM', 'Total']:
        redshift, dm = read_sightlines_snaponly(data_full, gas_type)
        #redshift, dm = read_sightlines_allz(data_full, gas_type)
        redshift_arr.append(redshift)
        dmarr.append(dm)

    dmarr = np.concatenate(dmarr).reshape(3, -1).T
    redshift_arr = np.concatenate(redshift_arr).reshape(3, -1).T

    np.save('RedshiftsHaloFilTotal.npy', redshift_arr)  
    np.save('DMsHaloFilTotal.npy', dmarr)

    return redshift_arr, dmarr


def fit(dms, zarr, z, offset=0.0):
    dmhalo = dms[zarr==z, 0]
    dmigm = dms[zarr==z, 1]

    halo_boost = 1#np.random.normal(1.4, 0.01, len(dmhalo))**2
    dmhalo = dmhalo * halo_boost

    dmigm = dmigm * (0.80 - (0.13 * np.mean(halo_boost) - 0.13))

    print(np.mean(dmigm) / np.mean(dmhalo))

    x, y = np.log(dmhalo), np.log(dmigm)
    x = x #+ np.log((0.137+offset)/0.137)
    y = y #+ np.log((0.80+offset)/0.80)

    ind = np.where(np.isfinite(x) & np.isfinite(y))[0]
    x, y = x[ind], y[ind]
    # Combine x and y into a single dataset
    data = np.vstack((x, y)).T

    # Fit a bivariate normal distribution
    mean = np.mean(data, axis=0)
    covariance = np.cov(data, rowvar=False)

    # Create a grid of points for plotting the distribution
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    x_grid, y_grid = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
#    pos = np.dstack((x_grid, y_grid))

    # Create the multivariate normal distribution
    rv = multivariate_normal(mean, covariance)

    # Extract the elements of the covariance matrix
    sigma_xx = covariance[0, 0]
    sigma_yy = covariance[1, 0]
    sigma_xy = covariance[0, 1]

    # Calculate the correlation coefficient
    rho = sigma_xy / np.sqrt(sigma_xx * sigma_yy)


    return mean, covariance, rho, rv, x_grid, y_grid

data_full = read_simdata()
z, dm = read_sightlines_snaponly(data_full, 'Total')
z, dm = dmz_by_gastype()
z = z[:, 0]

parr=[]
zlist = list(set(z))
zlist.sort()

for zz in zlist:
    means, covs, rho = fit(dm, z, zz, offset=0.10)[:3]
    print(zz, np.exp(means + 0.5 * np.diag(covs)))
    params = [1, means[0], means[1], covs[0,0]**0.5, covs[1,1]**0.5, rho]
    parr.append(params)

parr = np.concatenate(parr).reshape(-1, 6)
np.save('tng_params_new_morehalo.npy', parr)

