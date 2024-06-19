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

        DMnew = DMnew + (frac_z * dDM_z + (1-frac_z)*dDM_zp1) * delta_z * frac_z
        DM.append(DMnew)
        zarr[:, ii+1] = zarr[:, ii] + delta_z * frac_z
    
    redshift, dm = zarr[:, 1:].T.flatten(), array(DM).flatten()

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
        redshift_arr.append(redshift)
        dmarr.append(dm)

    dmarr = np.concatenate(dmarr).reshape(3, -1).T
    redshift_arr = np.concatenate(redshift_arr).reshape(3, -1).T

    np.save('RedshiftsHaloFilTotal.npy', redshift_arr)  
    np.save('DMsHaloFilTotal.npy', dmarr)

    return redshift_arr, dmarr