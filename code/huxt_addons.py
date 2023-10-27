#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 15:08:37 2023

@author: mrutala
"""
import datetime
import os
import urllib
import ssl

import astropy.units as u
from astropy.io import fits
from astropy.time import Time, TimeDelta
import httplib2
import numpy as np
import pandas as pd
from pyhdf.SD import SD, SDC
import h5py
from scipy.io import netcdf
from scipy import interpolate
from sunpy.coordinates import sun
from sunpy.net import Fido
from sunpy.net import attrs
from sunpy.timeseries import TimeSeries
import requests

import sys
sys.path.append('/Users/mrutala/projects/HUXt-DIAS/code/')
import huxt as H

def generate_vCarr_from_VSWUM(runstart, runend, nlon_grid=128, dt=1*u.day, 
                             ref_r = 329*u.solRad, corot_type = 'both'):
    """
    A function to download OMNI data and generate V_carr and time_grid
    for use with set_time_dependent_boundary

    Args:
        runstart: Start time as a datetime
        runend: End time as a datetime
        nlon_grid: Int, 128 by default
        dt: time resolution, in days is 1*u.day.
        ref_r: radial distance to produce v at, 215*u.solRad by default.
        corot_type: STring that determines corot type (both, back, forward)
    Returns:
        Time: Array of times as Julian dates
        Vcarr: Array of solar wind speeds mapped as a function of Carr long and time
        bcarr: Array of Br mapped as a function of Carr long and time
    """
    
    #check the coro_type is one of he accepted values
    assert corot_type == 'both' or corot_type == 'back' or corot_type == 'forward'

    #   MJR 20231023: I think this is (partially) to allow easier mapping to Carr. longitude later on, so I'm leaving it
    starttime = runstart - datetime.timedelta(days=28)
    endtime = runend + datetime.timedelta(days=28)
    
    #   Read in the VSWUM 1 hr data from file
    filepath = '/Users/mrutala/projects/HUXt-DIAS-Mars-Prop/data/Hour2014-2020.csv'
    cols_to_use = ['date_[utc]', 'mu_b_x_SW', 'sigma_b_x_SW', 'mu_v_mag_SW', 'sigma_v_mag_SW']
    
    vswum = pd.read_csv(filepath)
    data = vswum[cols_to_use].copy()
    data.rename(columns={'mu_v_mag_SW':'V', 'mu_b_x_SW':'Bx'}, inplace=True)
    
    #   MJR 20231023: Shouldn't be needed if there's no bad data in VSWUM,,,
    # # Set invalid data points to NaN
    #id_bad = data['V'] == 9999.0
    #data.loc[id_bad, 'V'] = np.NaN
    
    # create a datetime column
    data['datetime'] = pd.to_datetime(data['date_[utc]'], format='%Y-%m-%d %H:%M:%S')

    #   !!!! Do we need to adjust this for Mars synodic period? i.e., longer day?
    # compute the synodic rotation period
    daysec = 24 * 60 * 60 * u.s
    # synodic_period = 27.2753 * daysec  # Solar Synodic rotation period from Earth.
    synodic_period = 26.32 * daysec  # Solar Synodic rotation period from Mars
    omega_synodic = 2*np.pi * u.rad / synodic_period

    # find the period of interest
    mask = ((data['datetime'] > starttime) &
            (data['datetime'] < endtime))
    mask_data = data[mask]
    mask_data = mask_data.reset_index()
    mask_data['Time'] = Time(mask_data['datetime'])  #  Add an astropy Time
    
    #   Start and final MJD
    smjd = mask_data['Time'][0].mjd
    fmjd = mask_data['Time'][len(mask_data) - 1].mjd

    #   MJR 20231023: Input data is complete, and shouldn't need interpolation
    #   But I'm leaving it in case we want to interpolate through error-prone
    #   windows in the future
    #   Maybe with a "bad_data_value" optional input?
    # interpolate through OMNI V data gaps
    # # Set invalid data points to NaN
    # id_bad = data['V'] == 9999.0
    # data.loc[id_bad, 'V'] = np.NaN
    # omni_int = omni.interpolate(method='linear', axis=0).ffill().bfill()
    # del omni
    
    mask_data_int = mask_data
    del mask_data
    
    # compute carrington longitudes
    # cr = np.ones(len(mask_data_int))
    # cr_lon_init = np.ones(len(mask_data_int))*u.rad
    # for i in range(0, len(omni_int)):
    #     cr[i], cr_lon_init[i] = datetime2huxtinputs(omni_int['datetime'][i])

    import spiceypy as spice
    spice.furnsh('/Users/mrutala/projects/HUXt-DIAS-Mars-Prop/data/SPICE/generic/metakernel_HUXt_planetary.txt')
    
    #   Only keep the surface coordinates from spice.subpnt()
    #   NB This does *not* give CR #, but we shouldn't need it here
    ets = spice.datetime2et(mask_data_int['datetime'])
    xyz_coords = [spice.subpnt('NEAR POINT/ELLIPSOID', 'SUN', et, 'IAU_SUN', 'LT+S', 'MARS BARYCENTER')[0] for et in ets]
    rlonlat_coords = np.array([spice.reclat(xyz) for xyz in xyz_coords])
    cr_lon_init = rlonlat_coords[:,1]*u.rad
    
    mask_data_int['Carr_lon'] = cr_lon_init.value
    mask_data_int['Carr_lon_unwrap'] = np.unwrap(cr_lon_init.value)

    mask_data_int['mjd'] = [t.mjd for t in mask_data_int['Time'].array]
    
    # # get the Earth radial distance info.
    # dirs = H._setup_dirs_()
    # ephem = h5py.File(dirs['ephemeris'], 'r')
    # # convert ephemeric to mjd and interpolate to required times
    # all_time = Time(ephem['EARTH']['HEEQ']['time'], format='jd').value - 2400000.5
    # omni_int['R'] = np.interp(omni_int['mjd'], 
    #                           all_time, ephem['EARTH']['HEEQ']['radius'][:]) *u.km
    
    #   Get the Mars radial distance info.
    #   We just want the radius, so the reference frame shouldn't matter
    xyz_coords, _ = spice.spkpos('MARS BARYCENTER', ets, 'SUN_EARTH_CEQU', 'LT+S', 'SUN')
    mask_data_int['R'] = np.sqrt(np.sum(xyz_coords**2, axis=1)) * u.km
    
    spice.kclear()
    
    #map each point back/forward to the reference radial distance
    mask_data_int['mjd_ref'] = mask_data_int['mjd']
    mask_data_int['Carr_lon_ref'] = mask_data_int['Carr_lon_unwrap']
    #for t in range(0, len(mask_data_int)):
    for t, row in mask_data_int.iterrows():
        #time lag to reference radius
        delta_r = (ref_r.to(u.km) - row['R']*u.km).value  #  MJR: Don't know why *u.km needs to be added...
        delta_t = delta_r/row['V']/24/60/60
        mask_data_int.at[t, 'mjd_ref'] = row['mjd_ref'] + delta_t
        #change in Carr long of the measurement
        
        #   MJR 20231023: !!!! Need to think about this more carefully
        #   Reference Carrington Longitude is already calculated taking light travel time into account
        #   Here, we're taking plasma travel time into account (effectively) (I think)
        mask_data_int.at[t, 'Carr_lon_ref'] = row['Carr_lon_ref'] - \
            delta_t *daysec * 2 * np.pi /synodic_period
    
    #  Fix R to non-astropy, unitless values for compatibility
    mask_data_int['R'] = mask_data_int['R'].to_numpy('float64')
    
    #sort the omni data by Carr_lon_ref for interpolation
    # Why?
    mask_data_temp = mask_data_int.copy()
    mask_data_temp = mask_data_temp.sort_values(by = ['Carr_lon_ref'])
    
    # now remap these speeds back on to the original time steps
    mask_data_int['V_ref'] = np.interp(mask_data_int['Carr_lon_unwrap'], mask_data_temp['Carr_lon_ref'],
                                  mask_data_temp['V'])
    mask_data_int['Br_ref'] = np.interp(mask_data_int['Carr_lon_unwrap'], mask_data_temp['Carr_lon_ref'],
                                  -mask_data_temp['Bx'])

    # compute the longitudinal and time grids
    dphi_grid = 360/nlon_grid
    lon_grid = np.arange(dphi_grid/2, 360.1-dphi_grid/2, dphi_grid) * np.pi/180 * u.rad
    dt = dt.to(u.day).value
    time_grid = np.arange(smjd, fmjd + dt/2, dt)

    vgrid_carr_recon_back = np.ones((nlon_grid, len(time_grid))) * np.nan
    vgrid_carr_recon_forward = np.ones((nlon_grid, len(time_grid))) * np.nan
    vgrid_carr_recon_both = np.ones((nlon_grid, len(time_grid))) * np.nan
    
    bgrid_carr_recon_back = np.ones((nlon_grid, len(time_grid))) * np.nan
    bgrid_carr_recon_forward = np.ones((nlon_grid, len(time_grid))) * np.nan
    bgrid_carr_recon_both = np.ones((nlon_grid, len(time_grid))) * np.nan

    for t in range(0, len(time_grid)):
        # find nearest time and current Carrington longitude
        t_id = np.argmin(np.abs(mask_data_int['mjd'] - time_grid[t]))
        Elong = mask_data_int['Carr_lon'][t_id] * u.rad
        
        # get the Carrington longitude difference from current Earth pos
        dlong_back = H._zerototwopi_(lon_grid.value - Elong.value) * u.rad
        dlong_forward = H._zerototwopi_(Elong.value - lon_grid.value) * u.rad
        
        dt_back = (dlong_back / omega_synodic).to(u.day)
        dt_forward = (dlong_forward / omega_synodic).to(u.day)
        
        vgrid_carr_recon_back[:, t] = np.interp(time_grid[t] - dt_back.value, mask_data_int['mjd'], mask_data_int['V_ref'],
                                                left=np.nan, right=np.nan)
        bgrid_carr_recon_back[:, t] = np.interp(time_grid[t] - dt_back.value, mask_data_int['mjd'], mask_data_int['Br_ref'],
                                                left=np.nan, right=np.nan)
        # if ((tCime_grid[t] >= Time(runstart).mjd) & (time_grid[t] <= Time(runend).mjd)):
        #     import matplotlib.pyplot as plt
        #     plt.plot(time_grid[t] - dt_back.value)
        #     plt.plot(mask_data_int['mjd'])
        #     return time_grid[t] - dt_back.value, mask_data_int['mjd'], mask_data_int['V_ref'], vgrid_carr_recon_back[:, t]      

        vgrid_carr_recon_forward[:, t] = np.interp(time_grid[t] + dt_forward.value, mask_data_int['mjd'], mask_data_int['V_ref'],
                                                   left=np.nan, right=np.nan)
        bgrid_carr_recon_forward[:, t] = np.interp(time_grid[t] + dt_forward.value, mask_data_int['mjd'], mask_data_int['Br_ref'],
                                                   left=np.nan, right=np.nan)

        numerator = (dt_forward * vgrid_carr_recon_back[:, t] + dt_back * vgrid_carr_recon_forward[:, t])
        denominator = dt_forward + dt_back
        vgrid_carr_recon_both[:, t] = numerator / denominator
        
        numerator = (dt_forward * bgrid_carr_recon_back[:, t] + dt_back * bgrid_carr_recon_forward[:, t])
        bgrid_carr_recon_both[:, t] = numerator / denominator
    # cut out the requested time
    mask = ((time_grid >= Time(runstart).mjd) & (time_grid <= Time(runend).mjd))
    
    
    if corot_type == 'both':
        return time_grid[mask], vgrid_carr_recon_both[:, mask], bgrid_carr_recon_both[:, mask]
    elif corot_type == 'back':
        return time_grid[mask], vgrid_carr_recon_back[:, mask], bgrid_carr_recon_back[:, mask]
    elif corot_type == 'forward':
        return time_grid[mask], vgrid_carr_recon_forward[:, mask], bgrid_carr_recon_forward[:, mask]
    
    
def generate_vCarr_from_df(df, runstart, runend, nlon_grid=128, dt=1*u.day, 
                           ref_r = 329*u.solRad, corot_type = 'both', 
                           origin="Earth"):
    """
    A function to generate V_carr and time_grid from a CSV of date, u_r, B_x, 
    for use with set_time_dependent_boundary

    Args:
        runstart: Start time as a datetime
        runend: End time as a datetime
        nlon_grid: Int, 128 by default
        dt: time resolution, in days is 1*u.day.
        ref_r: radial distance to produce v at, 215*u.solRad by default.
        corot_type: String that determines corot type (both, back, forward)
    Returns:
        Time: Array of times as Julian dates
        Vcarr: Array of solar wind speeds mapped as a function of Carr long and time
        bcarr: Array of Br mapped as a function of Carr long and time
    """
    
    #check the coro_type is one of he accepted values
    assert corot_type == 'both' or corot_type == 'back' or corot_type == 'forward'

    #   MJR 20231023: I think this is (partially) to allow easier mapping to Carr. longitude later on, so I'm leaving it
    starttime = runstart - datetime.timedelta(days=28)
    endtime = runend + datetime.timedelta(days=28)
    if (starttime < df['datetime'].iloc[0]) or (endtime > df['datetime'].iloc[-1]):
        print("Input DataFrame does not have sufficient padding before runstart or after runend."+
              "28 (Earth) days of padding currently required.")
        return
    
    # #   Read in the VSWUM 1 hr data from file
    # filepath = '/Users/mrutala/projects/HUXt-DIAS-Mars-Prop/data/Hour2014-2020.csv'
    # cols_to_use = ['date_[utc]', 'mu_b_x_SW', 'sigma_b_x_SW', 'mu_v_mag_SW', 'sigma_v_mag_SW']
    
    # vswum = pd.read_csv(filepath)
    # data = vswum[cols_to_use].copy()
    # data.rename(columns={'mu_v_mag_SW':'V', 'mu_b_x_SW':'Bx'}, inplace=True)
    
    # #   MJR 20231023: Shouldn't be needed if there's no bad data in VSWUM,,,
    # # # Set invalid data points to NaN
    # #id_bad = data['V'] == 9999.0
    # #data.loc[id_bad, 'V'] = np.NaN
    
    # # create a datetime column
    # data['datetime'] = pd.to_datetime(data['date_[utc]'], format='%Y-%m-%d %H:%M:%S')

    # compute the synodic rotation period
    #  Should be 27.2753 at Earth and 26.3536 at Mars
    daysec = 24 * 60 * 60 * u.s
    sidereal_years = {'Mercury':     87.97,
                      'Earth':      365.25,
                      'Mars':       686.98}
    sidereal_solar_period = 25.38
    synodic_solar_period = sidereal_years[origin] * sidereal_solar_period / (sidereal_years[origin] - sidereal_solar_period) * daysec
    omega_synodic = 2*np.pi * u.rad / (synodic_solar_period)

    # find the period of interest
    mask = ((df['datetime'] > starttime) &
            (df['datetime'] < endtime))
    data = df[mask]
    data = data.reset_index()
    data['Time'] = Time(data['datetime'])  #  Add an astropy Time
    
    #   Start and final MJD
    smjd = data['Time'][0].mjd
    fmjd = data['Time'][len(data) - 1].mjd

    #   MJR 20231023: Input data is complete, and shouldn't need interpolation
    #   But I'm leaving it in case we want to interpolate through error-prone
    #   windows in the future
    #   Maybe with a "bad_data_value" optional input?
    # interpolate through OMNI V data gaps
    # # Set invalid data points to NaN
    # id_bad = data['V'] == 9999.0
    # data.loc[id_bad, 'V'] = np.NaN
    # omni_int = omni.interpolate(method='linear', axis=0).ffill().bfill()
    # del omni
    
        
    # =============================================================================
    #   Computer Carrington Longitudes (of the origin)
    # =============================================================================
    # compute carrington longitudes
    # cr = np.ones(len(mask_data_int))
    # cr_lon_init = np.ones(len(mask_data_int))*u.rad
    # for i in range(0, len(omni_int)):
    #     cr[i], cr_lon_init[i] = datetime2huxtinputs(omni_int['datetime'][i])
    # # get the Earth radial distance info.
    # dirs = H._setup_dirs_()
    # ephem = h5py.File(dirs['ephemeris'], 'r')
    # # convert ephemeric to mjd and interpolate to required times
    # all_time = Time(ephem['EARTH']['HEEQ']['time'], format='jd').value - 2400000.5
    # omni_int['R'] = np.interp(omni_int['mjd'], 
    #                           all_time, ephem['EARTH']['HEEQ']['radius'][:]) *u.km

    import spiceypy as spice
    dirs = H._setup_dirs_()
    spice.furnsh(dirs['spice'] + '/metakernel_HUXt_planetary.txt')
    
    #   Only keep the surface coordinates from spice.subpnt()
    #   NB This does *not* give CR #, but we shouldn't need it here
    ets = spice.datetime2et(data['datetime'])
    xyz_coords = [spice.subpnt('NEAR POINT/ELLIPSOID', 'SUN', et, 'IAU_SUN', 'LT+S', 'MARS BARYCENTER')[0] for et in ets]
    rlonlat_coords = np.array([spice.reclat(xyz) for xyz in xyz_coords])
    cr_lon_init = rlonlat_coords[:,1]*u.rad
    
    data['Carr_lon'] = cr_lon_init.value
    data['Carr_lon_unwrap'] = np.unwrap(cr_lon_init.value)

    data['mjd'] = [t.mjd for t in data['Time'].array]
    
    #   Get the origin radial distance info.
    #   We just want the radius, so the reference frame shouldn't matter
    xyz_coords, _ = spice.spkpos(origin + ' BARYCENTER', ets, 'SUN_EARTH_CEQU', 'LT+S', 'SUN')
    data['R'] = np.sqrt(np.sum(xyz_coords**2, axis=1)) * u.km
    
    spice.kclear()
    
    # =============================================================================
    #   map each point back/forward to the reference radial distance
    # =============================================================================
    data['mjd_ref'] = data['mjd']
    data['Carr_lon_ref'] = data['Carr_lon_unwrap']
    #for t in range(0, len(mask_data_int)):
    for t, row in data.iterrows():
        #time lag to reference radius
        delta_r = (ref_r.to(u.km) - row['R']*u.km).value  #  MJR: Don't know why *u.km needs to be added...
        delta_t = delta_r/row['V']/24/60/60
        data.at[t, 'mjd_ref'] = row['mjd_ref'] + delta_t
        #change in Carr long of the measurement
        
        #   MJR 20231023: !!!! Need to think about this more carefully
        #   Reference Carrington Longitude is already calculated taking light travel time into account
        #   Here, we're taking plasma travel time into account (effectively) (I think)
        data.at[t, 'Carr_lon_ref'] = row['Carr_lon_ref'] - \
            delta_t * daysec * 2 * np.pi / synodic_solar_period
    
    #  Fix R to non-astropy, unitless values for compatibility
    data['R'] = data['R'].to_numpy('float64')
    
    #sort the omni data by Carr_lon_ref for interpolation
    # Why?
    data_temp = data.copy()
    data_temp = data_temp.sort_values(by = ['Carr_lon_ref'])
    
    # now remap these speeds back on to the original time steps
    data['V_ref'] = np.interp(data['Carr_lon_unwrap'], data_temp['Carr_lon_ref'],
                                  data_temp['V'])
    data['Br_ref'] = np.interp(data['Carr_lon_unwrap'], data_temp['Carr_lon_ref'],
                                  -data_temp['Bx'])

    # compute the longitudinal and time grids
    dphi_grid = 360/nlon_grid
    lon_grid = np.arange(dphi_grid/2, 360.1-dphi_grid/2, dphi_grid) * np.pi/180 * u.rad
    dt = dt.to(u.day).value
    time_grid = np.arange(smjd, fmjd + dt/2, dt)

    vgrid_carr_recon_back = np.ones((nlon_grid, len(time_grid))) * np.nan
    vgrid_carr_recon_forward = np.ones((nlon_grid, len(time_grid))) * np.nan
    vgrid_carr_recon_both = np.ones((nlon_grid, len(time_grid))) * np.nan
    
    bgrid_carr_recon_back = np.ones((nlon_grid, len(time_grid))) * np.nan
    bgrid_carr_recon_forward = np.ones((nlon_grid, len(time_grid))) * np.nan
    bgrid_carr_recon_both = np.ones((nlon_grid, len(time_grid))) * np.nan

    for t in range(0, len(time_grid)):
        # find nearest time and current Carrington longitude
        t_id = np.argmin(np.abs(data['mjd'] - time_grid[t]))
        Elong = data['Carr_lon'][t_id] * u.rad
        
        # get the Carrington longitude difference from current Earth pos
        dlong_back = H._zerototwopi_(lon_grid.value - Elong.value) * u.rad
        dlong_forward = H._zerototwopi_(Elong.value - lon_grid.value) * u.rad
        
        dt_back = (dlong_back / omega_synodic).to(u.day)
        dt_forward = (dlong_forward / omega_synodic).to(u.day)
        
        vgrid_carr_recon_back[:, t] = np.interp(time_grid[t] - dt_back.value, data['mjd'], data['V_ref'],
                                                left=np.nan, right=np.nan)
        bgrid_carr_recon_back[:, t] = np.interp(time_grid[t] - dt_back.value, data['mjd'], data['Br_ref'],
                                                left=np.nan, right=np.nan)
        
        vgrid_carr_recon_forward[:, t] = np.interp(time_grid[t] + dt_forward.value, data['mjd'], data['V_ref'],
                                                   left=np.nan, right=np.nan)
        bgrid_carr_recon_forward[:, t] = np.interp(time_grid[t] + dt_forward.value, data['mjd'], data['Br_ref'],
                                                   left=np.nan, right=np.nan)

        numerator = (dt_forward * vgrid_carr_recon_back[:, t] + dt_back * vgrid_carr_recon_forward[:, t])
        denominator = dt_forward + dt_back
        vgrid_carr_recon_both[:, t] = numerator / denominator
        
        numerator = (dt_forward * bgrid_carr_recon_back[:, t] + dt_back * bgrid_carr_recon_forward[:, t])
        bgrid_carr_recon_both[:, t] = numerator / denominator
    # cut out the requested time
    mask = ((time_grid >= Time(runstart).mjd) & (time_grid <= Time(runend).mjd))
    
    
    if corot_type == 'both':
        return time_grid[mask], vgrid_carr_recon_both[:, mask], bgrid_carr_recon_both[:, mask]
    elif corot_type == 'back':
        return time_grid[mask], vgrid_carr_recon_back[:, mask], bgrid_carr_recon_back[:, mask]
    elif corot_type == 'forward':
        return time_grid[mask], vgrid_carr_recon_forward[:, mask], bgrid_carr_recon_forward[:, mask]