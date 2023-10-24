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

    # compute the syndoic rotation period
    daysec = 24 * 60 * 60 * u.s
    synodic_period = 27.2753 * daysec  # Solar Synodic rotation period from Earth.
    omega_synodic = 2*np.pi * u.rad / synodic_period

    # find the period of interest
    mask = ((data['datetime'] > starttime) &
            (data['datetime'] < endtime))
    omni = data[mask]
    omni = omni.reset_index()
    omni['Time'] = Time(omni['datetime'])
    
    smjd = omni['Time'][0].mjd
    fmjd = omni['Time'][len(omni) - 1].mjd

    # interpolate through OMNI V data gaps
    omni_int = omni.interpolate(method='linear', axis=0).ffill().bfill()
    del omni
    
    # compute carrington longitudes
    cr = np.ones(len(omni_int))
    cr_lon_init = np.ones(len(omni_int))*u.rad
    for i in range(0, len(omni_int)):
        cr[i], cr_lon_init[i] = datetime2huxtinputs(omni_int['datetime'][i])

    omni_int['Carr_lon'] = cr_lon_init.value
    omni_int['Carr_lon_unwrap'] = np.unwrap(omni_int['Carr_lon'].to_numpy())

    omni_int['mjd'] = [t.mjd for t in omni_int['Time'].array]
    
    # get the Earth radial distance info.
    dirs = H._setup_dirs_()
    ephem = h5py.File(dirs['ephemeris'], 'r')
    # convert ephemeric to mjd and interpolate to required times
    all_time = Time(ephem['EARTH']['HEEQ']['time'], format='jd').value - 2400000.5
    omni_int['R'] = np.interp(omni_int['mjd'], 
                              all_time, ephem['EARTH']['HEEQ']['radius'][:]) *u.km
    
    #map each point back/forward to the reference radial distance
    omni_int['mjd_ref'] = omni_int['mjd']
    omni_int['Carr_lon_ref'] = omni_int['Carr_lon_unwrap']
    for t in range(0, len(omni_int)):
        #time lag to reference radius
        delta_r = (ref_r.to(u.km) - omni_int['R'][t]).value
        delta_t = delta_r/omni_int['V'][t]/24/60/60
        omni_int['mjd_ref'][t] = omni_int['mjd_ref'][t]  + delta_t
        #change in Carr long of the measurement
        omni_int['Carr_lon_ref'][t] =  omni_int['Carr_lon_ref'][t] - \
            delta_t *daysec * 2 * np.pi /synodic_period
    
    #  Fix R to non-astropy, unitless values for compatibility
    omni_int['R'] = omni_int['R'].to_numpy('float64')
    
    #sort the omni data by Carr_lon_ref for interpolation
    omni_temp = omni_int.copy()
    omni_temp = omni_temp.sort_values(by = ['Carr_lon_ref'])
    
    # now remap these speeds back on to the original time steps
    omni_int['V_ref'] = np.interp(omni_int['Carr_lon_unwrap'], omni_temp['Carr_lon_ref'],
                                  omni_temp['V'])
    omni_int['Br_ref'] = np.interp(omni_int['Carr_lon_unwrap'], omni_temp['Carr_lon_ref'],
                                  -omni_temp['BX_GSE'])

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
        t_id = np.argmin(np.abs(omni_int['mjd'] - time_grid[t]))
        Elong = omni_int['Carr_lon'][t_id] * u.rad
        
        # get the Carrington longitude difference from current Earth pos
        dlong_back = _zerototwopi_(lon_grid.value - Elong.value) * u.rad
        dlong_forward = _zerototwopi_(Elong.value - lon_grid.value) * u.rad
        
        dt_back = (dlong_back / omega_synodic).to(u.day)
        dt_forward = (dlong_forward / omega_synodic).to(u.day)
        
        vgrid_carr_recon_back[:, t] = np.interp(time_grid[t] - dt_back.value, omni_int['mjd'], omni_int['V_ref'],
                                                left=np.nan, right=np.nan)
        bgrid_carr_recon_back[:, t] = np.interp(time_grid[t] - dt_back.value, omni_int['mjd'], omni_int['Br_ref'],
                                                left=np.nan, right=np.nan)
        
        # if ((time_grid[t] >= Time(runstart).mjd) & (time_grid[t] <= Time(runend).mjd)):
        #     import matplotlib.pyplot as plt
        #     plt.plot(time_grid[t] - dt_back.value)
        #     plt.plot(omni_int['mjd'])
        #     return time_grid[t] - dt_back.value, omni_int['mjd'], omni_int['V_ref'], vgrid_carr_recon_back[:, t]

        vgrid_carr_recon_forward[:, t] = np.interp(time_grid[t] + dt_forward.value, omni_int['mjd'], omni_int['V_ref'],
                                                   left=np.nan, right=np.nan)
        bgrid_carr_recon_forward[:, t] = np.interp(time_grid[t] + dt_forward.value, omni_int['mjd'], omni_int['Br_ref'],
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