#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 15:46:25 2023

@author: mrutala
"""

import datetime as dt
import time as benchmark_time
import sys
import astropy.units as u
import matplotlib.pyplot as plt

import huxt_addons as H_ad
sys.path.append('/Users/mrutala/projects/HUXt-DIAS/code/')
import huxt as H
import huxt_analysis as HA
import huxt_inputs as Hin

def example():

    runstart = dt.datetime(2016, 5, 1)
    runend = dt.datetime(2016, 7, 1)
    simtime = (runend-runstart).days * u.day
    r_min = 329 * u.solRad  #  Average Mars orbital distance
    
    benchmark_starttime = benchmark_time.time()
    
    #   Generate inputs from VSWUM
    time, vcarr, bcarr = H_ad.generate_vCarr_from_VSWUM(runstart, runend, ref_r=r_min)
    
    plt.imshow(vcarr)
    
    #set up the model, with (optional) time-dependent bpol boundary conditions
    model = Hin.set_time_dependent_boundary(vcarr, time, runstart, simtime, 
                                            r_min=r_min, r_max=1290*u.solRad, dt_scale=50.0, latitude=0*u.deg,
                                            bgrid_Carr = bcarr, lon_start=0*u.deg, lon_stop=360*u.deg, frame='sidereal')
    
    
    model.solve([], streak_carr=False)
    
    benchmark_totaltime = benchmark_time.time() - benchmark_starttime
    print('Time elapsed in solving the model: {}'.format(benchmark_totaltime))
    
    #HA.plot(model, (0/4.)*simtime)
    HA.plot(model, (1/4.)*simtime)
    HA.plot(model, (2/4.)*simtime)
    HA.plot(model, (3/4.)*simtime)
    HA.plot(model, (4/4.)*simtime)
    
    
    # =============================================================================
    # 
    # =============================================================================
    r_min = 215 * u.solRad  #  Average Mars orbital distance
    benchmark_starttime = benchmark_time.time()
    #   Generate inputs from VSWUM
    time, vcarr, bcarr = Hin.generate_vCarr_from_OMNI(runstart, runend, ref_r=r_min)
    
    plt.imshow(vcarr)
    
    #set up the model, with (optional) time-dependent bpol boundary conditions
    model = Hin.set_time_dependent_boundary(vcarr, time, runstart, simtime, 
                                            r_min=r_min, r_max=1290*u.solRad, dt_scale=50.0, latitude=0*u.deg,
                                            bgrid_Carr = bcarr, lon_start=0*u.deg, lon_stop=360*u.deg, frame='sidereal')
    
    
    model.solve([], streak_carr=False)
    
    benchmark_totaltime = benchmark_time.time() - benchmark_starttime
    print('Time elapsed in solving the model: {}'.format(benchmark_totaltime))
    
    #HA.plot(model, (0/4.)*simtime)
    HA.plot(model, (1/4.)*simtime)
    HA.plot(model, (2/4.)*simtime)
    HA.plot(model, (3/4.)*simtime)
    HA.plot(model, (4/4.)*simtime)
    
    return

def example_Gauss():

    runstart = dt.datetime(2016, 5, 1)
    runend = dt.datetime(2016, 7, 1)
    simtime = (runend-runstart).days * u.day
    r_min = 329 * u.solRad  #  Average Mars orbital distance
    
    benchmark_starttime = benchmark_time.time()
    
    #   Generate inputs from VSWUM
    
    # csv = pd.read_csv()
    # 
    
    time, vcarr, bcarr = H_ad.generate_vCarr_from_CSV(csv, runstart, runend, ref_r=r_min)
    
    plt.imshow(vcarr)
    
    #set up the model, with (optional) time-dependent bpol boundary conditions
    model = Hin.set_time_dependent_boundary(vcarr, time, runstart, simtime, 
                                            r_min=r_min, r_max=1290*u.solRad, dt_scale=50.0, latitude=0*u.deg,
                                            bgrid_Carr = bcarr, lon_start=0*u.deg, lon_stop=360*u.deg, frame='sidereal')
    
    
    model.solve([], streak_carr=False)
    
    benchmark_totaltime = benchmark_time.time() - benchmark_starttime
    print('Time elapsed in solving the model: {}'.format(benchmark_totaltime))
    
    #HA.plot(model, (0/4.)*simtime)
    HA.plot(model, (1/4.)*simtime)
    HA.plot(model, (2/4.)*simtime)
    HA.plot(model, (3/4.)*simtime)
    HA.plot(model, (4/4.)*simtime)