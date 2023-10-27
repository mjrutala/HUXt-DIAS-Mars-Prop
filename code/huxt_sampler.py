#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 12:16:48 2023

@author: mrutala
"""

import datetime as dt
import time as benchmark_time
import sys
import astropy.units as u
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np

import huxt_addons as H_ad
sys.path.append('/Users/mrutala/projects/HUXt-DIAS/code/')
import huxt as H
import huxt_analysis as HA
import huxt_inputs as Hin

def vSWUM_Sampler():
    
    #  Inputs-to-be
    filepath = '/Users/mrutala/projects/HUXt-DIAS-Mars-Prop/data/Hour2014-2020.csv'
    runstart = dt.datetime(2016, 6, 15)
    runend = dt.datetime(2016, 6, 20)
    simtime = (runend-runstart).days * u.day
    r_min = 329 * u.solRad  #  Average Mars orbital distance-- change dynamically with Mars location

    #   Read in the full VSWUM 1 hr data from file
    cols_to_use = ['date_[utc]', 'mu_b_x_SW', 'sigma_b_x_SW', 'mu_v_mag_SW', 'sigma_v_mag_SW']
    data = pd.read_csv(filepath)
    data = data[cols_to_use]
    
    # create a datetime column and set it as index
    data['datetime'] = pd.to_datetime(data['date_[utc]'], format='%Y-%m-%d %H:%M:%S')
    #data.index = data['datetime']
    
    #   MJR 20231023: Shouldn't be needed if there's no bad data in VSWUM,,,
    # # Set invalid data points to NaN
    #id_bad = data['V'] == 9999.0
    #data.loc[id_bad, 'V'] = np.NaN
    
    # =============================================================================
    #   Sampler
    # =============================================================================
    # sub_data = data.copy()
    # sub_data.rename(columns={'mu_v_mag_SW':'V', 'mu_b_x_SW':'Bx'}, inplace=True)
    
    seed = 19950612  #  For reproducability
    generator = np.random.default_rng(seed)
    n = 500
    
    vcarr_list = []
    for i in range(n):
        sub_data = data.copy()
        sub_data['V'] = generator.normal(sub_data['mu_v_mag_SW'], sub_data['sigma_v_mag_SW'])
        sub_data['Bx'] = generator.normal(sub_data['mu_b_x_SW'], sub_data['sigma_b_x_SW'])
        time, vcarr, bcarr = H_ad.generate_vCarr_from_df(sub_data, runstart, runend, ref_r=r_min, origin='Mars')
        
        vcarr_list.append(vcarr[:,0])
        if i % 10 == 0: print(i)
        
    sub_data['V'] = sub_data['mu_v_mag_SW']
    sub_data['Bx'] = sub_data['mu_b_x_SW']
    time0, vcarr0, bcarr0 = H_ad.generate_vCarr_from_df(sub_data, runstart, runend, ref_r=r_min, origin='Mars')
    
    time1, vcarr1, bcarr1 = Hin.generate_vCarr_from_OMNI(runstart, runend)
    
    fig, ax = plt.subplots(figsize=(8,6))
    for row in vcarr_list:
        ax.plot(np.arange(0, 128)*(360/128.), row, alpha = 1/100., color='blue')
        
    ax.plot(np.arange(0, 128)*(360/128.), vcarr0[:,0], color='red')
    ax.plot(np.arange(0, 128)*(360/128.), vcarr1[:,0], color='gold')
    
    ax.set_xticks([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330])
        
    
    plt.show()
    
    return vcarr_list
   

   