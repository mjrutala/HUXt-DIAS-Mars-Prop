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
    r_min = 329 * u.solRad  #  Average Mars orbital distance-- 
    #  !!!! change above dynamically with Mars location

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
    seed = 19950612  #  For reproducability
    generator = np.random.default_rng(seed)
    n = 1000
    
    vcarr_list = []
    for i in range(n):
        sub_data = data.copy()
        sub_data['V'] = generator.normal(sub_data['mu_v_mag_SW'], sub_data['sigma_v_mag_SW'])
        sub_data['Bx'] = generator.normal(sub_data['mu_b_x_SW'], sub_data['sigma_b_x_SW'])
        time, vcarr, bcarr = H_ad.generate_vCarr_from_df(sub_data, runstart, runend, ref_r=r_min, origin='Mars')
        
        vcarr_list.append(vcarr[:,0])
        if i % 10 == 0: print(i)
    
    #   Get nominal vSWUM output
    sub_data = data.copy()    
    sub_data['V'] = sub_data['mu_v_mag_SW']
    sub_data['Bx'] = sub_data['mu_b_x_SW']
    time0, vcarr0, bcarr0 = H_ad.generate_vCarr_from_df(sub_data, runstart, runend, ref_r=r_min, origin='Mars')
    
    time1, vcarr1, bcarr1 = Hin.generate_vCarr_from_OMNI(runstart, runend)
    
    def fig1():
        fig, axs = plt.subplots(ncols=2, figsize=(8,6))
        plt.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.90, wspace=0.2)
        
        lon = np.arange(0, 128)*(360/128.)
        axs[0].plot(lon, vcarr1[:,0], color='gold', label='OMNI @ Earth')
        axs[0].plot(lon, vcarr0[:,0], color='xkcd:navy blue', label='Nominal vSWUM @ Mars')
        axs[0].fill_between(lon, 
                            np.percentile(np.array(vcarr_list), 10, axis=0), 
                            np.percentile(np.array(vcarr_list), 90, axis=0),
                            color='xkcd:navy blue', alpha=0.5, label=r'10$^{th}$/90$^{th}$ percentile')
        axs[0].legend(loc='best', bbox_to_anchor=[0.5, 0.5, 0.5, 0.5], fontsize=8)
        axs[0].annotate('(a)', (0, 1), (1, -1), xycoords='axes fraction', 
                        textcoords='offset points', ha='left', va='top')
        axs[0].set(xlabel='Carrington Longitude [deg.]', xticks=[0, 90, 180, 270, 360],
                   ylabel=r'Solar Wind Flow Speed ($U_{mag}$) [km/s]')
                   
        sigmas = np.std(np.array(vcarr_list), axis=1)
        axs[1].hist(sigmas, bins=np.arange(55, 70+0.05, 0.05), 
                    color='xkcd:navy blue', alpha = 0.5, label='All Samples')
        axs[1].axvline(np.std(vcarr[:,0]), 
                       color='xkcd:navy blue', label='Nominal vSWUM @ Mars')
        
        axs[1].legend(loc='best', bbox_to_anchor=[0.5, 0.5, 0.5, 0.5], fontsize=8)
        axs[1].annotate('(b)', (0, 1), (1, -1), xycoords='axes fraction', 
                        textcoords='offset points', ha='left', va='top')
        axs[1].set(xlabel='$\sigma_{U_{mag}}$ [km/s]', 
                   ylabel='# of Samples')
        
        fig.suptitle(r'$n = {}$'.format(n))
        
        plt.show()
    fig1()    
    
    
    return vcarr_list
   

   