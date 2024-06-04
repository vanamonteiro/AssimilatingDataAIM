import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import matplotlib.dates as mdates
import glob
from datetime import datetime, timedelta
import matplotlib.ticker as mticker
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

"""
This is a python script with functions used for the
"Assimilating Morning, Evening, and Nighttime Greenhouse Gas Observations in Atmospheric Inversions"
Monteiro, et al., 2024

This code was written by Vanessa Monteiro.

Last update 04 June 2024


"""


# Read sites from a formatted file

def read_trace_gas(path_and_filename, header,year,gas):

    df = pd.read_csv(path_and_filename, comment = "#", names=header)
    df = df.rename({'Date': 'datetime_utc'}, axis='columns')    
    df.index = pd.to_datetime(df['datetime_utc']); del df['datetime_utc']; del df['time']
    df.loc[df[gas] == -9999] = np.nan #replace -9999 by NAN
    df = df.loc[(df.index.year == year)]
        
        
    return df

# Read weather files, original wind speed in mph. It returns a df with
#wind speed in ms, hourly averaged

def read_weather(path,year):
    
    df = pd.read_csv(path)
    df = df.rename({'valid': 'datetime_utc'}, axis='columns')
    df.index = pd.to_datetime(df['datetime_utc'], format = '%Y-%m-%d %H:%M')
    df = df.tz_localize(tz = 'UTC')
       
    df = pd.DataFrame(df.resample('H').mean())# resample so the mean is the full hour
    df['WS_OBS'] = df['sped']/2.237  #(wind speed in mph -- convert to m/s) 

    del df['sped']; del df['drct']

    df = df.loc[(df.index.year == year)]

    return df



# Read model outputs (datime utc)

def read_model_outputs(path_and_filename, header,year):
    
    df = pd.read_csv(path_and_filename, skiprows = 1,names=header)
    df = df.rename({'Date': 'datetime_utc'}, axis='columns')    
    try:
        df.index = pd.to_datetime(df['datetime_utc'], format = '%Y-%m-%d %H:%M:%S'); del df['datetime_utc']
    except:
        df.index = pd.to_datetime(df['datetime_utc'])
        
    df = df.resample('60min').mean()  
    df = df.tz_localize(tz = 'UTC')
    df = df.loc[(df.index.year == year)]
    
    
    return df


## categorize variables


def wind_category(df,var):
    
    df[var+'_CAT'] = np.nan
    
    df[var+'_CAT'].loc[(df[var] <2)] = 0 #'<2'
    df[var+'_CAT'].loc[(df[var] >=2)&(df[var] <3)] = 1# '2-3'
    df[var+'_CAT'].loc[(df[var] >=3)&(df[var] <4)] = 2#'3-4'
    df[var+'_CAT'].loc[(df[var] >=4)&(df[var] <5)] = 3#'4-5'
    df[var+'_CAT'].loc[(df[var] >=5)&(df[var] <6)] = 4#'5-6'
    df[var+'_CAT'].loc[(df[var] >=6)] = 5#'>=6'
    
    
    return np.array(df[var+'_CAT'])


def abl_category(df,var):

    df[var+'_CAT'] = np.nan

    df[var+'_CAT'].loc[(df[var] <50)] = 0 
    df[var+'_CAT'].loc[(df[var] >=50)&(df[var] <200)] = 1
    df[var+'_CAT'].loc[(df[var] >=200)&(df[var] <350)] = 2
    df[var+'_CAT'].loc[(df[var] >=350)&(df[var] <500)] = 3
    df[var+'_CAT'].loc[(df[var] >=500)] = 4

    return np.array(df[var+'_CAT'])



def tke_category(df,var):

    df[var+'_CAT'] = np.nan

    df[var+'_CAT'].loc[(df[var] <1)] = 0 
    df[var+'_CAT'].loc[(df[var] >=1)&(df[var] <1.2)] = 1
    df[var+'_CAT'].loc[(df[var] >=1.2)&(df[var] <1.4)] = 2
    df[var+'_CAT'].loc[(df[var] >=1.4)&(df[var] <1.6)] = 3
    df[var+'_CAT'].loc[(df[var] >=1.6)] = 4

    return np.array(df[var+'_CAT'])

#### add periods of day in local time

#- 5 - 9  am (night_subset == previous nighttime)
#- 0 - 11 am (night == all night)
#- 12 pm - 23 (day == all day)
#- 17 - 21 (day_subset == afternoon)


def period_cat(df):
    
    df['PERIOD'] = np.nan
    df['PERIOD'].loc[((df.index.hour >= 10)&(df.index.hour<=13))] = '5-8 AM LT' 
    df['PERIOD'].loc[((df.index.hour >= 14)&(df.index.hour<=16))] = '9-11 AM LT'       
    df['PERIOD'].loc[((df.index.hour >= 17)&(df.index.hour<=21))] = '12-4 PM LT'
    df['PERIOD'].loc[((df.index.hour >= 22)|(df.index.hour==0)|(df.index.hour==1))] = '5-8 PM LT' 
    df['PERIOD'].loc[((df.index.hour >= 2)&(df.index.hour<=4))] = '9-11 PM LT' 
    df['PERIOD'].loc[((df.index.hour >= 5)&(df.index.hour<=9))] = '0-4 AM LT' 
    
    
    return np.array(df['PERIOD'])

### add seasons (dormant vs growing)

def season_cat(df):

    df['SEASON'] = np.nan
    df['SEASON'].loc[((df.index.month == 1) | (df.index.month == 2))] = 'DORMANT' ##|(df.index.month == 11) | (df.index.month == 12))
    df['SEASON'].loc[((df.index.month == 5) | (df.index.month == 6) | 
                                      (df.index.month == 7) | (df.index.month == 8))] = 'GROWING'

    pd.options.mode.chained_assignment = None
    
    return np.array(df['SEASON'])



## errors // bias
def errors(df, column_name_model, column_name_obs):
    df['R_Error(%)'] = np.nan
    df['R_Error(%)'] = 100*((df[column_name_model]-df[column_name_obs])/df[column_name_obs])   
    
    df['ABS_Error'] = np.nan
    df['ABS_Error'] = abs(df[column_name_model]-df[column_name_obs])
    
    df['Error'] = np.nan
    df['Error'] = (df[column_name_model]-df[column_name_obs])
    
    return np.array(df['R_Error(%)']), np.array(df['ABS_Error']), np.array(df['Error'])


## Attribute emissions to periods of the day
def emissions(df):
    df['EMISSIONS'] = np.nan
    df['EMISSIONS'].loc[(df['SEASON'] == 'DORMANT') & (df['PERIOD'] == '0-4 AM LT')] = 0.48
    df['EMISSIONS'].loc[(df['SEASON'] == 'DORMANT') & (df['PERIOD'] == '5-8 AM LT')] = 0.72
    df['EMISSIONS'].loc[(df['SEASON'] == 'DORMANT') & (df['PERIOD'] == '9-11 AM LT')] = 0.87
    df['EMISSIONS'].loc[(df['SEASON'] == 'DORMANT') & (df['PERIOD'] == '12-4 PM LT')] = 0.82
    df['EMISSIONS'].loc[(df['SEASON'] == 'DORMANT') & (df['PERIOD'] == '5-8 PM LT')] = 0.96
    df['EMISSIONS'].loc[(df['SEASON'] == 'DORMANT') & (df['PERIOD'] == '9-11 PM LT')] = 0.67

    df['EMISSIONS'].loc[(df['SEASON'] == 'GROWING') & (df['PERIOD'] == '0-4 AM LT')] = 0.31
    df['EMISSIONS'].loc[(df['SEASON'] == 'GROWING') & (df['PERIOD'] == '5-8 AM LT')] = 0.63
    df['EMISSIONS'].loc[(df['SEASON'] == 'GROWING') & (df['PERIOD'] == '9-11 AM LT')] = 0.68
    df['EMISSIONS'].loc[(df['SEASON'] == 'GROWING') & (df['PERIOD'] == '12-4 PM LT')] = 0.81
    df['EMISSIONS'].loc[(df['SEASON'] == 'GROWING') & (df['PERIOD'] == '5-8 PM LT')] = 0.72
    df['EMISSIONS'].loc[(df['SEASON'] == 'GROWING') & (df['PERIOD'] == '9-11 PM LT')] = 0.46

    return np.array(df['EMISSIONS'])

#--------------------------------------------------------------------------------------------------------------------------

### plots (figures)

# vertical gradient normalized by time and categorized by wind speed

def fig_vg_norm_time_ws(df, periods_list,gas):
    fig, ax = plt.subplots(figsize=(35, 20))
    bw = 1/7.5 #barwidth is the distance from the center divided by the # of sites
    bw_n = 0
    i = 0
    n = 2*len(periods_list) #len(seasons)
    x = ['','<2','2-3','3-4','4-5','5-6','>6','']

    my_colors = ['#377eb8','#4daf4a','#ff7f00','#f7f7f7','#ffff33','#984ea3']

    for period in periods_list: 

        if period == '12-4 PM LT':
            hatch_period = 'X'
        if period != '12-4 PM LT':
            hatch_period = None

        ax.bar(df['VG'][period].index+bw_n-0.3, df['VG'][period], 
                yerr=0, ecolor='black',width = bw, 
                error_kw=dict(lw=8, capsize=(100*bw), capthick=3), edgecolor='black', linewidth=5, 
                alpha = 1, color = my_colors[i], hatch = hatch_period, 
                label = period ) 


        ax.set_xlabel('Wind Speed (m/s)', fontsize=95)
        if gas == 'co2':
            ax.set_ylabel(r'$\widetilde{VG}[$CO$_2$]$_{[time]}$', fontsize=95)

        if gas == 'ch4':
            ax.set_ylabel(r'$\widetilde{VG}[$CH$_4$]$_{[time]}$', fontsize=95)

        ax.set_ylim(0,32)

        # fixing yticks with matplotlib.ticker "FixedLocator"
        ticks_loc = ax.get_xticks().tolist()
        ax.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
        ax.set_xticklabels(x)
        ax.tick_params(axis = 'both',labelsize = 95)    
        ax.grid(axis = 'y',alpha = 1)
        bw_n = bw_n+bw
        i += 1

    ax.axhline(y = 2.5, color = 'r', linestyle = '-.', linewidth = 4, label = '2.5x afternoon')
    ax.axhline(y = 1, color = 'k', linestyle = '-.', linewidth = 4, label = '1.0x afternoon')
    ax.legend(loc = 'upper right',fontsize =45, title = 'Local time (LT)', title_fontsize = 45)
    return ax




# vertical gradient normalized by time and categorized by turbulent kinetic energy
def fig_vg_time_tke(df, periods_list,gas):
    fig, ax = plt.subplots(figsize=(35, 20))
    bw = 1/11 #barwidth is the distance from the center divided by the # of sites
    bw_n = 0
    i = 0
    n = 2*len(periods_list) #len(seasons)
    x = ['','<1.0','','1.0-1.2','','1.2-1.4','','1.4-1.6','','>1.6','']

    my_colors = ['#377eb8','#4daf4a','#ff7f00','#f7f7f7','#ffff33','#984ea3']

    for period in periods_list: 

        if period == '12-4 PM LT':
            hatch_period = 'X'
        if period != '12-4 PM LT':
            hatch_period = None

        ax.bar(df['VG'][period].index+bw_n-0.3, df['VG'][period], 
                yerr=0, ecolor='black',width = bw, 
                error_kw=dict(lw=8, capsize=(100*bw), capthick=3), edgecolor='black', linewidth=5, 
                alpha = 1, color = my_colors[i], hatch = hatch_period, 
                label = period ) 


        ax.set_xlabel('TKE (m$^2$/s$^2$)', fontsize=95)
        if gas == 'co2':
            ax.set_ylabel(r'$\widetilde{VG}[$CO$_2$]$_{[time]}$', fontsize=95)

        if gas == 'ch4':
            ax.set_ylabel(r'$\widetilde{VG}[$CH$_4$]$_{[time]}$', fontsize=95)

        ax.set_ylim(0,15)


        # fixing yticks with matplotlib.ticker "FixedLocator"
        ticks_loc = ax.get_xticks().tolist()
        ax.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
        ax.set_xticklabels(x)
        ax.tick_params(axis = 'both',labelsize = 95)    
        ax.grid(axis = 'y',alpha = 1)
        bw_n = bw_n+bw
        i += 1

    ax.axhline(y = 2.5, color = 'r', linestyle = '-.', linewidth = 4, label = '2.5x afternoon')
    ax.axhline(y = 1, color = 'k', linestyle = '-.', linewidth = 4, label = '1.0x afternoon')
    ax.legend(loc = 'upper right',fontsize =45, title = 'Local time (LT)', title_fontsize = 45)
    return ax



# vertical gradient normalized by time and emissions and categorized by wind speed

def fig_vg_time_emis_ws(df, periods_list,gas):
    fig, ax = plt.subplots(figsize=(35, 20))
    bw = 1/7.5 #barwidth is the distance from the center divided by the # of sites
    bw_n = 0
    i = 0
    n = 2*len(periods_list) #len(seasons)
    x = ['','<2','2-3','3-4','4-5','5-6','>6','']

    my_colors = ['#377eb8','#4daf4a','#ff7f00','#f7f7f7','#ffff33','#984ea3']

    for period in periods_list: 

        if period == '12-4 PM LT':
            hatch_period = 'X'
        if period != '12-4 PM LT':
            hatch_period = None

        ax.bar(df['VG_EM'][period].index+bw_n-0.3, df['VG_EM'][period], 
                yerr=0, ecolor='black',width = bw, 
                error_kw=dict(lw=8, capsize=(100*bw), capthick=3), edgecolor='black', linewidth=5, 
                alpha = 1, color = my_colors[i], hatch = hatch_period, 
                label = period ) 


        ax.set_xlabel('Wind Speed (m/s)', fontsize=95)

        ax.set_ylabel(r'$\widetilde{VG}[$CO$_2$]$_{[time;emission]}$', fontsize=65)


        ax.set_ylim(0,32)

        # fixing yticks with matplotlib.ticker "FixedLocator"
        ticks_loc = ax.get_xticks().tolist()
        ax.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
        ax.set_xticklabels(x)
        ax.tick_params(axis = 'both',labelsize = 95)    
        ax.grid(axis = 'y',alpha = 1)
        bw_n = bw_n+bw
        i += 1

    ax.axhline(y = 2.5, color = 'r', linestyle = '-.', linewidth = 4, label = '2.5x afternoon')
    ax.axhline(y = 1, color = 'k', linestyle = '-.', linewidth = 4, label = '1.0x afternoon')
    ax.legend(loc = 'upper right',fontsize =45, title = 'Local time (LT)', title_fontsize = 45)
    return ax



# vertical gradient normalized by time and emissions and categorized by turbulent kinetic energy

def fig_vg_time_emis_tke(df, periods_list,gas):
    fig, ax = plt.subplots(figsize=(35, 20))
    bw = 1/11 #barwidth is the distance from the center divided by the # of sites
    bw_n = 0
    i = 0
    n = 2*len(periods_list) #len(seasons)
    x = ['','<1.0','','1.0-1.2','','1.2-1.4','','1.4-1.6','','>1.6','']

    my_colors = ['#377eb8','#4daf4a','#ff7f00','#f7f7f7','#ffff33','#984ea3']

    for period in periods_list: 

        if period == '12-4 PM LT':
            hatch_period = 'X'
        if period != '12-4 PM LT':
            hatch_period = None

        ax.bar(df['VG_EM'][period].index+bw_n-0.3, df['VG_EM'][period], 
                yerr=0, ecolor='black',width = bw, 
                error_kw=dict(lw=8, capsize=(100*bw), capthick=3), edgecolor='black', linewidth=5, 
                alpha = 1, color = my_colors[i], hatch = hatch_period, 
                label = period ) 


        ax.set_xlabel('TKE (m$^2$/s$^2$)', fontsize=95)

        ax.set_ylabel(r'$\widetilde{VG}[$CO$_2$]$_{[time;emission]}$', fontsize=65)


        ax.set_ylim(0,15)

        # fixing yticks with matplotlib.ticker "FixedLocator"
        ticks_loc = ax.get_xticks().tolist()
        ax.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
        ax.set_xticklabels(x)
        ax.tick_params(axis = 'both',labelsize = 95)    
        ax.grid(axis = 'y',alpha = 1)
        bw_n = bw_n+bw
        i += 1

    ax.axhline(y = 2.5, color = 'r', linestyle = '-.', linewidth = 4, label = '2.5x afternoon')
    ax.axhline(y = 1, color = 'k', linestyle = '-.', linewidth = 4, label = '1.0x afternoon')
    ax.legend(loc = 'upper right',fontsize =45, title = 'Local time (LT)', title_fontsize = 45)
    return ax


# boundary layer relative bias categorized by wind speed

def fig_bld_RBias(df, periods_list,gas):
    fig, ax = plt.subplots(figsize=(35, 20))
    bw = 1/7.5 #barwidth is the distance from the center divided by the # of sites
    bw_n = 0
    i = 0
    n = 2*len(periods_list) #len(seasons)
    x = ['','<2','2-3','3-4','4-5','5-6','>6','']

    my_colors = ['#377eb8','#4daf4a','#ff7f00','#f7f7f7','#ffff33','#984ea3']

    for period in periods_list: 

        if period == '12-4 PM LT':
            hatch_period = 'X'
        if period != '12-4 PM LT':
            hatch_period = None

        ax.bar(df[period].index+bw_n-0.3, df[period], 
                yerr=0, ecolor='black',width = bw, 
                error_kw=dict(lw=8, capsize=(100*bw), capthick=3), edgecolor='black', linewidth=5, 
                alpha = 1, color = my_colors[i], hatch = hatch_period, 
                label = period ) 


        ax.set_xlabel('Wind Speed (m/s)', fontsize=95)

        ax.set_ylabel(r'BLD bias$_{[time]}$ (%)', fontsize=65)


        ax.set_ylim(-50,500)

        # fixing yticks with matplotlib.ticker "FixedLocator"
        ticks_loc = ax.get_xticks().tolist()
        ax.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
        ax.set_xticklabels(x)
        ax.tick_params(axis = 'both',labelsize = 95)    
        ax.grid(axis = 'y',alpha = 1)
        bw_n = bw_n+bw
        i += 1

    ax.axhline(y = 30, color = 'r', linestyle = '-.', linewidth = 4, label = '$\pm$30%')
    ax.axhline(y = -30, color = 'r', linestyle = '-.', linewidth = 4)
    ax.legend(loc = 'upper right',fontsize =45, title = 'Local time (LT)', title_fontsize = 45)
    return ax



# boundary layer relative bias categorized by turbulent kinetic energy

def fig_bld_RBias_tke(df, periods_list,gas):
    fig, ax = plt.subplots(figsize=(35, 20))
    bw = 1/11 #barwidth is the distance from the center divided by the # of sites
    bw_n = 0
    i = 0
    n = 2*len(periods_list) #len(seasons)
    x = ['','<1.0','','1.0-1.2','','1.2-1.4','','1.4-1.6','','>1.6','']

    my_colors = ['#377eb8','#4daf4a','#ff7f00','#f7f7f7','#ffff33','#984ea3']

    for period in periods_list: 

        if period == '12-4 PM LT':
            hatch_period = 'X'
        if period != '12-4 PM LT':
            hatch_period = None

        ax.bar(df['R_Error(%)'][period].index+bw_n-0.3, df['R_Error(%)'][period], 
                yerr=0, ecolor='black',width = bw, 
                error_kw=dict(lw=8, capsize=(100*bw), capthick=3), edgecolor='black', linewidth=5, 
                alpha = 1, color = my_colors[i], hatch = hatch_period, 
                label = period ) 


        ax.set_xlabel('TKE (m$^2$/s$^2$)', fontsize=95)

        ax.set_ylabel(r'BLD bias$_{[time]}$ (%)', fontsize=65)


        ax.set_ylim(-50,500)

        # fixing yticks with matplotlib.ticker "FixedLocator"
        ticks_loc = ax.get_xticks().tolist()
        ax.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
        ax.set_xticklabels(x)
        ax.tick_params(axis = 'both',labelsize = 95)    
        ax.grid(axis = 'y',alpha = 1)
        bw_n = bw_n+bw
        i += 1

    ax.axhline(y = 30, color = 'r', linestyle = '-.', linewidth = 4, label = '$\pm$30%')
    ax.axhline(y = -30, color = 'r', linestyle = '-.', linewidth = 4)
    ax.legend(loc = 'upper right',fontsize =45, title = 'Local time (LT)', title_fontsize = 45)
    return ax



# boundary layer bias categorized by wind speed

def fig_bld_Bias(df, periods_list,gas):
    fig, ax = plt.subplots(figsize=(35, 20))
    bw = 1/7.5 #barwidth is the distance from the center divided by the # of sites
    bw_n = 0
    i = 0
    n = 2*len(periods_list) #len(seasons)
    x = ['','<2','2-3','3-4','4-5','5-6','>6','']

    my_colors = ['#377eb8','#4daf4a','#ff7f00','#f7f7f7','#ffff33','#984ea3']

    for period in periods_list: 

        if period == '12-4 PM LT':
            hatch_period = 'X'
        if period != '12-4 PM LT':
            hatch_period = None

        ax.bar(df[period].index+bw_n-0.3, df[period], 
                yerr=0, ecolor='black',width = bw, 
                error_kw=dict(lw=8, capsize=(100*bw), capthick=3), edgecolor='black', linewidth=5, 
                alpha = 1, color = my_colors[i], hatch = hatch_period, 
                label = period ) 


        ax.set_xlabel('Wind Speed (m/s)', fontsize=95)

        ax.set_ylabel(r'BLD bias$_{[time]}$ (m)', fontsize=65)



        # fixing yticks with matplotlib.ticker "FixedLocator"
        ticks_loc = ax.get_xticks().tolist()
        ax.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
        ax.set_xticklabels(x)
        ax.tick_params(axis = 'both',labelsize = 95)    
        ax.grid(axis = 'y',alpha = 1)
        bw_n = bw_n+bw
        i += 1
    ax.legend(loc = 'upper right',fontsize =45, title = 'Local time (LT)', title_fontsize = 45)
    return ax


# enhancement categorized by wind speed

def fig_xs_ws(df, var, periods_list,gas):
    fig, ax = plt.subplots(figsize=(35, 20))
    bw = 1/7.5 #barwidth is the distance from the center divided by the # of sites
    bw_n = 0
    i = 0
    n = 2*len(periods_list) #len(seasons)

      

    my_colors = ['#377eb8','#4daf4a','#ff7f00','#f7f7f7','#ffff33','#984ea3']

    for period in periods_list: 
        
 

        if period == '12-4 PM LT':
            hatch_period = 'X'
        if period != '12-4 PM LT':
            hatch_period = None

        ax.bar(df[period].index+bw_n-0.3, df[period], 
                yerr=0, ecolor='black',width = bw, 
                error_kw=dict(lw=8, capsize=(100*bw), capthick=3), edgecolor='black', linewidth=5, 
                alpha = 1, color = my_colors[i], hatch = hatch_period, 
                label = period ) 


        ax.set_xlabel('Wind Speed (m/s)', fontsize=95)

        
        if var == 'XS':
            ax.set_ylabel(r'CO$_2$ xs$_{obs}$ (ppm)', fontsize=95)
            ax.set_ylim(0,15)
            
        if var == 'XS_MODEL':
            ax.set_ylabel(r'CO$_2$ xs$_{model}$ (ppm)', fontsize=95)
            ax.set_ylim(0,15)
        
        if var == 'XS_FF_BLD':
            ax.set_ylabel(r'CO$_2$ xs$_{obs}$/ff$_{emissions}$/BLD$_{obs}$' "\n" r'$\regular_{(ppm) * (mol/h * 10^{8})^{-1} * (m)^{-1}}$', fontsize=75)
            ax.set_ylim(0,0.4)
            
        if var == 'XSModel_FF_BLD':
            ax.set_ylabel(r'CO$_2$ xs$_{model}$/ff$_{emissions}$/BLD$_{model}$' "\n" r'$\regular_{(ppm) * (mol/h * 10^{8})^{-1} * (m)^{-1}}$', fontsize=75)
            ax.set_ylim(0,0.4)

        # fixing yticks with matplotlib.ticker "FixedLocator"
        ticks_loc = ax.get_xticks().tolist()
        ax.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
        try:
            x = ['','<2','2-3','3-4','4-5','5-6','>6','']
            ax.set_xticklabels(x)
        except ValueError:
            x = ['<2','2-3','3-4','4-5','5-6','>6'] 
            ax.set_xticklabels(x)
        
        
        ax.tick_params(axis = 'both',labelsize = 95)    
        ax.grid(axis = 'y',alpha = 1)
        bw_n = bw_n+bw
        i += 1
        
    ax.set_xlim(0.45,5.5)

  
    ax.legend(loc = 'upper right',fontsize =45, title = 'Local time (LT)', title_fontsize = 45)
    return ax



# enhancement categorized by turbulent kinetic energy

def fig_xs_tke(df, var, periods_list,gas):
    fig, ax = plt.subplots(figsize=(35, 20))
    bw = 1/11 #barwidth is the distance from the center divided by the # of sites
    bw_n = 0
    i = 0
    n = 2*len(periods_list) #len(seasons)
    x = ['','<1.0','','1.0-1.2','','1.2-1.4','','1.4-1.6','','>1.6','']

    my_colors = ['#377eb8','#4daf4a','#ff7f00','#f7f7f7','#ffff33','#984ea3']

    for period in periods_list: 

        if period == '12-4 PM LT':
            hatch_period = 'X'
        if period != '12-4 PM LT':
            hatch_period = None

        ax.bar(df[period].index+bw_n-0.3, df[period], 
                yerr=0, ecolor='black',width = bw, 
                error_kw=dict(lw=8, capsize=(100*bw), capthick=3), edgecolor='black', linewidth=5, 
                alpha = 1, color = my_colors[i], hatch = hatch_period, 
                label = period ) 


        ax.set_xlabel('TKE (m$^2$/s$^2$)', fontsize=95)

        
        if var == 'XS':
            ax.set_ylabel(r'CO$_2$ xs$_{obs}$ (ppm)', fontsize=95)
            ax.set_ylim(0,15)
            
        if var == 'XS_MODEL':
            ax.set_ylabel(r'CO$_2$ xs$_{model}$ (ppm)', fontsize=95)
            ax.set_ylim(0,15)
        
        if var == 'XS_FF_BLD':
            ax.set_ylabel(r'CO$_2$ xs$_{obs}$/ff$_{emissions}$/BLD$_{obs}$' "\n" r'$\regular_{(ppm) * (mol/h * 10^{8})^{-1} * (m)^{-1}}$', fontsize=75)
            ax.set_ylim(0,0.22)
            
        if var == 'XSModel_FF_BLD':
            ax.set_ylabel(r'CO$_2$ xs$_{model}$/ff$_{emissions}$/BLD$_{model}$' "\n" r'$\regular_{(ppm) * (mol/h * 10^{8})^{-1} * (m)^{-1}}$', fontsize=75)
            ax.set_ylim(0,0.22)

        # fixing yticks with matplotlib.ticker "FixedLocator"
        ticks_loc = ax.get_xticks().tolist()
        ax.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
        ax.set_xticklabels(x)
        ax.tick_params(axis = 'both',labelsize = 95)    
        ax.grid(axis = 'y',alpha = 1)
        bw_n = bw_n+bw
        i += 1
        
   
    ax.legend(loc = 'upper right',fontsize =45, title = 'Local time (LT)', title_fontsize = 45)

    return ax


# enhancement/co2difference categorized by wind speed

def fig_xs_diff(df,df_sem, periods_list,gas):

    n = 2*len(periods_list)

    fig, ax = plt.subplots(figsize=(45, 25)) #DCO2 VS TKE
    bw = 1/11 #barwidth is the distance from the center divided by the # of sites
    bw_n = 0
    i = 0
    my_colors = ['#377eb8','#4daf4a','#ff7f00','#f7f7f7','#ffff33','#984ea3']

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    ax.bar(df.index, df, 
            yerr=(df_sem), ecolor='black',width = bw*11, 
            error_kw=dict(lw=8, capsize=(100*bw), capthick=3), edgecolor='black', linewidth=5, 
            color = my_colors, hatch = np.where(df.index == '12-4 PM LT','X',None),
            label = None) #color = site01_color, 

    ax.set_ylabel(r'CO$_2$ xs$_{obs}$/$\mid$CO$_2$diff$\mid$', fontsize=95)

    ax.tick_params(axis = 'y',labelsize = 95)    
    ax.tick_params(axis = 'x',labelsize=80,rotation = 30)
    ax.grid(axis='y',alpha = 1)
    ax.set_ylim(0,20)
    bw_n = bw_n+bw
    i = i+1
    ax.axhline(y = 1, color = 'k', linestyle = '-.', linewidth = 4, label = r'CO$_2$ xs$_{obs}$/$\mid$CO$_2$diff$\mid$ = 1.0')
    ax.legend(loc = 'upper right',fontsize =55)#,title='Wind speed > 5m/s', title_fontsize = 55)


# vertical gradient categorized by wind speed

def fig_vg_time_ws(df, periods_list,gas):
    fig, ax = plt.subplots(figsize=(35, 20))
    bw = 1/7.5 #barwidth is the distance from the center divided by the # of sites
    bw_n = 0
    i = 0
    n = 2*len(periods_list) #len(seasons)
    x = ['','<2','2-3','3-4','4-5','5-6','>6','']

    my_colors = ['#377eb8','#4daf4a','#ff7f00','#f7f7f7','#ffff33','#984ea3']

    for period in periods_list: 

        if period == '12-4 PM LT':
            hatch_period = 'X'
        if period != '12-4 PM LT':
            hatch_period = None

        ax.bar(df['VG'][period].index+bw_n-0.3, df['VG'][period], 
                yerr=0, ecolor='black',width = bw, 
                error_kw=dict(lw=8, capsize=(100*bw), capthick=3), edgecolor='black', linewidth=5, 
                alpha = 1, color = my_colors[i], hatch = hatch_period, 
                label = period ) 


        ax.set_xlabel('Wind Speed (m/s)', fontsize=95)
        if gas == 'co2':
            ax.set_ylabel(r'VG[CO$_2$]$_{[time]}$ (ppm/m)', fontsize=95)
            ax.set_ylim(-0.78,0.1)

        if gas == 'ch4':
            ax.set_ylabel(r'VG[CH$_4$]$_{[time]}$ (ppb/m)', fontsize=95)
            ax.set_ylim(-5,0.1)

        # fixing yticks with matplotlib.ticker "FixedLocator"
        ticks_loc = ax.get_xticks().tolist()
        ax.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
        ax.set_xticklabels(x)
        ax.tick_params(axis = 'both',labelsize = 95)    
        ax.grid(axis = 'y',alpha = 1)
        bw_n = bw_n+bw
        i += 1
    ax.legend(fontsize =45, title = 'Local time (LT)', title_fontsize = 45)
    return ax


# vertical gradient normalized by ff emissions categorized by wind speed

def fig_VGFF_ws(df, var, periods_list,gas):
    fig, ax = plt.subplots(figsize=(35, 20))
    bw = 1/7.5 #barwidth is the distance from the center divided by the # of sites
    bw_n = 0
    i = 0
    n = 2*len(periods_list) #len(seasons)
    x = ['','<2','2-3','3-4','4-5','5-6','>6','']
  

    my_colors = ['#377eb8','#4daf4a','#ff7f00','#f7f7f7','#ffff33','#984ea3']

    for period in periods_list: 

        if period == '12-4 PM LT':
            hatch_period = 'X'
        if period != '12-4 PM LT':
            hatch_period = None

        ax.bar(df[period].index+bw_n-0.3, df[period], 
                yerr=0, ecolor='black',width = bw, 
                error_kw=dict(lw=8, capsize=(100*bw), capthick=3), edgecolor='black', linewidth=5, 
                alpha = 1, color = my_colors[i], hatch = hatch_period, 
                label = period ) 


        ax.set_xlabel('Wind Speed (m/s)', fontsize=95)

        ax.set_ylabel(r'VG[CO$_2$]$_{[time]}$/ff$_{emissions}$$_{[time]}$' "\n" r'$\regular_{(ppm/m) * (mol/h * 10^{8})^{-1} }$', fontsize=75)


        # fixing yticks with matplotlib.ticker "FixedLocator"
        ticks_loc = ax.get_xticks().tolist()
        ax.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
        ax.set_xticklabels(x)
        ax.tick_params(axis = 'both',labelsize = 95)    
        ax.grid(axis = 'y',alpha = 1)
        bw_n = bw_n+bw
        i += 1
        
    ax.legend(fontsize =45, title = 'Local time (LT)', title_fontsize = 45)
    return ax

# vertical gradient normalized by ff emissions categorized by turbulent kinetic energy

def fig_VGFF_tke(df, var, periods_list,gas):
    fig, ax = plt.subplots(figsize=(35, 20))
    bw = 1/11 #barwidth is the distance from the center divided by the # of sites
    bw_n = 0
    i = 0
    n = 2*len(periods_list) #len(seasons)
    x = ['','<1.0','','1.0-1.2','','1.2-1.4','','1.4-1.6','','>1.6','']

       

    my_colors = ['#377eb8','#4daf4a','#ff7f00','#f7f7f7','#ffff33','#984ea3']

    for period in periods_list: 

        if period == '12-4 PM LT':
            hatch_period = 'X'
        if period != '12-4 PM LT':
            hatch_period = None

        ax.bar(df[period].index+bw_n-0.3, df[period], 
                yerr=0, ecolor='black',width = bw, 
                error_kw=dict(lw=8, capsize=(100*bw), capthick=3), edgecolor='black', linewidth=5, 
                alpha = 1, color = my_colors[i], hatch = hatch_period, 
                label = period ) 


        ax.set_xlabel('TKE (m$^2$/s$^2$)', fontsize=95)

        ax.set_ylabel(r'VG[CO$_2$]$_{[time]}$/ff$_{emissions}$$_{[time]}$' "\n" r'$\regular_{(ppm/m) * (mol/h * 10^{8})^{-1} }$', fontsize=75)
            #ax.set_ylim(-1.7,0)
            

        # fixing yticks with matplotlib.ticker "FixedLocator"
        ticks_loc = ax.get_xticks().tolist()
        ax.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
        ax.set_xticklabels(x)
        ax.tick_params(axis = 'both',labelsize = 95)    
        ax.grid(axis = 'y',alpha = 1)
        bw_n = bw_n+bw
        i += 1

    ax.legend(fontsize =45, title = 'Local time (LT)', title_fontsize = 45)
    return ax


# wind speed model vs obs
def ws_mod_ob(df,season):

    from sklearn.metrics import r2_score

    lg_labels = [ '00:00 - 04:59','05:00 - 08:59','09:00 - 11:59', '12:00 - 16:59',
                 '17:00 - 20:59','21:00 - 23:59']

    i=0
    for key in [ '0-4 AM LT','5-8 AM LT','9-11 AM LT', '12-4 PM LT', '5-8 PM LT','9-11 PM LT']:
        fig, ax4 = plt.subplots(figsize=(20, 20)) #PLOT OF VERTICAL GRADIENTS VS BLD CATEGORIES

        xx = df['WS_OBS'].loc[(df['PERIOD'] == key)&(df['SEASON'] == season)]
        yy = df['WRF_WS'].loc[(df['PERIOD'] == key)&(df['SEASON'] == season)]
        idx = np.isfinite(xx) & np.isfinite(yy)
        m,b = np.polyfit(xx[idx], yy[idx], 1)
        r = np.corrcoef(xx[idx], yy[idx])
        ax4.plot(xx,
                 yy, 'o', color = 'black',
                 markersize= 20, linewidth=5, label = lg_labels[i])

        ax4.plot(xx, m*xx + b, "r-", lw=5, label = 'r$^2$ = '+str(r[1,0].round(2)))
        ax4.axline((0, 0), slope=1, linestyle = '-.',lw=3, color = 'black', label = '1:1')


        ax4.set_xlabel(r'Observed wind speed (m/s)', fontsize=95)
        ax4.set_ylabel(r'Modeled wind speed (m/s)', fontsize=95)
        ax4.set_ylim(-0.1,20)
        ax4.set_xlim(-0.1,20)
        ax4.legend(loc = 'best',fontsize =45)
        ax4.tick_params(axis = 'both',labelsize = 95) 
        i = i+1

        print(key,': y=',m.round(2),'x +',b.round(2))
        
    return ax4

# wind speed residuals
def fig_ws_bias(df, season): #DATA['WSP']
    i=0
    
    lg_labels = [ '00:00 - 04:59','05:00 - 08:59','09:00 - 11:59', '12:00 - 16:59',
             '17:00 - 20:59','21:00 - 23:59']
    
    for key in [ '0-4 AM LT','5-8 AM LT','9-11 AM LT', '12-4 PM LT', '5-8 PM LT','9-11 PM LT']:
        fig, ax = plt.subplots(figsize=(20, 20)) #PLOT OF VERTICAL GRADIENTS VS BLD CATEGORIES

        
        #wind speed < 5 m/s (equivalent to wsp category 0 to 3 as defined at the beginning of the code)
        temp = df.loc[(df['WS_OBS_CAT']<4)&(df['PERIOD'] == key)&(df['SEASON'] == season)]
        WSP_l5 = temp['Error'].mean()

        # wind speed >= 5 m/s (equivalent to wsp category >= 4 as defined at the beginning of the code)
        temp = df.loc[(df['WS_OBS_CAT']>=4)&(df['PERIOD'] == key)&(df['SEASON'] == season)]
        WSP_g5 = temp['Error'].mean()


        
        xx2 = df.loc[(df['WS_OBS_CAT']<4)&(df['PERIOD'] == key)&(df['SEASON']==season)].index
        yy2 = df['Error'].loc[(df['WS_OBS_CAT']<4)&(df['PERIOD'] == key) &(df['SEASON']==season)]

        xx1 = df.loc[(df['WS_OBS_CAT']>=4)&(df['PERIOD'] == key)&(df['SEASON']==season)].index
        yy1 = df['Error'].loc[(df['WS_OBS_CAT']>=4)&(df['PERIOD'] == key) &(df['SEASON']==season)]

        ax.plot(xx1,
                 yy1, 'o', color = 'black', 
                 markersize= 20, linewidth=5, label = r'$\geq$ 5 m/s')

        ax.plot(xx2,
                 yy2, 'o', color = 'black', 
                 alpha = 0.3, markersize= 20, linewidth=5, label = r'< 5 m/s')

        ax.axhline(y=WSP_g5, color = 'k', linestyle = '-.',
                    linewidth = 4, label = r'Mean bias ($\geq$ 5m/s) =' +str(WSP_g5.round(1)))

        ax.axhline(y=WSP_l5, color = 'k', linestyle = '-.',alpha =0.3,
                    linewidth = 4, label = r'Mean bias (< 5m/s) =' +str(WSP_l5.round(1)))



        ax.set_ylabel(r'Wind speed bias (m/s)', fontsize=95)
        ax.set_ylim(-6.5,6.5)
        ax.legend(loc = 'best',fontsize =35, title = lg_labels[i], title_fontsize = 45)
        ax.tick_params(axis = 'y',labelsize = 95) 
        ax.tick_params(axis = 'x',labelsize = 55, rotation = 30) 
        plt.grid(alpha=0.5)
        i = i+1

    return fig
    
#--------------------------------------------------------------------------------------------------------

#### TABLE
def table_mae_bias(df_in, var_obs, var_model, periods_list):

    from sklearn.metrics import mean_absolute_error as mae
    Errors = pd.DataFrame(index= periods_list, columns = ['N','MEAN','MAE','BIAS'])


    for key in periods_list: 

        #%% MAE and BIAS 
        
        df = df_in.loc[df_in['PERIOD'] == key]

        y_copy = df[[var_obs,var_model]].copy()
        y_copy = y_copy.dropna()
        y_true = y_copy[var_obs]
        y_pred = y_copy[var_model]


        bias = (y_pred.sum() - y_true.sum())/len(y_true) #BIAS
        error = mae(y_true,y_pred ) #MAE


        Errors.loc[key]['MAE'] = error.round(1) 
        Errors.loc[key]['BIAS'] = bias.round(1) 
        Errors.loc[key]['MEAN'] = df[var_obs].mean().round(1) 
        Errors.loc[key]['N'] = len(y_pred)   # NUMBER OF POINTS

    return Errors
