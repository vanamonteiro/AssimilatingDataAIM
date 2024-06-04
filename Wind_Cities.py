import numpy as np
import matplotlib.pyplot as plt
import time
import matplotlib.dates as mdates
import chardet
import glob
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.ticker as mticker
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

"""
This is a python script with functions used for the
Monteiro, et al., 2024 -- FIGURES 8 AND 9 (WIND FOR DIFFERENT CITIES)

This code was written by Vanessa Monteiro.

Last update 04 June 2024

"""

# READ WIND SPEED FROM ALL CITIES 
def read_wsp_cities(path):

    weather = {}
    for file in glob.glob(path):
        with open(file, 'rb') as f:
            result = chardet.detect(f.read())  # or readline if the file is large

        df = pd.read_csv(file, comment = "#", encoding=result['encoding'])
        key = df['station'][0]
        df.index = pd.to_datetime(df['valid'], format = '%Y-%m-%d %H:%M')
        df = df.tz_localize(tz = 'UTC')
        df = pd.DataFrame(df.resample('H').mean())
        df['sped_ms'] = df['sped']/2.237  #it is in mph, so divide the speed value by 2.237 to m/s
        
        weather[key] = df

    key_list = list(weather.keys())
    
    return key_list, weather

##  Plot figures for each city

def wind_by_cities(key_list,weather):
    
    
    # will convert to LOCAL TIME
    for key in key_list:
    
        if key == 'BOS':
            weather[key].index = weather[key].index.tz_convert('America/New_York')

        if key == 'BWI':
            weather[key].index = weather[key].index.tz_convert('America/New_York')

        if key == 'CYYZ':
            weather[key].index = weather[key].index.tz_convert('America/Toronto')

        if key == 'EDDM':
            weather[key].index = weather[key].index.tz_convert('Europe/Paris')
        
        if key == 'EHRD':
            weather[key].index = weather[key].index.tz_convert('Europe/Paris')
        
        if key == 'IAD':
            weather[key].index = weather[key].index.tz_convert('America/New_York')

        if key == 'LAX':
            weather[key].index = weather[key].index.tz_convert('America/Los_Angeles')
            
        if key == 'LFPG':
            weather[key].index = weather[key].index.tz_convert('Europe/Paris')
            
        if key == 'LSZH':
            weather[key].index = weather[key].index.tz_convert('Europe/Zurich')

        if key == 'NZAA':
            weather[key].index = weather[key].index.tz_convert('Pacific/Auckland')

        if key == 'RJAA':
            weather[key].index = weather[key].index.tz_convert('Asia/Tokyo')

        if key == 'SBGR':
            weather[key].index = weather[key].index.tz_convert('America/Sao_Paulo')

        if key == 'SLC':
            weather[key].index = weather[key].index.tz_convert('America/Denver')
            
        if key == 'WIII':
            weather[key].index = weather[key].index.tz_convert('Asia/Jakarta')

        if key == 'YMML':
            weather[key].index = weather[key].index.tz_convert('Australia/Melbourne')

        if key == 'ZBAD':
            weather[key].index = weather[key].index.tz_convert('Asia/Shanghai')



    #%%  ######## CATEGORIZED WIND SPEED
    for key in key_list:
    
        # created a dic to store the wind speed by wind cetegory
        weather[key]['Wind_category'] = np.nan

        weather[key]['Wind_category'].loc[(weather[key]['sped_ms'] <2)] = 0 #'<2'
        weather[key]['Wind_category'].loc[(weather[key]['sped_ms'] >=2)&(weather[key]['sped_ms'] <3)] = 1# '2-3'
        weather[key]['Wind_category'].loc[(weather[key]['sped_ms'] >=3)&(weather[key]['sped_ms'] <4)] = 2#'3-4'
        weather[key]['Wind_category'].loc[(weather[key]['sped_ms'] >=4)&(weather[key]['sped_ms'] <5)] = 3#'4-5'
        weather[key]['Wind_category'].loc[(weather[key]['sped_ms'] >=5)&(weather[key]['sped_ms'] <6)] = 4#'5-6'
        weather[key]['Wind_category'].loc[(weather[key]['sped_ms'] >=6)] = 5#'>=6'


    ## subset by periods of the day
    periods = [ '0-4 AM LT','5-8 AM LT','9-11 AM LT', '12-4 PM LT', '5-8 PM LT','9-11 PM LT', '24 Hours']
    # created a dic to store wind speed subsets by period of the day
    weather_subset = {}

    for key in key_list:
        for hours in periods:
            if hours == '24 Hours':
                weather_subset[key,hours] = weather[key].copy()

            if hours == '5-8 AM LT':
                weather_subset[key,hours] = weather[key].loc[((weather[key].index.hour >= 5)&(weather[key].index.hour<=8))].copy()  

            if hours == '9-11 AM LT':
                weather_subset[key,hours] = weather[key].loc[((weather[key].index.hour >= 9)&(weather[key].index.hour<=11))].copy()

            if hours == '12-4 PM LT':
                weather_subset[key,hours] = weather[key].loc[((weather[key].index.hour >= 12)&(weather[key].index.hour<=16))].copy()

            if hours == '5-8 PM LT':
                weather_subset[key,hours] = weather[key].loc[((weather[key].index.hour >= 17)&(weather[key].index.hour<=20))].copy()  

            if hours == '9-11 PM LT':
                weather_subset[key,hours] = weather[key].loc[((weather[key].index.hour >= 21)&(weather[key].index.hour<=23))].copy() 

            if hours == '0-4 AM LT':
                weather_subset[key,hours] = weather[key].loc[((weather[key].index.hour >= 0)&(weather[key].index.hour<=4))].copy() 



    #Figures
    
    # Wind speed list (>= 0, >=2, >=3, so on...)
    ws_list = ['WS_0','WS_2','WS_3','WS_4','WS_5','WS_6']
    #threshold list (equivalent to ws_list)
    th_list = [0,1,2,3,4,5,6]

    # dic to store the wind speed greater than a threshold
    dic_ws_greater = {}

    for th in th_list:
        for key in key_list:
            
            #N : number of points, P : percentage, MEAN : mean from the subset
            dic_ws_greater[key,'WS_'+str(th)] = pd.DataFrame(index= periods, columns = ['Name','N','P','MEAN'])

            for hours in periods:

                #%% WIND SPEED AIRPORT VS WRF
                
                #subset by threshold
                weather_subset[key,hours,'WS_'+str(th)] = weather_subset[key,hours].loc[weather_subset[key,hours]['sped_ms']>=th] 
                y_size_temp = weather_subset[key,'24 Hours'].loc[weather_subset[key,'24 Hours']['sped_ms']>=0] #0
                # length of the subset
                y_size = len(y_size_temp[['sped_ms']].dropna())
                

                y_copy = weather_subset[key,hours,'WS_'+str(th)][['sped_ms']].copy()
                y_copy = y_copy.dropna()
                y_pred = y_copy['sped_ms']


                dic_ws_greater[key,'WS_'+str(th)].loc[hours]['Name'] = 'WS_'+str(th)
                dic_ws_greater[key,'WS_'+str(th)].loc[hours]['P'] =  len(y_pred)/y_size
                dic_ws_greater[key,'WS_'+str(th)].loc[hours]['MEAN'] = weather_subset[key,hours,'WS_'+str(th)]['sped_ms'].mean()
                dic_ws_greater[key,'WS_'+str(th)].loc[hours]['N'] = len(y_pred)  

    
    ## city by city
    my_colors2 = ['#377eb8','#4daf4a','#ff7f00','#ffff33','#984ea3']
    color_afternoon='#f7f7f7'
    lg_labels_datafraction = ['12:00-04:59 PM \n (WS $\geq$0 m/s)', '00:00-04:59 AM','05:00-08:59 AM','09:00-11:59 AM', 
                 '05:00-08:59 PM','09:00-11:59 PM']


    for key in key_list:  
    
        #this is a trick to create the stacked figure // numbers 0 to 6 represent the period of the day. 
        y0 = [dic_ws_greater[key,'WS_0']['P'][0],dic_ws_greater[key,'WS_1']['P'][0],dic_ws_greater[key,'WS_2']['P'][0],dic_ws_greater[key,'WS_3']['P'][0],
              dic_ws_greater[key,'WS_4']['P'][0],dic_ws_greater[key,'WS_5']['P'][0],dic_ws_greater[key,'WS_6']['P'][0]]
        y1 = [dic_ws_greater[key,'WS_0']['P'][1],dic_ws_greater[key,'WS_1']['P'][1],dic_ws_greater[key,'WS_2']['P'][1],dic_ws_greater[key,'WS_3']['P'][1],
              dic_ws_greater[key,'WS_4']['P'][1],dic_ws_greater[key,'WS_5']['P'][1],dic_ws_greater[key,'WS_6']['P'][1]]
        y2 = [dic_ws_greater[key,'WS_0']['P'][2],dic_ws_greater[key,'WS_1']['P'][2],dic_ws_greater[key,'WS_2']['P'][2],dic_ws_greater[key,'WS_3']['P'][2],
              dic_ws_greater[key,'WS_4']['P'][2],dic_ws_greater[key,'WS_5']['P'][2],dic_ws_greater[key,'WS_6']['P'][2]]
        y3 = [dic_ws_greater[key,'WS_0']['P'][3],dic_ws_greater[key,'WS_0']['P'][3],dic_ws_greater[key,'WS_0']['P'][3],dic_ws_greater[key,'WS_0']['P'][3],
              dic_ws_greater[key,'WS_0']['P'][3],dic_ws_greater[key,'WS_0']['P'][3],dic_ws_greater[key,'WS_0']['P'][3]]
        y4 = [dic_ws_greater[key,'WS_0']['P'][4],dic_ws_greater[key,'WS_1']['P'][4],dic_ws_greater[key,'WS_2']['P'][4],dic_ws_greater[key,'WS_3']['P'][4],
              dic_ws_greater[key,'WS_4']['P'][4],dic_ws_greater[key,'WS_5']['P'][4],dic_ws_greater[key,'WS_6']['P'][4]]
        y5 = [dic_ws_greater[key,'WS_0']['P'][5],dic_ws_greater[key,'WS_1']['P'][5],dic_ws_greater[key,'WS_2']['P'][5],dic_ws_greater[key,'WS_3']['P'][5],
              dic_ws_greater[key,'WS_4']['P'][5],dic_ws_greater[key,'WS_5']['P'][5],dic_ws_greater[key,'WS_6']['P'][5]]
        y6= [dic_ws_greater[key,'WS_0']['P'][6],dic_ws_greater[key,'WS_1']['P'][6],dic_ws_greater[key,'WS_2']['P'][6],dic_ws_greater[key,'WS_3']['P'][6],
              dic_ws_greater[key,'WS_4']['P'][6],dic_ws_greater[key,'WS_5']['P'][6],dic_ws_greater[key,'WS_6']['P'][6]]
        
        x = [r'$\geq$0',r'$\geq$1',r'$\geq$2',r'$\geq$3',r'$\geq$4', r'$\geq$5',r'$\geq$6']
       
        # stacking percentages
        index = pd.Index(x, name='test')
        data = {r'0-4 AM LT': y0,'5-8 AM LT': y1, '9-11 AM LT': y2,
                 '5-8 PM LT': y4, '9-11 PM LT': y5}
        df = pd.DataFrame(data, index=index)
        data2 = {r'12-4 PM LT ($\geq$ 0 m/s)': y3}
        df2 = pd.DataFrame(data2, index=index)
        

        ax = df2.plot(kind='bar', stacked=True, figsize=(35, 20),edgecolor='black', linewidth=5, hatch = 'x', color = color_afternoon,legend=None)


        df.plot(kind = 'bar',figsize=(35, 20), stacked=True,edgecolor='black', linewidth=5, 
                     color = my_colors2, bottom = df2['12-4 PM LT ($\geq$ 0 m/s)'], ax = ax,legend=None)#, label = None)

        ax.set_ylim(0,1)
        ax.set_xlabel('Wind Speed (m/s)', fontsize = 95)
        ax.set_ylabel('Data fraction', fontsize = 95)

        plt.yticks(fontsize = 95)
        plt.xticks(rotation= 0,fontsize = 85) 

        # label cities
        if key == 'BOS':
            city = 'Boston'
        if key == 'BWI':
            city = 'Baltimore'
        if key == 'CYYZ':
            city = 'Toronto'
        if key == 'EDDM':
            city = 'Munich'
        if key == 'EHRD':
            city = 'Rotterdam'
        if key == 'IAD':
            city = 'Washington DC'
        if key == 'IND':
            city = 'Indianapolis'
        if key == 'LAX':
            city = 'Los Angeles'
        if key == 'LFPG':
            city = 'Paris'
        if key == 'LSZH':
            city = 'Zurich'
        if key == 'NZAA':
            city = 'Auckland'
        if key == 'RJAA':
            city = 'Tokyo'
        if key == 'SBGR':
            city = 'Sao Paulo'
        if key == 'SLC':
            city = 'Salt Lake City'
        if key == 'WIII':
            city = 'Jakarta'
        if key == 'YMML':
            city = 'Melbourne'
        if key == 'ZBAD':
              city = 'Beijing' 
        if key == 'LFQA':
            city = 'Reims'
            
        ax.text(.7,.9, city+' ('+key+')', horizontalalignment = 'center',
               fontsize = 105,
               transform=ax.transAxes)


    return ax, dic_ws_greater
    

# summary of all cities  for wind speed greater than 5 m/s only
    
def all_cities(dic_ws_greater):

    ## all cities
    
    my_colors2 = ['#377eb8','#4daf4a','#ff7f00','#ffff33','#984ea3']
    color_afternoon='#f7f7f7'
    lg_labels_datafraction = ['12:00-16:59 (WS $\geq$2 m/s)', '00:00-04:59 (WS $\geq$5 m/s)','05:00-08:59 (WS $\geq$5 m/s)',
                              '09:00-11:59 (WS $\geq$5 m/s)', '17:00-20:59 (WS $\geq$5 m/s)','21:00-23:59 (WS $\geq$5 m/s)']

    df2 = pd.DataFrame()
    df = pd.DataFrame()


    key_list_ordered = ['IND','YMML','BOS','NZAA','CYYZ','LFPG','EHRD','SLC','RJAA','IAD','LAX','BWI',
                       'WIII','EDDM','SBGR','ZBAD','LSZH']


    for key in key_list_ordered:  
        y0 = [dic_ws_greater[key,'WS_5']['P'][0]]
        y1 = [dic_ws_greater[key,'WS_5']['P'][1]]
        y2 = [dic_ws_greater[key,'WS_5']['P'][2]]
        y3 = [dic_ws_greater[key,'WS_2']['P'][3]]
        y4 = [dic_ws_greater[key,'WS_5']['P'][4]]
        y5 = [dic_ws_greater[key,'WS_5']['P'][5]]
        y6 = [dic_ws_greater[key,'WS_5']['P'][6]]
        
        #label cities       
        if key == 'BOS':
            city = 'Boston'
        if key == 'BWI':
            city = 'Baltimore'
        if key == 'CYYZ':
            city = 'Toronto'
        if key == 'EDDM':
            city = 'Munich'
        if key == 'EHRD':
            city = 'Rotterdam'
        if key == 'IAD':
            city = 'Washington DC'
        if key == 'LAX':
            city = 'Los Angeles'
        if key == 'LFPG':
            city = 'Paris'
        if key == 'LSZH':
            city = 'Zurich'
        if key == 'NZAA':
            city = 'Auckland'
        if key == 'RJAA':
            city = 'Tokyo'
        if key == 'SBGR':
            city = 'Sao Paulo'
        if key == 'SLC':
            city = 'Salt Lake City'
        if key == 'WIII':
            city = 'Jakarta'
        if key == 'YMML':
            city = 'Melbourne'
        if key == 'ZBAD':
              city = 'Beijing'           
        if key == 'IND':
            city = 'Indianapolis' 
        if key == 'LFQA':
            city = 'Reims'
            
        x = [city]

        index = pd.Index(x, name='test')
        data = {r'0-4 AM LT': y0,'5-8 AM LT': y1, '9-11 AM LT': y2,
                 '5-8 PM LT': y4, '9-11 PM LT': y5}
        data1 = pd.DataFrame(data, index=index)
        df = df.append(data1)
        data2 = {r'12-4 PM LT ($\geq$ 2 m/s)': y3}
        data22 = pd.DataFrame(data2, index=index)
        df2 = df2.append(data22)
        
    df['sum'] = np.nan
    df['sum'] = df.sum(axis = 1)
    df3 = df.sort_values('sum', ascending=False)
    del df['sum']
    del df3['sum']



    ax2 = df2.plot(kind='bar', stacked=True, figsize=(35, 20),edgecolor='black', linewidth=5, hatch = 'x', color = color_afternoon,legend=None)


    df.plot(kind = 'bar',figsize=(35, 20), stacked=True,edgecolor='black', linewidth=5, 
                 color = my_colors2, bottom = df2['12-4 PM LT ($\geq$ 2 m/s)'], ax = ax2,legend=None)

    ax2.set_ylim(0,1)
    ax2.set_ylabel('Data fraction', fontsize = 75)
    ax2.set_xlabel(' ', fontsize = 95)
    plt.legend(lg_labels_datafraction,loc='upper right',fontsize=45,title='Year: 2021 - Local time (LT)', title_fontsize = 45)#, bbox_to_anchor=(1.0, 1), loc='upper left')
    plt.yticks(fontsize = 75)
    plt.xticks(rotation= 90,fontsize = 55) 
    plt.grid(axis = 'y',alpha = 0.5)

    return ax2
