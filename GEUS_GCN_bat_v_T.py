#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 13:38:41 2022

@author: jason
"""
from datetime import datetime
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from numpy.polynomial.polynomial import polyfit
from datetime import date

if os.getlogin() == 'jason':
    base_path = '/Users/jason/Dropbox/AWS/CARRA-TU_GEUS/'

today = date.today()
versionx= today.strftime('%Y-%m-%d')

os.chdir(base_path)

th=1 ; th=2 # line thickness
formatx='{x:,.3f}'; fs=24 ; fs=16
plt.rcParams["font.size"] = fs
plt.rcParams['axes.facecolor'] = 'w'
plt.rcParams['axes.edgecolor'] = 'k'
plt.rcParams['axes.grid'] = False
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.8
plt.rcParams['grid.color'] = "#cccccc"
plt.rcParams["legend.facecolor"] ='w'
plt.rcParams["mathtext.default"]='regular'
plt.rcParams['grid.linewidth'] = th/2
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['figure.figsize'] = 12, 20


meta = pd.read_excel('/Users/jason/Dropbox/AWS/test/IMEI_numbers_station_2021-10-27.xlsx')
print(meta.columns)

names=meta.ASSET

cols=['time','counter','Pressure_L','Pressure_U','Asp_temp_L','Asp_temp_U','Humidity_L','Humidity_U','WindSpeed_L','WindDirection_L','WindSpeed_U','WindDirection_U','SWD','SWU','LW Downward','LW Upward','TemperatureRadSensor','SR_A','SR_B','T_firn_1','T_firn_2','T_firn_3','T_firn_4','T_firn_5','T_firn_6','T_firn_7','T_firn_8','T_firn_9','T_firn_10','T_firn_11','Roll','Pitch','Heading','Rain_amount_L','Rain_amount_U','counterx','Latitude','Longitude','Altitude','ss','Giodal','GeoUnit','Battery','NumberSatellites','HDOP','FanCurrent_L','FanCurrent_U','Quality','LoggerTemp']
varnamx=['time','counter','Pressure_L','Pressure_U','air temperature','air temperature','Humidity_L','Humidity_U','WindSpeed_L','WindDirection_L','WindSpeed_U','WindDirection_U','SWD','SWU','LW Downward\nSky Tempearure_effective','LW Upward','TemperatureRadSensor','SR_A','SR_B','T_firn_1','T_firn_2','T_firn_3','T_firn_4','T_firn_5','T_firn_6','T_firn_7','T_firn_8','T_firn_9','T_firn_10','T_firn_11','Roll','Pitch','Heading','rainfall','rainfall','counterx','Latitude','Longitude','Altitude','ss','Giodal','GeoUnit','Battery','NumberSatellites','HDOP','FanCurrent_L','FanCurrent_U','Quality','LoggerTemp']
unitsx=['time','counter','Pressure_L','Pressure_U','deg. C','deg. C','Humidity_L','Humidity_U','WindSpeed_L','WindDirection_L','WindSpeed_U','WindDirection_U','SWD','SWU','LW Downward','LW Upward','TemperatureRadSensor','SR_A','SR_B','T_firn_1','T_firn_2','T_firn_3','T_firn_4','T_firn_5','T_firn_6','T_firn_7','T_firn_8','T_firn_9','T_firn_10','T_firn_11','Roll','Pitch','Heading','mm','mm','counterx','Latitude','Longitude','Altitude','ss','Giodal','GeoUnit','Battery','NumberSatellites','HDOP','FanCurrent_L','FanCurrent_U','Quality','LoggerTemp']

# 	Sample(1,SR_A,FP2)
# 	Sample(1,SR_B,FP2)
# 	Sample(1,IceHeight,FP2)
# 	Sample(8,TemperatureIce(),FP2)
# 	FieldNames("TemperatureIce1m,TemperatureIce2m,TemperatureIce3m,TemperatureIce4m,TemperatureIce5m,TemperatureIce6m,TemperatureIce7m,TemperatureIce10m") 
# 	Sample(1,TiltX,FP2)
# 	Sample(1,TiltY,FP2)
# 	Sample(1,TimeGPS,String)
# 	Sample(1,Latitude,String)
# 	Sample(1,Longitude,String)
# 	Sample(1,Altitude,String)
# 	Sample(1,HDOP,String)
#  	Sample(1,currents(3),FP2)
#  	Sample(1,VoltageBatteries,FP2)
# ' Values below are added at DMI's request
# 	Sample(1,AirPressureMinus1000,FP2)
#   Sample(1,Temperature,FP2)
# 	Sample(1,RelativeHumidity,FP2)
#  	WindVector (1,WindSpeed,WindDirection,FP2,False,0,0,1)
#  	FieldNames("WindSpeed,WindDirection")
# Sample(1,InstantDataTerminator,String)

# cols=['time','counter','Pressure_L','Pressure_U','Asp_temp_L','Asp_temp_U','Humidity_L','Humidity_U','WindSpeed_L','WindDirection_L','WindSpeed_U','WindDirection_U','SWD','SWU','LW Downward','LW Upward','TemperatureRadSensor','SR_A','SR_B','T_firn_1','T_firn_2','T_firn_3','T_firn_4','T_firn_5','T_firn_6','T_firn_7','T_firn_8','T_firn_9','T_firn_10','T_firn_11','Roll','Pitch','Heading','Rain_amount_L','Rain_amount_U','counterx','Latitude','Longitude','Altitude','ss','Giodal','GeoUnit','Battery','NumberSatellites','HDOP','FanCurrent_L','FanCurrent_U','Quality','LoggerTemp']

# cols=['SR_A','SR_B','IceHeight','TemperatureIce1','TemperatureIce2','TemperatureIce3','TemperatureIce4','TemperatureIce5','TemperatureIce6','TemperatureIce7','TemperatureIce8','TiltX','TiltY','TimeGPS','Latitude','Longitude','Altitude','HDOP','currents','Battery','AirPressureMinus1000','Temperature','RelativeHumidity','WindSpeed','WindDirection','InstantDataTerminator']

cols_requested=['year','day','month','hour','airpressureminus1000','temperature','relativehumidity','windspeed','winddirection','Lat_decimal','Lon_decimal','elev']

for i,name in enumerate(names):
    if i>=0:
        print()
        print(i,names[i])
        # if names[i][0:5]=='GCNET':
        cols=['timestamp','seconds_since_1990','airpressure','temperature0','temperature2','relativehumidity0','windspeed0','winddirection0','shortwaveradiationin','shortwaveradiationout','longwaveradiationin','longwaveradiationout','temperatureradsensor','SR_A','SR_B','iceheight','temperatureice1m','temperatureice2m','temperatureice3m','temperatureice4m','temperatureice5m','temperatureice6m','temperatureice7m','temperatureice10m','tiltx','tilty','timegps','Latitude','Longitude','Altitude','hdop','currents','Battery','airpressureminus1000','temperature','relativehumidity','windspeed','winddirection']
        skip=6
        if names[i][0:5]=='GCNET':
            skip=10
            cols=['timestamp','seconds_since_1990','airpressureminus1000','Pressure_U','temperature','Asp_temp_U','relativehumidity','Humidity_U','windspeed','winddirection','WindSpeed_U','WindDirection_U','SWD','SWU','LW Downward','LW Upward','TemperatureRadSensor','SR_A','SR_B','T_firn_1','T_firn_2','T_firn_3','T_firn_4','T_firn_5','T_firn_6','T_firn_7','T_firn_8','T_firn_9','T_firn_10','T_firn_11','Roll','Pitch','Heading','Rain_amount_L','Rain_amount_U','counterx','Latitude','Longitude','Altitude','ss','Giodal','GeoUnit','Battery','NumberSatellites','HDOP','FanCurrent_L','FanCurrent_U','Quality','LoggerTemp']
        if meta.alt_name[i]=='SWC' or meta.alt_name[i]=='JAR':
            skip=4
            cols=['time','seconds_since_1990','airpressureminus1000','temperature','relativehumidity','windspeed','winddirection',
                  'SWU','SWD',
                 'LW Downward','LW Upward','TemperatureRadSensor','SR_A','SR_B','solar?',
                 'thermistorstring_1','thermistorstring_2','thermistorstring_3','thermistorstring_4','thermistorstring_5','thermistorstring_6','thermistorstring_7','thermistorstring_8',
                 'roll','pitch','heading','Rain_amount_L','gpstime','Latitude','Longitude','Altitude','giodal','geounit?',
                 'Battery','?1','asp_temp_u','humidity_u','##','##2','##3']
        # if meta.alt_name[i]=='SWC':
        # if ((meta.network[i]=='g')or(meta.network[i]=='p')):
        if meta.network[i]=='g':
        # if meta.alt_name[i]=='KAN_U':
            print(names[i])
            fn='/Users/jason/Dropbox/AWS/test/aws_data/AWS_'+str(meta.IMEI[i])+'.txt'
            # print(fn)
            # os.system('open '+fn)
            
            fn2="/tmp/"+str(meta.IMEI[i])+".txt"
            msg="sed 's/,,/,/g' "+fn+" > "+fn2
            os.system(msg)
            fn=fn2    
            df=pd.read_csv(fn,header=None,names=cols,skiprows=skip)
            df[df=="NAN"]=np.nan
            df[df=="nan"]=np.nan
            df["date"]=pd.to_datetime(df.iloc[:,0])
            # df=df.drop(df.columns[0:33], axis=1)
            df['year'] = df['date'].dt.year
            t0=datetime(2020, 8, 1) ; t1=datetime(2021, 7, 8)
            t0=datetime(2021,12, 1) ; t1=datetime(2022, 4, 1)

            df['month'] = df['date'].dt.month
            df['day'] = df['date'].dt.day
            df['hour'] = df['date'].dt.hour
            df['doy'] = df['date'].dt.dayofyear
            df['days_per_year'] = 365
            df['days_per_year'][df['year']==2020]=366
            df['dec_year'] = df['year']+df['date'].dt.dayofyear/df['days_per_year']

            drop=1
            if drop:
                # if meta.alt_name[i]!='JAR':
                df.drop(df[df.dec_year<2021.5].index, inplace=True)
                df.reset_index(drop=True, inplace=True)
                t0=datetime(2021, 12, 1) ; t1=datetime(2022, 5, 1)

            # plt.plot(df['dec_year'])
            ##%%

            for kk,col in enumerate(cols):
                if kk>0:
                    df[col] = pd.to_numeric(df[col])
            # df['temperature'] = pd.to_numeric(df['temperature'])
            # df['relativehumidity'] = pd.to_numeric(df['relativehumidity'])
            # df['airpressureminus1000'] = pd.to_numeric(df['airpressureminus1000'])
            df['relativehumidity'][df['relativehumidity']>105]=np.nan
            df['airpressureminus1000'][df['airpressureminus1000']>1000]=np.nan
            if meta.network[i]=='g':
                df.airpressureminus1000-=1000
            # df.columns

            # # filter stuck wind sensor data
            # filter_me=['windspeed','winddirection']
            # # filter_me=['winddirection']
            # window=6
            # for param in filter_me:
            #     for ii in range(window,len(df)):
            #         for jj in range(window):
            #             v=df[param][ii:ii+window]==df[param][ii]
            #             if sum(v)==6:df[param][ii:ii+window]=np.nan
            #             # print(ii,sum(v))
            #             # if np.nanstd(df[param][ii:ii+window])<0.2:
            #             #     df[param][ii:ii+window]=np.nan
            #             # # print(ii,v)
                        
            # ------------------- position
            df.Latitude=df.Latitude.astype(float)
            df.Longitude=df.Longitude.astype(float)
            df.Altitude=df.Altitude.astype(float)
            ##%% lat lon slow
            df['Lat_decimal']=np.nan
            df['Lon_decimal']=np.nan
            
            v=np.where(np.isfinite(df.Latitude))
            v=v[0]
            df['lat_min']=np.nan
            df['lon_min']=np.nan
            df['Lat_decimal']=np.nan
            df['Lon_decimal']=np.nan
            df['lat_min'][v]=(df.Latitude[v]/100-(df.Latitude[v]/100).astype(int))*100
            df['lon_min'][v]=(df.Longitude[v]/100-(df.Longitude[v]/100).astype(int))*100
            df['Lat_decimal'][v]=(df.Latitude[v]/100).astype(int)+df['lat_min'][v]/60
            df['Lon_decimal'][v]=(df.Longitude[v]/100).astype(int)+df['lon_min'][v]/60
            
            # df['Lat_decimal']
            if len(df['Lat_decimal'][df['Lat_decimal']<60])>0:
                plt.close()
                plt.plot(df['Lat_decimal'])
                plt.title(meta.alt_name[i])
                plt.show()
                print('dropping test data')
                df.drop(df['Lat_decimal'][df['Lat_decimal']<60].index, inplace=True)
                df.reset_index(drop=True, inplace=True)
                
            # latest_position
            # reject if more than 3 x std from mean
            v=abs(df['Altitude']-np.nanmean(df['Altitude']))>(3*np.nanstd(df['Altitude']))
            df['Altitude'][v]=np.nan
            # plt.plot(df['Altitude'])
            v=np.isfinite(df['Altitude'])
            df['elev']=np.nan
            if sum(v)>0:
                b, m = polyfit(df['dec_year'][v], df['Altitude'][v], 1)
                df['elev']=df['dec_year']*m+b-1.4
            # plt.plot(abs(df['Altitude']-temp))
            
            # print(df['Lat_decimal'],df['lat_min'])
   

            df.index = pd.to_datetime(df.date)
            
            N=len(df)
            if N>0:
                cols=df.columns
                k=1
                

                plt.close()
                n_rows=3
                fig, ax = plt.subplots(n_rows,1,figsize=(10,14))
                cc=0

                ax[cc].set_title(meta.alt_name[i]+' GEUS AWS transmissions June 2021 until '+str(df.date[-1]))
                # ax[cc].plot(df.airpressureminus1000,'.',label='airpressureminus1000')
                # ax[cc].get_xaxis().set_visible(False)
                # ax[cc].legend()
                # ax[cc].set_xlim(t0,t1)
                # cc+=1

                ax[cc].plot(df.temperature,'.',label='temperature')
                # ax[cc].get_xaxis().set_visible(False)
                ax[cc].legend()
                ax[cc].set_xlim(t0,t1)
                cc+=1

                # ax[cc].plot(df.relativehumidity,'.',label='relativehumidity')
                # ax[cc].get_xaxis().set_visible(False)
                # ax[cc].legend()
                # ax[cc].set_xlim(t0,t1)
                # cc+=1

                # ax[cc].plot(df.windspeed,'.',label='windspeed')
                # ax[cc].get_xaxis().set_visible(False)
                # ax[cc].legend()
                # ax[cc].set_xlim(t0,t1)
                # cc+=1

                # ax[cc].plot(df.winddirection,'.',label='winddirection')
                # ax[cc].get_xaxis().set_visible(False)
                # ax[cc].legend()
                # ax[cc].set_xlim(t0,t1)
                # cc+=1

                v=((np.isfinite(df.temperature.values))&(np.isfinite(df.Battery.values)))
                x=df.temperature.values[v]
                
                y=df.Battery.values[v]
                
                ax[cc].plot(x,y,'.',label='t vs batt')
                ax[cc].legend()
                # ax[cc].set_xlim(t0,t1)
                cc+=1                
                
                ax[cc].plot(df.Battery,'.',label='Battery')
                ax[cc].set_xlim(t0,t1)
                ax[cc].legend()
                
                plt.setp(ax[cc].xaxis.get_majorticklabels(), rotation=90,ha='center' )
                ax[cc].xaxis.set_major_formatter(mdates.DateFormatter('%Y %b %d'))
                
                ly='p'
                if ly == 'p':
                    fig_path='/Users/jason/Dropbox/AWS/GCNET/Figs/GCN_GEUS_bat_v_t/'
                    os.system('mkdir -p '+fig_path)
                    plt.savefig(fig_path+meta.alt_name[i]+'.png', bbox_inches='tight', dpi=72)
                
                
                # else:
                #     df2.to_csv(opath+meta.alt_name[i]+'.csv')
            
    
                
            # plt.close()
            # n_rows=2
            # fig, ax = plt.subplots(n_rows,1,figsize=(12,14))
            # cc=0
            # plt.subplots_adjust(wspace=0, hspace=0)

            # z1=df.SR_B[0]-df.SR_B
            # z1[z1>1.5]=np.nan
            # z1[z1<-0.1]=np.nan
            # z2=df.SR_A[0]-df.SR_A
            # z2[z2>1.7]=np.nan
            # z2[z2<-0.1]=np.nan

            # ax[cc].set_title(meta.alt_name[i]+' AWS transmissions until '+str(df.date[-1].strftime('%Y %b %d %HUTC')))
            # ax[cc].plot(z1,'.',label='SR-50 on stake')
            # ax[cc].get_xaxis().set_visible(False)
            # ax[cc].legend()
            # ax[cc].set_xlim(t0,t1)
            # cc+=1

            # ax[cc].plot(z2,'.',label='SR-50 on AWS')
            # ax[cc].set_xlim(t0,t1)
            # ax[cc].legend()
            
            # plt.setp(ax[cc].xaxis.get_majorticklabels(), rotation=90,ha='center' )
            # ax[cc].xaxis.set_major_formatter(mdates.DateFormatter('%Y %b %d'))
            
            # ly='x'
            # if ly == 'p':
            #     plt.savefig(opath+meta.alt_name[i]+'.png', bbox_inches='tight', dpi=300)
