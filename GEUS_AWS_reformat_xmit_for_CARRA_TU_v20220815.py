#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 13:38:41 2022
# %matplotlib inline
# %matplotlib osx

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

th=2 # line thickness
formatx='{x:,.3f}' ; fs=16
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

# -------------------------------- dates covered by this delivery
date0='2021-06-01'; date1='2022-04-01'
date0='2022-04-01'; date1='2022-05-31'
date0='2022-05-31'; date1='2022-07-31'

do_plot=1

# ----------------------------------------------------------- adjuster routine
# jason box
# procedure with different filter functions
# counts cases rejected by filters
# outputs filter to a format that is compatible for GC-Net-level-1-data-processing/GC-Net-level-1-data-processing.py
def adjuster(site,df,var_list,y0,m0,d0,func,y1,m1,d1,comment,val):
    
    tstring='%Y-%m-%dT%H:%M:%S'+'+00:00'

    df_out = df.copy()
    
    t0=datetime(y0,m0,d0)
    t1=datetime(y1,m1,d1)

    # two variables required for abs_diff
    if func == 'abs_diff': 
        tmp0=df_out.loc[t0:t1,var_list[0]].values
        tmp1=df_out.loc[t0:t1,var_list[1]].values
        tmp = df_out.loc[t0:t1,var_list[1]].values-df_out.loc[t0:t1,var_list[0]].values
        count=sum(abs(tmp)>val)
        tmp0[abs(tmp)>val] = np.nan
        tmp1[abs(tmp)>val] = np.nan
        df_out.loc[t0:t1,var_list[0]] = tmp0
        df_out.loc[t0:t1,var_list[1]] = tmp1

    if func == 'abs_diff_del_instrument_2': 
        tmp0=df_out.loc[t0:t1,var_list[0]].values
        tmp1=df_out.loc[t0:t1,var_list[1]].values
        tmp = df_out.loc[t0:t1,var_list[1]].values-df_out.loc[t0:t1,var_list[0]].values
        count=sum(abs(tmp)>val)
        # tmp0[abs(tmp)>val] = np.nan
        tmp1[abs(tmp)>val] = np.nan
        # df_out.loc[t0:t1,var_list[0]] = tmp0
        df_out.loc[t0:t1,var_list[1]] = tmp1
 
    if func == 'swap':
            val_var = df_out.loc[t0:t1,var_list[0]].values.copy()
            val_var2 = df_out.loc[t0:t1,var_list[1]].values.copy()
            df_out.loc[t0:t1,var_list[1]] = val_var
            df_out.loc[t0:t1,var_list[0]] = val_var2
            count=len(df_out.loc[t0:t1,var_list[1]])

    for var in var_list:
        # set to nan stuck values
        if func == 'nan_constant': 
            tmp = df_out.loc[t0:t1,var]
            count=sum(tmp.diff()==0)
            tmp[tmp.diff()==0]=np.nan
            df_out.loc[t0:t1,var] = tmp

        if func == 'min_filter': 
            tmp = df_out.loc[t0:t1,var].values
            count=sum(tmp<val)
            tmp[tmp<val] = np.nan
            df_out.loc[t0:t1,var] = tmp

        if func == 'nan_var': 
            tmp = df_out.loc[t0:t1,var].values
            count=len(tmp)
            tmp[:] = np.nan
            df_out.loc[t0:t1,var] = tmp
            
        if func == 'max_filter': 
            tmp = df_out.loc[t0:t1,var].values
            count=sum(tmp>val)
            tmp[tmp>val] = np.nan
            df_out.loc[t0:t1,var] = tmp

        # if 'swap_with_' in func: 
        #     var2 = func[10:]
        #     val_var = df_out.loc[t0:t1,var].values.copy()
        #     val_var2 = df_out.loc[t0:t1,var2].values.copy()
        #     df_out.loc[t0:t1,var2] = val_var
        #     df_out.loc[t0:t1,var] = val_var2

        msg=datetime(y0,m0,d0).strftime(tstring)+\
        ','+datetime(y1,m1,d1).strftime(tstring)+\
        ','+var+','+func+','+str(val)+','+comment+','+str(count)
        # print(msg)

        # dfx=pd.read_csv('/Users/jason/Dropbox/AWS/GCNET/GC-Net-level-1-data-processing/metadata/adjustments/'+site+'.csv')
        # print(dfx)

        wo=1
        if wo:
            opath_adjustments='./metadata/adjustments/'+site+'/'
            os.system('mkdir -p '+opath_adjustments)
            out_fn=opath_adjustments+var+'_'+func+'_'+datetime(y0,m0,d0).strftime('%Y-%m-%d')+'_'+datetime(y1,m1,d1).strftime('%Y-%m-%d')+'.csv'
            out_concept=open(out_fn,'w')
            out_concept.write('t0,t1,variable,adjust_function,adjust_value,comment,count\n')
            out_concept.write(msg)
            out_concept.close()

    return(df_out)
    # ----------------------------------------------------------- end adjuster routine
today = date.today()
versionx= today.strftime('%Y-%m-%d')

os.chdir(base_path)

meta_file='./metadata/IMEI_numbers_station_2021-10-27.xlsx'
# os.system('open '+meta_file)
meta = pd.read_excel(meta_file)
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

# actual names?
#  cols=['time','counter','Pressure_L','Pressure_U','Asp_temp_L','Asp_temp_U','Humidity_L','Humidity_U','WindSpeed_L','WindDirection_L','WindSpeed_U','WindDirection_U','SWD','SWU','LW Downward','LW Upward','TemperatureRadSensor','SR_A','SR_B','T_firn_1','T_firn_2','T_firn_3','T_firn_4','T_firn_5','T_firn_6','T_firn_7','T_firn_8','T_firn_9','T_firn_10','T_firn_11','Roll','Pitch','Heading','Rain_amount_L','Rain_amount_U','counterx','Latitude','Longitude','Altitude','ss','Giodal','GeoUnit','Battery','NumberSatellites','HDOP','FanCurrent_L','FanCurrent_U','Quality','LoggerTemp']

# cols=['SR_A','SR_B','IceHeight','TemperatureIce1','TemperatureIce2','TemperatureIce3','TemperatureIce4','TemperatureIce5','TemperatureIce6','TemperatureIce7','TemperatureIce8','TiltX','TiltY','TimeGPS','Latitude','Longitude','Altitude','HDOP','currents','Battery','AirPressureMinus1000','Temperature','RelativeHumidity','WindSpeed','WindDirection','InstantDataTerminator']

# cols_requested=['year','day','month','hour','airpressureminus1000','temperature','relativehumidity','windspeed','winddirection','Lat_decimal','Lon_decimal','elev']
cols_requested=['airpressureminus1000','temperature','relativehumidity','windspeed','winddirection','Lat_decimal','Lon_decimal','elev']

# time range to consider
t0=datetime(2020, 8, 1) ; t1=datetime(2021, 7, 8)
t0=datetime(2021,12, 1) ; t1=datetime(2022, 4, 1)
t0=datetime(2022,5, 31) ; t1=datetime(2022, 7, 31)

ly='p'
wo=1

for i,name in enumerate(names):
    if i>=0:
        # if meta.alt_name[i]=='SWC':
        # if meta.alt_name[i]=='SDM':
        # if ((meta.network[i]=='g')or(meta.network[i]=='p')):
        # if meta.network[i]=='g':
        if meta.alt_name[i]=='NSE':
            print()
            print(i,names[i],meta.alt_name[i])
            
            site=meta.alt_name[i]
            # asas
            # if names[i][0:5]=='GCNET':
            cols=['timestamp','seconds_since_1990','airpressure','temperature','temperature2','relativehumidity','windspeed','winddirection',
                  'shortwaveradiationin','shortwaveradiationout','longwaveradiationin','longwaveradiationout','temperatureradsensor','SR_A','SR_B','iceheight',
                  'temperatureice1m','temperatureice2m','temperatureice3m','temperatureice4m','temperatureice5m','temperatureice6m','temperatureice7m','temperatureice10m',
                  'tiltx','tilty','timegps','Latitude','Longitude','Altitude','hdop',
                  'currents','Battery','airpressureminus1000x','temperature0','relativehumidity0','windspeed0','winddirection0']
            skip=6
            if names[i][0:5]=='GCNET':
                skip=10
                # cols=['timestamp','seconds_since_1990','airpressureminus1000','Pressure_U','temperature','Asp_temp_U','relativehumidity','Humidity_U','windspeed','winddirection','WindSpeed_U','WindDirection_U','SWD','SWU','LW Downward','LW Upward','TemperatureRadSensor','SR_A','SR_B','T_firn_1','T_firn_2','T_firn_3','T_firn_4','T_firn_5','T_firn_6','T_firn_7','T_firn_8','T_firn_9','T_firn_10','T_firn_11','Roll','Pitch','Heading','Rain_amount_L','Rain_amount_U','counterx','Latitude','Longitude','Altitude','ss','Giodal','GeoUnit','Battery','NumberSatellites','HDOP','FanCurrent_L','FanCurrent_U','Quality','LoggerTemp']
                cols=['timestamp','seconds_since_1990','airpressure','Pressure_U','temperature','Asp_temp_U','relativehumidity','Humidity_U','windspeed','winddirection','WindSpeed_U','WindDirection_U','SWD','SWU','LW Downward','LW Upward','TemperatureRadSensor','SR_A','SR_B','T_firn_1','T_firn_2','T_firn_3','T_firn_4','T_firn_5','T_firn_6','T_firn_7','T_firn_8','T_firn_9','T_firn_10','T_firn_11','Roll','Pitch','Heading','Rain_amount_L','Rain_amount_U','counterx','Latitude','Longitude','Altitude','ss','Giodal','GeoUnit','Battery','NumberSatellites','HDOP','FanCurrent_L','FanCurrent_U','Quality','LoggerTemp']
            if meta.alt_name[i]=='SWC' or meta.alt_name[i]=='JAR':
                skip=4
                # cols=['time','seconds_since_1990','airpressureminus1000','temperature','relativehumidity','windspeed','winddirection',
                cols=['time','seconds_since_1990','airpressure','temperature','relativehumidity','windspeed','winddirection',
                      'SWU','SWD',
                     'LW Downward','LW Upward','TemperatureRadSensor','SR_A','SR_B','solar?',
                     'thermistorstring_1','thermistorstring_2','thermistorstring_3','thermistorstring_4','thermistorstring_5','thermistorstring_6','thermistorstring_7','thermistorstring_8',
                     'roll','pitch','heading','Rain_amount_L','gpstime','Latitude','Longitude','Altitude','giodal','geounit?',
                     'Battery','?1','asp_temp_u','humidity_u','##','##2','##3']

            print(names[i])
            fn='/Users/jason/Dropbox/AWS/aws_data/AWS_'+str(meta.IMEI[i])+'.txt'
            print(fn)
            
            # filtering of double commas
            fn2="/tmp/"+str(meta.IMEI[i])+".txt"
            msg="sed 's/,,/,/g' "+fn+" > "+fn2
            os.system(msg)
            fn=fn2    
            df=pd.read_csv(fn,header=None,names=cols,skiprows=skip)
            
            # filtering of how NaN is spelled
            df[df=="NAN"]=np.nan
            df[df=="nan"]=np.nan
            
            # date
            df["date"]=pd.to_datetime(df.iloc[:,0])
            df['year'] = df['date'].dt.year

            df['month'] = df['date'].dt.month
            df['day'] = df['date'].dt.day
            df['hour'] = df['date'].dt.hour
            df['doy'] = df['date'].dt.dayofyear
            df['days_per_year'] = 365
            df['days_per_year'][df['year']==2020]=366
            # decimal year used in height regression
            df['dec_year'] = df['year']+df['date'].dt.dayofyear/df['days_per_year']

            drop=1
            if drop:
                # if meta.alt_name[i]!='JAR':
                df.drop(df[df.date<date0].index, inplace=True)
                df.reset_index(drop=True, inplace=True)
                df.drop(df[df.date>=date1].index, inplace=True)
                df.reset_index(drop=True, inplace=True)
            print(df)
            print(df.columns)

                # t0=datetime(2022, 4, 1) ; t1=datetime(2022, 5, 31)

            # dfx=df.copy()
            # dfx=dfx.drop(dfx.columns[1:3], axis=1)
            # dfx.to_csv('/Users/jason/Dropbox/AWS/GCNET/GCNv2_xmit/output/'+names[i]+'.csv')

            # plt.plot(df['dec_year'])
            ##%%

            for kk,col in enumerate(cols):
                if kk>0:
                    df[col] = pd.to_numeric(df[col])
            # df['temperature'] = pd.to_numeric(df['temperature'])
            # df['relativehumidity'] = pd.to_numeric(df['relativehumidity'])
            # df['airpressureminus1000'] = pd.to_numeric(df['airpressureminus1000'])
            df['relativehumidity'][df['relativehumidity']>105]=np.nan
            df['relativehumidity'][df['relativehumidity']<30]=np.nan
            df['airpressureminus1000']=df.airpressure-1000
            # df['airpressureminus1000']=df.airpressure
            df['airpressureminus1000'][df['airpressureminus1000']>1000]=np.nan
            df.winddirection[df.winddirection.diff()==0]=np.nan
            df.windspeed[df.windspeed.diff()==0]=np.nan

            # df['relativehumidity'][((df['relativehumidity'].diff()==0)&(df['relativehumidity']<30))]=np.nan

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

            df['Lat_decimal']=np.nan
            df['Lon_decimal']=np.nan
            
            v=np.where(np.isfinite(df.Latitude)) ; v=v[0]
            df['lat_min']=np.nan
            df['lon_min']=np.nan
            df['Lat_decimal']=np.nan
            df['Lon_decimal']=np.nan
            df['lat_min'][v]=(df.Latitude[v]/100-(df.Latitude[v]/100).astype(int))*100
            df['lon_min'][v]=(df.Longitude[v]/100-(df.Longitude[v]/100).astype(int))*100
            df['Lat_decimal'][v]=(df.Latitude[v]/100).astype(int)+df['lat_min'][v]/60
            df['Lon_decimal'][v]=(df.Longitude[v]/100).astype(int)+df['lon_min'][v]/60

            if len(df['Lat_decimal'][df['Lat_decimal']<60])>0:

                # plt.close()
                # plt.plot(df['Lat_decimal'])
                # plt.title(meta.alt_name[i])
                # plt.show()
                # print('dropping test data')
                df.drop(df['Lat_decimal'][df['Lat_decimal']<60].index, inplace=True)
                df.reset_index(drop=True, inplace=True)
                
            # reject if more than 3 x std from mean
            v=abs(df['Altitude']-np.nanmean(df['Altitude']))>(2*np.nanstd(df['Altitude']))
            df['Altitude'][v]=np.nan
            # plt.plot(df['Altitude'])
            v=np.isfinite(df['Altitude'])
            df['elev']=np.nan
            if sum(v)>0:
                x=df['dec_year'][v].values ; y=df['Altitude'][v].values
                b, m = polyfit(x,y, 1)
                df['elev']=df['dec_year']*m+b-1.4
                
            # plt.plot(abs(df['Altitude']-temp))
            # print(df['Lat_decimal'],df['lat_min'])
   

            df.index = pd.to_datetime(df.date)

#             print(df['Lat_decimal'])
# #%%

            # print(df.temperature)
#             plt.plot(df.windspeed)
# #%%

            if site=='NSE':
                df=adjuster(site,df,['airpressureminus1000'],2022,4,1,'min_filter',2022,5,31,'xmit outlier?',-350)
                df=adjuster(site,df,['airpressureminus1000'],2022,5,31,'min_filter',2022,7,31,'xmit outlier?',-1300)
                df=adjuster(site,df,['airpressureminus1000'],2022,5,31,'max_filter',2022,7,31,'xmit outlier?',-1100)
                df=adjuster(site,df,['temperature'],2022,4,1,'min_filter',2022,7,31,'xmit outlier?',-90)
                df=adjuster(site,df,['temperature'],2022,4,1,'max_filter',2022,7,31,'xmit outlier?',9)
                
            if site=='QAS_M':
                # df=adjuster(site,df,['TA1','TA2'],2021,6,1,'max_filter',2022,4,1,'xmit outlier?',2)
                df=adjuster(site,df,['temperature','relativehumidity'],2022,4,1,'nan_var',2022,5,31,'instrument burial!?',0)

            if site=='QAS_U':
                # df=adjuster(site,df,['TA1','TA2'],2021,6,1,'max_filter',2022,4,1,'xmit outlier?',2)
                df=adjuster(site,df,['temperature','relativehumidity'],2022,4,1,'nan_var',2022,5,28,'instrument burial!?',0)
                # df=adjuster(site,df,['relativehumidity'],2022,4,1,'nan_var',2022,5,28,'instrument burial!?',0)

            if site=='TAS_A':
                df=adjuster(site,df,['temperature','relativehumidity','windspeed','winddirection'],2022,4,1,'nan_var',2022,6,24,'instrument burial!?',0)
                df=adjuster(site,df,['winddirection'],2022,6,30,'nan_var',2022,7,31,'instrument fail!?',0)
                df=adjuster(site,df,['windspeed'],2022,7,1,'nan_var',2022,7,31,'instrument fail!?',0)
                
            if site=='NUK_U':
                df=adjuster(site,df,['windspeed'],2022,4,1,'nan_var',2022,5,31,'propellor not secure?',0)

            if do_plot:

                N=len(df)
                if N>0:

                    opath='./AWS_data_for_CARRA-TU/data_range_'+date0+'_to_'+date1+'/PROMICE_GC-Net_GEUS/'
                    os.system('mkdir -p '+opath)
    
                    fs=20
                    plt.close()
                    plt.rcParams["font.size"] = fs
                    plt.scatter(x,y,color='grey')
                    plt.plot((x[0],x[-1]),(x[0]*m+b,x[-1]*m+b),linewidth=th*3,color='r')
                    plt.ylabel('elevation, m')
                    plt.xlabel('year')
                    plt.title(meta.alt_name[i])
                    
                    if ly == 'p':
                        fig_path=opath+'Figs/'
                        os.system('mkdir -p '+fig_path)
                        fig_path=opath+'Figs/elev/'
                        os.system('mkdir -p '+fig_path)
                        plt.savefig(fig_path+meta.alt_name[i]+'_elev.png', bbox_inches='tight', dpi=72)

                    cols=df.columns
                    k=1
                    var=df[cols[k]]
                    df2=df[cols_requested]

                    # df2.iloc[j,4:9]
                    # bads=[]
                    # for j in range(N):
                    #     if sum(np.isnan(df2.iloc[j,4:9]))==5:
                    #         bads.append(j)
                        
                    # # bads=1
                    # df2.drop(bads, axis=0, inplace=True)

                    # df2.drop(df[((df2.airpressureminus1000 == np.nan)&(df2.temperature == np.nan)&(df2.relativehumidity == np.nan)&(df2.windspeed == np.nan)&(df2.winddirection == np.nan))].index, inplace=True)
                    # df2.drop(df2[((df2.airpressureminus1000 == np.nan)&(df2.relativehumidity == np.nan)&(df2.windspeed == np.nan)&(df2.winddirection == np.nan))].index, inplace=True)
                    # df2 = 
                    # df2[((np.isnan(df2.airpressureminus1000))&(np.isnan(df2.relativehumidity))&(np.isnan(df2.windspeed))&(np.isnan(df2.winddirection)))]
                    # ix = df2[((df2.airpressureminus1000 == np.nan)&(df2.temperature == np.nan)&(df2.relativehumidity == np.nan)&(df2.windspeed == np.nan)&(df2.winddirection == np.nan))].index
                    # ix = df2[df2.airpressureminus1000 == np.nan].index
                    # print(ix)
                    # df2.drop(ix)
                    # df2.index = df2.index.date
                    # df2.index = pd.to_datetime(df.date)
                    formats = {'airpressureminus1000': '{:.1f}','windspeed': '{:.1f}','relativehumidity': '{:.1f}','winddirection': '{:.1f}','relativehumidity': '{:.1f}','Lat_decimal': '{:.5f}','Lon_decimal': '{:.5f}','elev': '{:.2f}'}
                    for col, f in formats.items():
                        df2[col] = df2[col].map(lambda x: f.format(x))
                    os.system('mkdir -p '+opath)
                    df2.to_csv(opath+meta.alt_name[i]+'.csv')
                    
    
                    plt.close()
                    n_rows=6
                    fig, ax = plt.subplots(n_rows,1,figsize=(10,14))

                    fs=15
                    plt.rcParams["font.size"] = fs

                    cc=0
                    datex=df.date[-1].strftime('%Y %b %d %H')
                    ax[cc].set_title(site+' GEUS AWS transmissions June 2021 until '+str(datex)+'h UTC')
                    ax[cc].plot(df.airpressureminus1000,'.',label='airpressureminus1000')
                    ax[cc].get_xaxis().set_visible(False)
                    ax[cc].legend()
                    ax[cc].set_xlim(t0,t1)
                    cc+=1
    
                    ax[cc].plot(df.temperature,'.',label='temperature')
                    ax[cc].get_xaxis().set_visible(False)
                    ax[cc].legend()
                    ax[cc].set_xlim(t0,t1)
                    cc+=1
    
                    ax[cc].plot(df.relativehumidity,'.',label='relativehumidity')
                    ax[cc].get_xaxis().set_visible(False)
                    ax[cc].legend()
                    ax[cc].set_xlim(t0,t1)
                    cc+=1
    
                    ax[cc].plot(df.windspeed,'.',label='windspeed')
                    ax[cc].get_xaxis().set_visible(False)
                    ax[cc].legend()
                    ax[cc].set_xlim(t0,t1)
                    cc+=1
    
                    ax[cc].plot(df.winddirection,'.',label='winddirection')
                    ax[cc].get_xaxis().set_visible(False)
                    ax[cc].legend()
                    ax[cc].set_xlim(t0,t1)
                    cc+=1
    
                    ax[cc].plot(df.elev,'.',label='elev')
                    # ax[cc].get_xaxis().set_visible(False)
                    ax[cc].legend()
                    ax[cc].set_xlim(t0,t1)
                    # cc+=1                
                    
                    # ax[cc].plot(df.Battery,'.',label='Battery')
                    # ax[cc].set_xlim(t0,t1)
                    # ax[cc].legend()
                    
                    plt.setp(ax[cc].xaxis.get_majorticklabels(), rotation=90,ha='center' )
                    ax[cc].xaxis.set_major_formatter(mdates.DateFormatter('%Y %b %d'))
                    
                    if ly == 'p':
                        fig_path=opath+'Figs/'
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
