#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 08:20:34 2022

@author: jason
"""

from datetime import datetime
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from numpy.polynomial.polynomial import polyfit

sites=[
# "gits",
"humboldt",
"petermann",
"tunu_n",
# "swisscamp_10m_tower",
# "swisscamp",
# "crawfordpoint",
"nasa_u",
"summit",
"dye2",
# "jar1",
# "saddle",
# "southdome",
"nasa_east",
# "nasa_southeast",
"neem",
"east_grip"
]


# params=[swin, swin_maximum, swout, swout_minimum, netrad, netrad_maximum, airtemp1, airtemp1_maximum, airtemp1_minimum, airtemp2, airtemp2_maximum, airtemp2_minimum, airtemp_cs500air1, airtemp_cs500air2, rh1, rh2, windspeed1, windspeed_u1_maximum, windspeed_u1_stdev,windspeed2, windspeed_u2_maximum, windspeed_u2_stdev, winddir1, winddir2, pressure, sh1, sh2, battvolt, reftemp
# ]
api_path='https://www.envidat.ch/data-api/gcnet/csv/'

params='airtemp1,airtemp2,airtemp_cs500air1,airtemp_cs500air2,rh1,rh2,windspeed1,windspeed2,winddir1,winddir2,pressure,battvolt,sh1,sh2'
date_range='2021-12-01/2022-04-19'

for st,site in enumerate(sites):
    # if st>=0:
    # if site=='tunu_n':
    if site=='dye2':
    # if site=='summit':
        print(site)
        tmpfile='/Users/jason/Dropbox/AWS/CARRA-TU_GEUS/raw/Envidat/'+site+'.csv'
        msg='curl '+api_path+site+'/'+params+'/end/-999/'+date_range+'/ > '+tmpfile
        msg='curl https://www.envidat.ch/data-api/gcnet/csv/'+site+'/'+params+'/end/-999/'+date_range+'/ > '+tmpfile
        print(msg)
        os.system(msg)
        # os.system('open '+tmpfile)

#%%

meta=pd.read_csv('/Users/jason/Dropbox/AWS/GCNET/ancillary/GC-Net_info_incl_1999.csv')
print(meta.columns)
#%%


sites2=[
# "gits",
"HUM",
"PET",
"TUN",
# "swisscamp_10m_tower",
# "swisscamp",
# "crawfordpoint",
"NAU",
"SUM",
"DY2",
# "jar1",
# "saddle",
# "southdome",
"NAE",
# "nasa_southeast",
"NEM",
"EGP"
]

cols_requested=['year','day','month','hour','airpressureminus1000','temperature','relativehumidity','windspeed','winddirection','Lat_decimal','Lon_decimal','elev']

for st,site in enumerate(sites):
    # if st>=0:
    # if site=='tunu_n':
    # if site=='summit':
    print(st,site,sites2[st])
    if sites2[st]=='DY2':
        v_meta=np.where(sites2[st]==meta.name) ; v_meta=v_meta[0][0]
        print(site)
        df=pd.read_csv('/Users/jason/Dropbox/AWS/CARRA-TU_GEUS/raw/Envidat/'+site+'.csv')
        df = df.rename({'battvolt': 'Battery', 'pressure': 'airpressureminus1000'}, axis=1)

        cols=df.columns
        print(df.columns)
        for kk,col in enumerate(cols):
            if kk>0:
                df[col] = pd.to_numeric(df[col])
                df[col][df[col]<-990]=np.nan

        df["date"]=pd.to_datetime(df.iloc[:,0])

        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['hour'] = df['date'].dt.hour
        
        df["time"]=pd.to_datetime(df[['year', 'month', 'day', 'hour']])
        
        df.index = pd.to_datetime(df.time)

        df['airpressureminus1000_raw']=df.airpressureminus1000
        df['airpressureminus1000_raw']=df.airpressureminus1000
        df['airtemp1_raw']=df.airtemp1
        df['airtemp2_raw']=df.airtemp2

        df['rh1_raw']=df.rh1
        df['rh2_raw']=df.rh2
        
        if sites2[st]=='DY2':
            df = df.rename({'windspeed1': 'temp1', 'windspeed2': 'temp2'}, axis=1)
            df = df.rename({'temp1': 'windspeed2', 'temp2': 'windspeed1'}, axis=1)
            df.airtemp1[( (df.time>datetime(2021,12,1)) & (df.time<datetime(2021,12,31)) & (df.airtemp1>0) )]=np.nan
            df.airtemp2[( (df.time>datetime(2021,12,1)) & (df.time<datetime(2021,12,31)) & (df.airtemp2>0) )]=np.nan
            df.Battery[( (df.time>datetime(2021,12,1)) & (df.time<datetime(2021,12,31)) & (df.Battery>16) )]=np.nan

            df.rh1[df.rh1<30]=np.nan
            df.rh2[:]=np.nan
            
            df.airpressureminus1000[( (df.time>datetime(2022,3,20)) & (df.time<datetime(2022,4,15)) & (df.airpressureminus1000<960) )]=np.nan

        if sites2[st]=='NAE':
            df.airtemp1[( (df.time>datetime(2021,12,1)) & (df.time<datetime(2031,12,31)) & (df.airtemp1>0) )]=np.nan
            df.airtemp2[( (df.time>datetime(2021,12,1)) & (df.time<datetime(2031,12,31)) & (df.airtemp2>0) )]=np.nan

            df.rh1[( (df.time>datetime(2021,12,1)) & (df.time<datetime(2031,12,31)) & (df.rh1<45) )]=np.nan
            df.rh2[( (df.time>datetime(2021,12,1)) & (df.time<datetime(2031,12,31)) & (df.rh2<45) )]=np.nan

            df.Battery[( (df.time>datetime(2021,12,1)) & (df.time<datetime(2031,12,31)) & (df.Battery>16) )]=np.nan

        #,header=none,names=cols)
        #'/Users/jason/Dropbox/AWS/CARRA-TU_GEUS_AWS/output/'+site+'.csv'

        df['winddirection']=df[['winddir1', 'winddir2']].mean(axis=1)

        df['temperature']=df[['airtemp1', 'airtemp2']].mean(axis=1)
        if sites2[st]!='DY2':
            df['relativehumidity']=df[['rh1', 'rh2']].mean(axis=1)
        else:
            df['relativehumidity']=df.rh1
        df['windspeed']=df[['windspeed1', 'windspeed2']].mean(axis=1)        

        t0=datetime(2021, 12, 1) ; t1=datetime(2022, 4, 20)
        
        df['Lat_decimal']=np.nan ; df['Lat_decimal'][:]=meta.lat[v_meta]
        df['Lon_decimal']=np.nan ; df['Lon_decimal'][:]=meta.lon[v_meta]
        df['elev']=np.nan ; df['elev'][:]=meta.elev[v_meta]

        # cols=df.columns
        # k=1
        # var=df[cols[k]]
        wo=1
        if wo:
            df2=df[cols_requested]
            df2.index = df2.index.date
            df2.index = pd.to_datetime(df.date)
            df2.airpressureminus1000-=1000
            if len(df)>0:
                formats = {'airpressureminus1000': '{:.1f}','windspeed': '{:.1f}','relativehumidity': '{:.1f}','temperature': '{:.1f}','winddirection': '{:.1f}','relativehumidity': '{:.1f}','Lat_decimal': '{:.5f}','Lon_decimal': '{:.5f}','elev': '{:.2f}'}
            for col, f in formats.items():
                df2[col] = df2[col].map(lambda x: f.format(x))
            df2.to_csv('./AWS_data_for_CARRA-TU/GC-Net_Envidat/'+sites2[st]+'.csv')
                

        plt.close()
        n_rows=6
        fig, ax = plt.subplots(n_rows,1,figsize=(10,14))
        cc=0

        ax[cc].set_title(sites2[st]+' WSL AWS transmissions '+str(df.date[0].strftime('%Y %b %d %H'))+' until '+str(df.date[-1].strftime('%Y %b %d %H')))
        ax[cc].plot(df.airpressureminus1000_raw,'.r',label='pressure rejected')
        ax[cc].plot(df.airpressureminus1000,'.',label='pressure')
        ax[cc].get_xaxis().set_visible(False)
        ax[cc].legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax[cc].set_xlim(t0,t1)
        cc+=1

        ax[cc].plot(df.airtemp1,'.',label='airtemp1')
        ax[cc].plot(df.airtemp2,'.',label='airtemp2')
        ax[cc].plot(df.temperature,'.',label='temperature')
        ax[cc].get_xaxis().set_visible(False)
        ax[cc].legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax[cc].set_xlim(t0,t1)
        cc+=1

        ax[cc].plot(df.rh1_raw,'.r',label='rh1 rejected')
        ax[cc].plot(df.rh2_raw,'.k',label='rh2 rejected')

        ax[cc].plot(df.rh1,'.',label='rh1')
        ax[cc].plot(df.rh2,'.',label='rh2')
        ax[cc].plot(df.relativehumidity,'.',label='relativehumidity')
        ax[cc].get_xaxis().set_visible(False)
        ax[cc].legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax[cc].set_xlim(t0,t1)
        cc+=1

        ax[cc].plot(df.windspeed1,'.',label='windspeed1')
        ax[cc].plot(df.windspeed2,'.',label='windspeed2')
        ax[cc].plot(df.windspeed,'.',label='windspeed')
        ax[cc].get_xaxis().set_visible(False)
        ax[cc].legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax[cc].set_xlim(t0,t1)
        cc+=1

        ax[cc].plot(df.winddir1,'.',label='winddirection 1')
        ax[cc].plot(df.winddir2,'.',label='winddirection 2')
        ax[cc].plot(df.winddirection,'.',label='winddirection')
        ax[cc].get_xaxis().set_visible(False)
        ax[cc].legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax[cc].set_xlim(t0,t1)
        cc+=1

        # ax[cc].plot(df.elev,'.',label='elev')
        # ax[cc].get_xaxis().set_visible(False)
        # ax[cc].legend()
        # ax[cc].set_xlim(t0,t1)
        # cc+=1                
        
        ax[cc].plot(df.Battery,'.',label='Battery')
        ax[cc].set_xlim(t0,t1)
        ax[cc].legend(loc='center left', bbox_to_anchor=(1, 0.5))
        
        plt.setp(ax[cc].xaxis.get_majorticklabels(), rotation=90,ha='center' )
        ax[cc].xaxis.set_major_formatter(mdates.DateFormatter('%Y %b %d'))
        
        ly='p'
        if ly == 'p':
            fig_path='./AWS_data_for_CARRA-TU/GC-Net_Envidat/Figs/'
            os.system('mkdir -p '+fig_path)
            plt.savefig(fig_path+sites2[st]+'.png', bbox_inches='tight', dpi=72)
    # else:
    #     df2.to_csv('/Users/jason/Dropbox/AWS/CARRA-TU_GEUS_AWS/output/Envidat/'+site2[st]+'.csv')