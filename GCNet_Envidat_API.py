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
"swisscamp",
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
    # if site=='summit':
        print(site)
        tmpfile='/tmp/'+site+'.csv'
        msg='curl '+api_path+site+'/'+params+'/end/-999/'+date_range+'/ > '+tmpfile
        msg='curl https://www.envidat.ch/data-api/gcnet/csv/'+site+'/'+params+'/end/-999/'+date_range+'/ > '+tmpfile
        print(msg)
        os.system(msg)
        # os.system('open '+tmpfile)

#%%
        df=pd.read_csv(tmpfile)
        df = df.rename({'battvolt': 'Battery', 'pressure': 'airpressureminus1000'}, axis=1)  # new method
        cols=df.columns
        for kk,col in enumerate(cols):
            if kk>0:
                df[col] = pd.to_numeric(df[col])
                df[col][df[col]<-990]=np.nan

        #,header=none,names=cols)
        #'/Users/jason/Dropbox/AWS/CARRA-TU_GEUS_AWS/output/'+site+'.csv'
        df['temperature']=df[['airtemp1', 'airtemp2']].mean(axis=1)
        
        df["date"]=pd.to_datetime(df.iloc[:,0])
        df.index = pd.to_datetime(df.date)

        t0=datetime(2021, 12, 1) ; t1=datetime(2022, 4, 20)
         
        wo=1
        
        if wo:
            # cols=df.columns
            # k=1
            # var=df[cols[k]]
            # df2=df[cols_requested]
            # df2.index = df2.index.date
            # df2.index = pd.to_datetime(df.date)
            if len(df)>0:
                # formats = {'airpressureminus1000': '{:.1f}','windspeed': '{:.1f}','relativehumidity': '{:.1f}','winddirection': '{:.1f}','relativehumidity': '{:.1f}','Lat_decimal': '{:.5f}','Lon_decimal': '{:.5f}','elev': '{:.2f}'}
                # for col, f in formats.items():
                #     df2[col] = df2[col].map(lambda x: f.format(x))
                # opath='/Users/jason/Dropbox/AWS/_merged/CARRA_TU_GEUS_xmit_2021_to_present/'
                # df2.to_csv(opath+site+'.csv')
                

                plt.close()
                n_rows=3
                fig, ax = plt.subplots(n_rows,1,figsize=(10,14))
                cc=0

                ax[cc].set_title(site+' WSL AWS transmissions June 2021 until '+str(df.date[-1]))
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

                # ax[cc].plot(df.elev,'.',label='elev')
                # ax[cc].get_xaxis().set_visible(False)
                # ax[cc].legend()
                # ax[cc].set_xlim(t0,t1)
                # cc+=1                
                
                ax[cc].plot(df.Battery,'.',label='Battery')
                ax[cc].set_xlim(t0,t1)
                ax[cc].legend()
                
                plt.setp(ax[cc].xaxis.get_majorticklabels(), rotation=90,ha='center' )
                ax[cc].xaxis.set_major_formatter(mdates.DateFormatter('%Y %b %d'))
                
                ly='x'
                if ly == 'p':
                    plt.savefig(opath+site2[st]+'.png', bbox_inches='tight', dpi=300)
            # else:
            #     df2.to_csv('/Users/jason/Dropbox/AWS/CARRA-TU_GEUS_AWS/output/Envidat/'+site2[st]+'.csv')