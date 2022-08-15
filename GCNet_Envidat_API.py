#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 08:20:34 2022

%matplotlib inline
%matplotlib osx

@author: jason
"""

from datetime import datetime
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from numpy.polynomial.polynomial import polyfit
from os.path import exists


# -------------------------------- dates covered by this delivery
date0='2021-06-01' ; date1='2022-04-01'

date0='2022-04-01' ; date1='2022-05-31'

date0='2022-05-31' ; date1='2022-07-31'

os.chdir('/Users/jason/Dropbox/AWS/CARRA-TU_GEUS/')

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


# params=[swin, swin_maximum, swout, swout_minimum, netrad, netrad_maximum, airtemp1, airtemp1_maximum, airtemp1_minimum, airtemp2, airtemp2_maximum, airtemp2_minimum, airtemp_cs500air1, airtemp_cs500air2, rh1, rh2, windspeed1, windspeed_u1_maximum, windspeed_u1_stdev,VW2, windspeed_u2_maximum, windspeed_u2_stdev, winddir1, winddir2, pressure, sh1, sh2, battvolt, reftemp
# ]
api_path='https://www.envidat.ch/data-api/gcnet/csv/'

params='airtemp1,airtemp2,airtemp_cs500air1,airtemp_cs500air2,rh1,rh2,windspeed1,windspeed2,winddir1,winddir2,pressure,battvolt,sh1,sh2'
date_range=date0+'/'+date1
date_range2=date0+'_to_'+date1

do_gather=1

if do_gather:
    for st,site in enumerate(sites):
        if st>=0:
        # if site=='tunu_n':
        # if site=='dye2':
        # if site=='summit':
        # if site=='nasa_east':
        # if site=='humboldt':
            print(site)
            opath='/Users/jason/Dropbox/AWS/CARRA-TU_GEUS/raw/Envidat_'+date_range2+'/'
            os.system('mkdir -p '+opath)
            tmpfile=opath+site+'.csv'
            msg='curl '+api_path+site+'/'+params+'/end/-999/'+date_range+'/ > '+tmpfile
            msg='curl https://www.envidat.ch/data-api/gcnet/csv/'+site+'/'+params+'/end/-999/'+date_range+'/ > '+tmpfile
            print(msg)
            os.system(msg)
            # os.system('open '+tmpfile)

inpath=opath
#%%

# created by /Users/jason/Dropbox/AWS/GCNET/GCNet_positions/GCN_positions_1999.py
meta=pd.read_csv('/Users/jason/Dropbox/AWS/GCNET/GCNet_positions/output/GC-Net_info_incl_1999.csv')
print(meta.columns)
print(meta)

# meta = pd.read_excel('./metadata/IMEI_numbers_station_2021-10-27.xlsx')

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

sites_original_convention=[
# "gits",
"HUM",
"PET",
"TUNU-N",
# "swisscamp_10m_tower",
# "swisscamp",
# "crawfordpoint",
"NAU",
"Summit",
"DYE2",
# "jar1",
# "saddle",
# "southdome",
"NAE",
# "nasa_southeast",
"NEEM",
"E-GRIP"
]



show_rejected=0
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

        wo=0
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
#    
#
#
#
#
#

plt_diagnostic=0
do_batch1=1 # 1 is for P, TA and 0 is for the others
wo=1
ly='p'

for st,site in enumerate(sites):
    # if st>=0:
    # if site=='tunu_n':
    # if site=='summit':
    print(st,site,sites2[st])
    if sites2[st]=='NAU':
    # if sites2[st]!='null' :
        
        if exists(inpath+site+'.csv'):
            meta.columns
            v_meta=np.where(sites2[st]==meta.name) ; v_meta=v_meta[0][0]
            print(site)

            df=pd.read_csv(inpath+site+'.csv')
            df = df.rename({'battvolt': 'Battery', 'pressure': 'P'}, axis=1)
            #'airtemp1', 'airtemp2', 'airtemp_cs500air1','airtemp_cs500air2'        
            if sites2[st]!='PET':
                df = df.rename({'airtemp1': 'TA1', 'airtemp2': 'TA2'}, axis=1)
            df = df.rename({'windspeed1': 'VW1', 'windspeed2': 'VW2'}, axis=1)
            if sites2[st]=='PET':
                df = df.rename({'airtemp_cs500air1' : 'TA1', 'airtemp_cs500air2' : 'TA2'}, axis=1)
            
            df = df.rename({'timestamp_iso': 'date'}, axis=1)
            
            'Lat_decimal','Lon_decimal','elev'
            # df.index = df.index.date
    
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
    
            # df["SR_"+sensor_level]*=df['tr']

            df['P_raw']=df.P
            df['TA1_raw']=df.TA1
            df['TA2_raw']=df.TA2
            df['VW1_raw']=df.VW1
            df['VW2_raw']=df.VW2
            df['winddir1_raw']=df.winddir1
            df['winddir2_raw']=df.winddir2 
            
            df['rh1_raw']=df.rh1
            df['rh2_raw']=df.rh2
    
            df=adjuster(sites_original_convention[st],df,['VW1','VW2'],2021,6,1,'nan_constant',2022,4,12,'nan stuck values',0)
            df=adjuster(sites_original_convention[st],df,['winddir1','winddir2'],2021,6,1,'nan_constant',2022,4,12,'nan stuck values',0)
            df=adjuster(sites_original_convention[st],df,['TA1','TA2'],2021,6,1,'abs_diff',2022,5,8,'xmit outlier?',2.5)
            df=adjuster(sites_original_convention[st],df,['rh1','rh2'],2021,6,1,'min_filter',2022,4,1,'xmit outlier?',35)
    
            if sites2[st]=='NEM':
                df=adjuster(sites_original_convention[st],df,['TA1','TA2'],2021,6,1,'max_filter',2022,4,1,'xmit outlier?',2)
                df=adjuster(sites_original_convention[st],df,['TA1','TA2'],2021,9,20,'nan_var',2022,4,1,'instrument burial!?',0)
                df=adjuster(sites_original_convention[st],df,['rh1','rh2'],2021,9,20,'nan_var',2022,7,31,'instrument burial!?',0)
                df=adjuster(sites_original_convention[st],df,['VW2'],2021,11,1,'nan_var',2022,5,31,'instrument burial!?',0)
                df=adjuster(sites_original_convention[st],df,['VW1'],2021,6,1,'max_filter',2022,5,31,'xmit outlier?',8)
                df=adjuster(sites_original_convention[st],df,['VW1','VW2'],2021,9,1,'max_filter',2021,10,1,'instrument burial!?',12.5)
                df=adjuster(sites_original_convention[st],df,['winddir2'],2021,10,20,'nan_var',2022,5,31,'instrument malfunction!?',0)

                df=adjuster(sites_original_convention[st],df,['rh1'],2022,5,31,'nan_var',2022,7,31,'instrument burial!?',0)
                df=adjuster(sites_original_convention[st],df,['rh2'],2022,5,31,'min_filter',2022,7,31,'xmit outlier?',40)
                df=adjuster(sites_original_convention[st],df,['VW1'],2022,6,1,'max_filter',2022,7,31,'xmit outlier?',15)

                df=adjuster(sites_original_convention[st],df,['VW2','winddir2'],2022,6,1,'nan_var',2022,7,31,'xmit outlier?',15)
                df=adjuster(sites_original_convention[st],df,['TA1'],2022,5,31,'nan_var',2022,7,31,'instrument burial!?',0)
                df=adjuster(sites_original_convention[st],df,['TA2'],2022,5,31,'max_filter',2022,7,31,'xmit outlier?',2)

            if sites2[st]=='PET':
                df=adjuster(sites_original_convention[st],df,['TA1','TA2'],2021,6,1,'max_filter',2022,4,1,'xmit outlier?',4)
                df=adjuster(sites_original_convention[st],df,['TA1','TA2'],2022,1,1,'max_filter',2022,4,1,'xmit outlier?',-10)
                df=adjuster(sites_original_convention[st],df,['TA1','TA2'],2021,6,1,'abs_diff',2022,4,1,'xmit outlier?',1.8)
                df=adjuster(sites_original_convention[st],df,['P'],2021,6,1,'max_filter',2022,4,1,'xmit outlier?',950)
                df=adjuster(sites_original_convention[st],df,['P'],2021,6,1,'max_filter',2022,4,1,'xmit outlier?',950)
                df=adjuster(sites_original_convention[st],df,['P'],2021,6,1,'max_filter',2021,7,15,'xmit outlier?',909)
                df=adjuster(sites_original_convention[st],df,['P'],2021,6,1,'min_filter',2022,4,1,'xmit outlier?',830)
                df=adjuster(sites_original_convention[st],df,['rh1','rh2'],2021,6,1,'min_filter',2022,4,1,'xmit outlier?',45)
                df=adjuster(sites_original_convention[st],df,['rh1','rh2'],2021,8,1,'min_filter',2021,9,20,'xmit outlier?',60)
                df=adjuster(sites_original_convention[st],df,['VW1','VW2'],2021,6,1,'max_filter',2022,4,1,'xmit outlier?',18)
    
            if sites2[st]=='NAU':
                df=adjuster(sites_original_convention[st],df,['TA1','TA2'],2021,6,1,'max_filter',2022,4,11,'xmit outlier?',4)
                df=adjuster(sites_original_convention[st],df,['VW1','VW2'],2021,6,1,'max_filter',2022,7,31,'xmit outlier?',20)
                df=adjuster(sites_original_convention[st],df,['P'],2021,10,11,'max_filter',2021,11,1,'xmit outlier?',785)
                df=adjuster(sites_original_convention[st],df,['winddir1','winddir2'],2022,2,1,'nan_var',2022,7,31,'instrument unwired?',0)
                df=adjuster(sites_original_convention[st],df,['P'],2022,4,8,'nan_var',2022,5,31,'instrument malfunction!?',0)
                df=adjuster(sites_original_convention[st],df,['rh1','rh2'],2022,6,1,'min_filter',2022,7,31,'xmit outlier?',55)
                df=adjuster(sites_original_convention[st],df,['TA1','TA2'],2022,5,31,'max_filter',2022,7,31,'xmit outlier?',4)
                df=adjuster(sites_original_convention[st],df,['TA1','TA2'],2022,7,23,'max_filter',2022,7,31,'xmit outlier?',0)

            if sites2[st]=='HUM':
                df=adjuster(sites_original_convention[st],df,['TA1'],2021,12,1,'min_filter',2022,1,12,'xmit outlier?',-20)
                df=adjuster(sites_original_convention[st],df,['TA1','TA2'],2021,12,1,'max_filter',2022,1,12,'xmit outlier?',10)
                df=adjuster(sites_original_convention[st],df,['TA1','TA2'],2021,12,1,'abs_diff',2022,1,12,'xmit outlier?',2)
                df=adjuster(sites_original_convention[st],df,['P'],2021,8,12,'min_filter',2021,12,24,'xmit outlier?',725)
                df=adjuster(sites_original_convention[st],df,['P'],2021,8,12,'max_filter',2021,12,24,'xmit outlier?',810)
                df=adjuster(sites_original_convention[st],df,['TA1','TA2'],2021,8,12,'abs_diff',2021,12,24,'xmit outlier?',2)
                df=adjuster(sites_original_convention[st],df,['TA1','TA2'],2021,8,12,'max_filter',2021,12,12,'xmit outlier?',2)
                df=adjuster(sites_original_convention[st],df,['TA1','TA2'],2021,9,15,'min_filter',2021,12,12,'xmit outlier?',-27)
                df=adjuster(sites_original_convention[st],df,['TA1','TA2'],2021,9,20,'min_filter',2021,12,12,'xmit outlier?',-23)
                df=adjuster(sites_original_convention[st],df,['TA1','TA2'],2021,10,15,'min_filter',2021,11,10,'xmit outlier?',-17)
                df=adjuster(sites_original_convention[st],df,['TA1','TA2'],2021,11,15,'min_filter',2021,12,10,'xmit outlier?',-17)
                df=adjuster(sites_original_convention[st],df,['TA1','TA2'],2021,8,12,'max_filter',2021,9,4,'xmit outlier?',-15)
                df=adjuster(sites_original_convention[st],df,['VW1','VW2'],2021,8,12,'max_filter',2022,12,24,'xmit outlier?',20)
                df=adjuster(sites_original_convention[st],df,['winddir2'],2021,10,10,'max_filter',2022,12,24,'instrument partially out! should get mores study',290)
                df=adjuster(sites_original_convention[st],df,['rh1','rh2'],2021,8,12,'min_filter',2022,12,24,'xmit outlier?',50)
                df=adjuster(sites_original_convention[st],df,['rh1','rh2'],2021,8,12,'max_filter',2022,12,24,'xmit outlier?',97.5)
                # df=adjuster(sites_original_convention[st],df,['winddir2'],2021,8,12,'min_filter',2022,12,24,'instrument partially out! should get mores study',20)
                # df=adjuster(sites_original_convention[st],df,['winddir2'],2021,8,12,'nan_var',2022,12,24,'instrument partially out! should get mores study there is some apparently good data here but for now',0)
                # df=adjuster(sites_original_convention[st],df,['winddir1','winddir2'],2021,8,12,'abs_diff_del_instrument_2',2021,12,24,'instrument partially out! should get mores study there is some apparently good data here but for now',15)
    
            if sites2[st]=='TUN':
                df=adjuster(sites_original_convention[st],df,['P'],2021,6,1,'min_filter',2022,4,15,'xmit outlier?',725)
                df=adjuster(sites_original_convention[st],df,['P'],2021,6,1,'min_filter',2021,6,15,'xmit outlier?',768)
                df=adjuster(sites_original_convention[st],df,['P'],2021,7,21,'min_filter',2021,8,2,'xmit outlier?',771)
                df=adjuster(sites_original_convention[st],df,['P'],2021,12,21,'max_filter',2022,1,12,'xmit outlier?',799)
                df=adjuster(sites_original_convention[st],df,['P'],2021,11,21,'max_filter',2021,12,20,'xmit outlier?',790)
                df=adjuster(sites_original_convention[st],df,['P'],2022,3,1,'min_filter',2022,4,12,'xmit outlier?',750)
                df=adjuster(sites_original_convention[st],df,['rh1','rh2'],2021,6,1,'min_filter',2022,4,12,'xmit outlier?',40)
                df=adjuster(sites_original_convention[st],df,['TA2'],2021,6,1,'min_filter',2021,9,12,'xmit outlier?',-50)
                df=adjuster(sites_original_convention[st],df,['TA1','TA2'],2021,6,1,'min_filter',2022,4,12,'xmit outlier?',-60)
                df=adjuster(sites_original_convention[st],df,['TA1','TA2'],2021,6,1,'max_filter',2022,4,12,'xmit outlier?',6)
                df=adjuster(sites_original_convention[st],df,['TA1','TA2'],2021,6,1,'min_filter',2021,7,15,'xmit outlier?',-22)
                df=adjuster(sites_original_convention[st],df,['TA1','TA2'],2021,9,1,'min_filter',2021,9,15,'xmit outlier?',-50)
                df=adjuster(sites_original_convention[st],df,['TA1','TA2'],2021,6,1,'max_filter',2021,7,1,'xmit outlier?',-3)
                df=adjuster(sites_original_convention[st],df,['TA1','TA2'],2021,12,1,'max_filter',2022,4,12,'xmit outlier?',-15)
                df=adjuster(sites_original_convention[st],df,['TA1','TA2'],2021,12,1,'max_filter',2021,12,18,'xmit outlier?',-17)
                df=adjuster(sites_original_convention[st],df,['TA1','TA2'],2021,12,30,'max_filter',2022,2,18,'xmit outlier?',-22)
                df=adjuster(sites_original_convention[st],df,['VW1','VW2'],2021,6,1,'max_filter',2022,4,12,'xmit outlier?',17.5)
                df=adjuster(sites_original_convention[st],df,['winddir2'],2021,6,1,'nan_var',2022,4,22,'instrument out?',0)
    
            if sites2[st]=='SUM':
                df=adjuster(sites_original_convention[st],df,['rh1'],2021,12,1,'min_filter',2022,4,12,'xmit outlier?',35)
                df=adjuster(sites_original_convention[st],df,['rh1','rh2'],2021,4,12,'min_filter',2022,7,31,'xmit outlier?',41)
    
                # df.VW1[:]=np.nan
                # df.VW2[( (df.time>datetime(2022,4,1)) & (df.time<datetime(2022,5,31)))]=np.nan
                # df.winddir1[:]=np.nan
                # df.winddir2[( (df.time>t0) & (df.time<t1))]=np.nan
    
                # df=adjuster(sites_original_convention[st],df,['VW1','VW2'],2022,4,1,'swap',2022,5,31,'swap lev 1 and 2',0)
    
                df=adjuster(sites_original_convention[st],df,['rh1','rh2'],2021,4,12,'min_filter',2022,5,31,'xmit outlier?',41)
                df=adjuster(sites_original_convention[st],df,['TA1','TA2'],2021,12,1,'max_filter',2022,7,31,'xmit outlier?',0)
                df=adjuster(sites_original_convention[st],df,['VW1','VW2'],2022,4,1,'max_filter',2022,5,31,'xmit outlier?',20)
                df=adjuster(sites_original_convention[st],df,['VW1','VW2'],2022,4,1,'max_filter',2022,7,31,'xmit outlier?',20)
                df.rh1[( (df.time>datetime(2021,12,1)) & (df.time<datetime(2022,4,15)) & (df.rh1<35) )]=np.nan
                df.rh2[( (df.time>datetime(2021,12,1)) & (df.time<datetime(2022,4,15)) & (df.rh2<35) )]=np.nan
                df=adjuster(sites_original_convention[st],df,['winddir1'],2022,5,1,'nan_var',2022,5,16,'instrument stuck-ish',0)
                df=adjuster(sites_original_convention[st],df,['winddir1'],2022,4,25,'nan_var',2022,4,30,'instrument stuck-ish',0)
                df=adjuster(sites_original_convention[st],df,['winddir1'],2022,4,1,'nan_var',2022,4,22,'instrument stuck-ish',0)
                df=adjuster(sites_original_convention[st],df,['VW1','VW2'],2021,4,1,'min_filter',2022,4,17,'instrument stuck-ish',1)
                df=adjuster(sites_original_convention[st],df,['VW1'],2022,7,1,'max_filter',2022,7,31,'xmit outlier?',12)
                df=adjuster(sites_original_convention[st],df,['P'],2022,6,26,'min_filter',2022,6,30,'xmit outlier?',737)

            if sites2[st]=='DY2':
                df = df.rename({'VW1': 'temp1', 'VW2': 'temp2'}, axis=1)
                df = df.rename({'temp1': 'VW2', 'temp2': 'VW1'}, axis=1)
                df=adjuster(sites_original_convention[st],df,['VW1'],2021,6,1,'max_filter',2021,8,1,'xmit outlier?',20)
                df=adjuster(sites_original_convention[st],df,['VW2'],2021,12,1,'max_filter',2021,12,31,'xmit outlier?',30)
                df=adjuster(sites_original_convention[st],df,['VW1','VW2'],2022,4,23,'max_filter',2022,4,27,'xmit outlier?',20)
                df.Battery[( (df.time>datetime(2021,12,1)) & (df.time<datetime(2021,12,31)) & (df.Battery>16) )]=np.nan
                
                df=adjuster(sites_original_convention[st],df,['P'],2021,6,1,'min_filter',2022,5,1,'xmit outlier?',725)
                df=adjuster(sites_original_convention[st],df,['rh1'],2021,6,1,'min_filter',2022,4,22,'xmit outlier?',30)
                df=adjuster(sites_original_convention[st],df,['rh1'],2022,4,1,'min_filter',2022,5,31,'xmit outlier?',51)
    
                df=adjuster(sites_original_convention[st],df,['P'],2021,6,1,'nan_var',2022,7,31,'partially broken barometer can fix with regression?',0)
                # new batch 8 Aug
                df=adjuster(sites_original_convention[st],df,['TA1','TA2'],2021,5,1,'max_filter',2022,7,31,'xmit outlier?',4)
                df=adjuster(sites_original_convention[st],df,['VW1','VW2'],2022,6,1,'max_filter',2022,7,31,'xmit outlier?',20)
                df=adjuster(sites_original_convention[st],df,['rh1'],2022,6,1,'min_filter',2022,7,31,'xmit outlier?',65)
    
            if sites2[st]=='NAE':
                df=adjuster(sites_original_convention[st],df,['TA1','TA2'],2021,6,1,'max_filter',2022,5,8,'xmit outlier?',4)
                df=adjuster(sites_original_convention[st],df,['rh1','rh2'],2021,6,1,'min_filter',2022,5,8,'xmit outlier?',40)
                df=adjuster(sites_original_convention[st],df,['VW1','VW2'],2021,6,1,'max_filter',2022,5,8,'xmit outlier?',15)
                df=adjuster(sites_original_convention[st],df,['winddir2'],2021,6,1,'nan_var',2022,7,31,'partially broken?',0)
                df=adjuster(sites_original_convention[st],df,['VW1'],2021,6,1,'nan_var',2022,7,31,'partially broken?',0)
                df=adjuster(sites_original_convention[st],df,['VW2'],2022,5,10,'min_filter',2022,5,22,'frozen propellor?',1)
    
                df.Battery[( (df.time>datetime(2021,12,1)) & (df.time<datetime(2031,12,31)) & (df.Battery>16) )]=np.nan

                df=adjuster(sites_original_convention[st],df,['rh1','rh2'],2022,5,31,'min_filter',2022,7,31,'xmit outlier?',50)
                df=adjuster(sites_original_convention[st],df,['VW1','VW2'],2022,6,1,'max_filter',2022,7,31,'xmit outlier?',15)
                df=adjuster(sites_original_convention[st],df,['TA1','TA2'],2022,5,31,'max_filter',2022,7,31,'xmit outlier?',4)


            if sites2[st]=='EGP':
                # df=adjuster(sites_original_convention[st],df,['TA1','TA2'],2021,6,1,'max_filter',2022,5,8,'xmit outlier?',4)
                # df=adjuster(sites_original_convention[st],df,['rh1','rh2'],2021,6,1,'min_filter',2022,5,8,'xmit outlier?',40)
                # df=adjuster(sites_original_convention[st],df,['VW2'],2022,4,28,'max_filter',2022,5,1,'xmit outlier?',12)
                df=adjuster(sites_original_convention[st],df,['winddir1','VW1'],2022,4,1,'nan_var',2022,5,31,'broken or unwired?',0)
                df=adjuster(sites_original_convention[st],df,['P'],2022,6,1,'min_filter',2022,7,31,'xmit outlier?',706)
                df=adjuster(sites_original_convention[st],df,['P'],2022,6,25,'min_filter',2022,7,31,'xmit outlier?',721)
                df=adjuster(sites_original_convention[st],df,['TA1','TA2'],2021,5,1,'max_filter',2022,7,31,'xmit outlier?',4)
                df=adjuster(sites_original_convention[st],df,['rh1','rh2'],2022,5,31,'min_filter',2022,7,31,'xmit outlier?',50)
                df=adjuster(sites_original_convention[st],df,['rh1','winddir1','TA1','VW1'],2022,6,1,'nan_var',2022,7,31,'lower arm burried',0)
                df=adjuster(sites_original_convention[st],df,['VW1','VW2'],2022,6,1,'max_filter',2022,7,31,'xmit outlier?',15)
                df=adjuster(sites_original_convention[st],df,['VW1','VW2'],2022,7,8,'max_filter',2022,7,31,'xmit outlier?',8.5)

            #,header=none,names=cols)
            #'/Users/jason/Dropbox/AWS/CARRA-TU_GEUS_AWS/output/'+site+'.csv'
    
            df['winddirection']=df[['winddir1', 'winddir2']].mean(axis=1)
    
            df['temperature']=df[['TA1', 'TA2']].mean(axis=1)
            
            if sites2[st]!='DY2':
                df['relativehumidity']=df[['rh1', 'rh2']].mean(axis=1)
            else:
                df['relativehumidity']=df.rh1
            df['windspeed']=df[['VW1', 'VW2']].mean(axis=1)        
    
            t0=datetime(2022, 6, 1) ; t1=datetime(2022, 7, 31)
    
            # if sites2[st]=='SUM':
            #     t0=datetime(2022,3,13,14)
            #     t1=datetime(2022,4,20,12)
    
            # if sites2[st]=='HUM':
            #     t0=datetime(2021,8,12,14)
            #     t1=datetime(2021,12,24,12)
            
            df['Lat_decimal']=np.nan ; df['Lat_decimal'][:]=meta.lat[v_meta]
            df['Lon_decimal']=np.nan ; df['Lon_decimal'][:]=meta.lon[v_meta]
            df['elev']=np.nan ; df['elev'][:]=meta.elev[v_meta]
    
            # cols=df.columns
            # k=1
            # var=df[cols[k]]
                    
    
            if plt_diagnostic:
                plt.close()
                n_rows=3
                fig, ax = plt.subplots(n_rows,1,figsize=(10,14))
                cc=0
        
                ax[cc].set_title(sites2[st]+' WSL AWS transmissions '+str(df.date[0].strftime('%Y %b %d %H'))+' until '+str(df.date[-1].strftime('%Y %b %d %H')))
                if show_rejected:ax[cc].plot(df.P_raw,'.r',label='pressure rejected')
        
        # Pressure
                if do_batch1:
                    do_P=1
                else:do_P=0
                if do_P:
                    ax[cc].plot(df.P,'.',label='pressure')
                    # ax[cc].get_xaxis().set_visible(False)
                    ax[cc].legend(loc='center left', bbox_to_anchor=(1, 0.5))
                    ax[cc].set_xlim(t0,t1)
                    cc+=1
        
        # air temperature
                if do_batch1:
                    do_T=1
                else:do_T=0
                if do_T:
                    ax[cc].plot(df.TA1,'.',label='TA1')
                    ax[cc].plot(df.TA2,'.',label='TA2')
                    # ax[cc].plot(df.temperature,'.',label='temperature')
                    # ax[cc].get_xaxis().set_visible(False)
                    ax[cc].legend(loc='center left', bbox_to_anchor=(1, 0.5))
                    ax[cc].set_xlim(t0,t1)
                    cc+=1
        # dT
                if do_batch1:
                    do_dT=1
                else:do_dT=0
                if do_dT:
                    # ax[cc].plot(df.TA1,'.',label='TA2')
                    ax[cc].plot(df.TA2-df.TA1,'.',label='TA2-TA1')
                    # ax[cc].plot(df.TA2,'.',label='TA2')
                    # ax[cc].plot(df.temperature,'.',label='temperature')
                    # ax[cc].get_xaxis().set_visible(False)
                    ax[cc].legend(loc='center left', bbox_to_anchor=(1, 0.5))
                    ax[cc].set_xlim(t0,t1)
                    # cc+=1
        # humidity
                if do_batch1==0:
                    do_rh=1
                else:do_rh=0
                if do_rh:
                    if show_rejected:
                        ax[cc].plot(df.rh1_raw,'.r',label='rh1 rejected')
                        ax[cc].plot(df.rh2_raw,'.k',label='rh2 rejected')
            
                    ax[cc].plot(df.rh1,'.',label='rh1')
                    ax[cc].plot(df.rh2,'.',label='rh2')
                    # ax[cc].plot(df.relativehumidity,'.',label='relativehumidity')
                    # ax[cc].get_xaxis().set_visible(False)
                    ax[cc].legend(loc='center left', bbox_to_anchor=(1, 0.5))
                    ax[cc].set_xlim(t0,t1)
                    cc+=1
        # wind speed
                if do_batch1==0:
                    do_VW=1
                else:do_VW=0
                if do_VW:
                    # ax[cc].plot(df.VW1_raw,'.r',label='wind1 rejected')
                    # ax[cc].plot(df.VW2_raw,'.k',label='wind2 rejected')
                    ax[cc].plot(df.VW1,'.',label='VW1')
                    ax[cc].plot(df.VW2,'.',label='VW2')
                    # ax[cc].plot(df.windspeed,'.',label='windspeed')
                    # ax[cc].get_xaxis().set_visible(False)
                    ax[cc].legend(loc='center left', bbox_to_anchor=(1, 0.5))
                    ax[cc].set_xlim(t0,t1)
                    cc+=1
        # wind dir
                if do_batch1==0:
                    do_dir=1
                else:do_dir=0
                if do_dir:
                    # ax[cc].plot(df.winddir1_raw,'.r',label='wind dir1 rejected')
                    # ax[cc].plot(df.winddir2_raw,'.k',label='wind dir2 rejected')
                    ax[cc].plot(df.winddir1,'.',label='winddirection 1')
                    ax[cc].plot(df.winddir2,'.',label='winddirection 2')
                    # ax[cc].plot(df.winddirection,'.',label='winddirection')
                    # ax[cc].get_xaxis().set_visible(False)
                    ax[cc].legend(loc='center left', bbox_to_anchor=(1, 0.5))
                    ax[cc].set_xlim(t0,t1)
                    # cc+=1
        # elev
                do_elev=0
                if do_elev:
                    ax[cc].plot(df.elev,'.',label='elev')
                    ax[cc].get_xaxis().set_visible(False)
                    ax[cc].legend()
                    ax[cc].set_xlim(t0,t1)
                    cc+=1                
        # battery    
                do_bat=0
                if do_bat:
                    ax[cc].plot(df.Battery,'.',label='Battery')
                    ax[cc].set_xlim(t0,t1)
                    ax[cc].legend(loc='center left', bbox_to_anchor=(1, 0.5))
                
                plt.setp(ax[cc].xaxis.get_majorticklabels(), rotation=90,ha='center' )
                ax[cc].xaxis.set_major_formatter(mdates.DateFormatter('%Y %b %d'))
                
                if ly == 'x':
                    plt.show()
        
            if wo:
                drop=1
                if drop:
                    # if meta.alt_name[i]!='JAR':
                    # df.drop(df[df.dec_year<2021.42].index, inplace=True)
                    df.drop(df[df.date>date1].index, inplace=True)
                    df.reset_index(drop=True, inplace=True)
                df['airpressureminus1000']=df.P-1000
                # cols_requested=['year','day','month','hour','airpressureminus1000','temperature','relativehumidity','windspeed','winddirection','Lat_decimal','Lon_decimal','elev']
                cols_requested=['date','airpressureminus1000','temperature','relativehumidity','windspeed','winddirection','Lat_decimal','Lon_decimal','elev']
    
                df.columns
                # cols_requested=['airpressureminus1000','temperature','relativehumidity','windspeed','winddirection','Lat_decimal','Lon_decimal','elev']
                # df["date"]=pd.to_datetime(df[['year','month','day','hour']])
                df2=df[cols_requested]
    
                df2.index=pd.to_datetime(df[['year','month','day','hour']])
                # df2.index = df2.index.date
                # df2.index = pd.to_datetime(df.date)
                
        
                if len(df)>0:
                    formats = {'airpressureminus1000': '{:.1f}','windspeed': '{:.1f}','relativehumidity': '{:.1f}','temperature': '{:.1f}','winddirection': '{:.1f}','relativehumidity': '{:.1f}','Lat_decimal': '{:.5f}','Lon_decimal': '{:.5f}','elev': '{:.2f}'}
                for col, f in formats.items():
                    df2[col] = df2[col].map(lambda x: f.format(x))
                opath='./AWS_data_for_CARRA-TU/data_range_'+date0+'_to_'+date1+'/GC-Net_WSL/'
                os.system('mkdir -p '+opath)
    
                ofile=opath+sites2[st]+'.csv'
                df2.to_csv(ofile,index=None)
            
                #-------------------------------------------------- plot all
                df=pd.read_csv(ofile)
                df["date"]=pd.to_datetime(df.iloc[:,0])
                # df.index = df.index.date
                print(df)
                
                plt.close()
                n_rows=5
                fig, ax = plt.subplots(n_rows,1,figsize=(10,14))
                cc=0
                
                n=len(df)
                ax[cc].set_title(sites2[st]+' WSL AWS transmissions '+str(df.date[0].strftime('%Y %b %d %H'))+' until '+str(df.date[n-1].strftime('%Y %b %d %H')+' UTC'))
                df.set_index('date', inplace=True)
        
        # Pressure        
                ax[cc].plot(df.airpressureminus1000,'.',label='airpressureminus1000')
                # ax[cc].get_xaxis().set_visible(False)
                ax[cc].legend(loc='center left', bbox_to_anchor=(1, 0.5))
                ax[cc].set_xlim(t0,t1)
                cc+=1
        
        # air temperature
                ax[cc].plot(df.temperature,'.',label='temperature')
                # ax[cc].plot(df.temperature,'.',label='temperature')
                # ax[cc].get_xaxis().set_visible(False)
                ax[cc].legend(loc='center left', bbox_to_anchor=(1, 0.5))
                ax[cc].set_xlim(t0,t1)
                cc+=1
        
        # humidity
                # if show_rejected:
                #     ax[cc].plot(df.rh1_raw,'.r',label='rh1 rejected')
                #     ax[cc].plot(df.rh2_raw,'.k',label='rh2 rejected')
        
                ax[cc].plot(df.relativehumidity,'.',label='relativehumidity')
                # ax[cc].get_xaxis().set_visible(False)
                ax[cc].legend(loc='center left', bbox_to_anchor=(1, 0.5))
                ax[cc].set_xlim(t0,t1)
                cc+=1
        # wind speed
                # ax[cc].plot(df.VW1_raw,'.r',label='wind1 rejected')
                # ax[cc].plot(df.VW2_raw,'.k',label='wind2 rejected')
                ax[cc].plot(df.windspeed,'.',label='windspeed')
                # ax[cc].plot(df.windspeed,'.',label='windspeed')
                # ax[cc].get_xaxis().set_visible(False)
                ax[cc].legend(loc='center left', bbox_to_anchor=(1, 0.5))
                ax[cc].set_xlim(t0,t1)
                cc+=1
        # wind dir
                # ax[cc].plot(df.winddir1_raw,'.r',label='wind dir1 rejected')
                # ax[cc].plot(df.winddir2_raw,'.k',label='wind dir2 rejected')
                ax[cc].plot(df.winddirection,'.',label='wind dir')
                # ax[cc].plot(df.winddirection,'.',label='winddirection')
                # ax[cc].get_xaxis().set_visible(False)
                ax[cc].legend(loc='center left', bbox_to_anchor=(1, 0.5))
                ax[cc].set_xlim(t0,t1)
                
                plt.setp(ax[cc].xaxis.get_majorticklabels(), rotation=90,ha='center' )
                ax[cc].xaxis.set_major_formatter(mdates.DateFormatter('%Y %b %d'))
                
                # if ly == 'p':
                fig_path=opath+'Figs/ '
                os.system('mkdir -p '+fig_path)
                plt.savefig(fig_path+sites2[st]+'.png', bbox_inches='tight', dpi=72)