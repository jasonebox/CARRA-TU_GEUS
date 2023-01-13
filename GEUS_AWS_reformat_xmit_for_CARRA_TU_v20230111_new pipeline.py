#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 13:38:41 2022
# %matplotlib inline
# %matplotlib osx

"""
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from numpy.polynomial.polynomial import polyfit
from datetime import date
import os

today = date.today()
versionx= today.strftime('%Y-%m-%d')


if os.getlogin() == 'jason':
    base_path = '/Users/jason/Dropbox/AWS/CARRA-TU_GEUS/'

os.chdir(base_path)

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

# sites=['KPC_L','KPC_Lv3','KPC_U','KPC_Uv3','EGP','SCO_L','SCO_U','MIT*','TAS_L','TAS_U***','TAS_A','QAS_L','QAS_Lv3','QAS_M','QAS_U','QAS_Uv3','QAS_A***','NUK_L','NUK_U','NUK_Uv3','NUK_N***','KAN_B**','KAN_L','KAN_M','KAN_U','UPE_L','UPE_U','THU_L','THU_L2','THU_U***','THU_U2','CEN***','GC-NET','CEN2','CP1DY2','HUM','JAR','JAR_O','NAE','NAU','NEM','NSE','SDL','SDM','SWC','SWC_O','TUN','GlacioBasis','NUK_K*','ZAK_Lv3*','ZAK_Uv3*']

# tx
sitesx = pd.read_csv('/Users/jason/0_dat/AWS/aws-l3/AWS_station_locations.csv')
sitesx = sitesx.rename({'stid': 'site'}, axis=1)

print(sitesx.columns)
sites=sitesx.site

# -------------------------------- dates covered by this delivery
date0='2022-07-01'; date1='2022-12-31'

do_plot=1

cols_requested=['year','day','month','hour','airpressureminus1000','temperature','relativehumidity','windspeed','winddirection','Lat_decimal','Lon_decimal','elev']

# meta_file='./metadata/IMEI_numbers_station_2021-10-27.xlsx'
# # os.system('open '+meta_file)
# meta = pd.read_excel(meta_file)
# print(meta.columns)

# names=meta.ASSET

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

    if func == "rotate":
        df_out.loc[t0:t1, var_list[0]] = df_out.loc[t0:t1, var_list[0]].values + val
        df_out.loc[t0:t1, var_list[0]][df_out.loc[t0:t1, var_list[0]] > 360] = (
            df_out.loc[t0:t1, var_list[0]] - 360
        )
        count=len(df_out.loc[t0:t1])

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

#%%
tx_or_standard='tx'
# tx_or_standard='level_3'

for i,site in enumerate(sites):

    # flag=np.sum(names==site)
    # print(site)
    # if flag:
    if site=='NSE':
    # if i>=0:

        if site!='ZAK_L' and site!='Roof_GEUS' and site!='Roof_PROMICE' and site!='MIT' and site!='KPC_Uv3':
            # time range to consider
            t0=datetime(2022,7,1) ; t1=datetime(2022, 12, 31)
            
            ly='p'
            wo=1
            
            print()

            # tx
            fn='/Users/jason/0_dat/AWS/aws-l3/'+tx_or_standard+'/'+site+'/'+site+'_hour.csv'
            # fn='/Users/jason/0_dat/AWS/aws-l3/level_3/'+site+'/'+site+'_hour.csv'
            print(fn)
            df=pd.read_csv(fn)
            print(df.columns)
            
            # filtering of how NaN is spelled
            # df[df=="NAN"]=np.nan
            # df[df=="nan"]=np.nan
            
            # date
            df["date"]=pd.to_datetime(df.time)
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
                # if site!='JAR':
                df.drop(df[df.date<date0].index, inplace=True)
                df.reset_index(drop=True, inplace=True)
                df.drop(df[df.date>date1].index, inplace=True)
                df.reset_index(drop=True, inplace=True)
                # df.drop(df[df.date>=date1].index, inplace=True)
                # df.reset_index(drop=True, inplace=True)
                # assa
            # print(df.gps_alt)
            
            print(df.columns)

                # t0=datetime(2022, 4, 1) ; t1=datetime(2022, 5, 31)
    
            # dfx=df.copy()
            # dfx=dfx.drop(dfx.columns[1:3], axis=1)
            # dfx.to_csv('/Users/jason/Dropbox/AWS/GCNET/GCNv2_xmit/output/'+names[i]+'.csv')
    
            # for kk,col in enumerate(cols):
            #     if kk>0:
            #         df[col] = pd.to_numeric(df[col])
            #         print(col,df[col][10])
    
            # df['temperature'] = pd.to_numeric(df['temperature'])
            # df['relativehumidity'] = pd.to_numeric(df['relativehumidity'])
            # df['airpressureminus1000'] = pd.to_numeric(df['airpressureminus1000'])
            df['rh_u_cor'][df['rh_u_cor']>105]=np.nan
            df['rh_u_cor'][df['rh_u_cor']<30]=np.nan

            if tx_or_standard=='tx':
                df['airpressureminus1000']=df.p_i#-1000
                df['winddirection']=df.wdir_i
                df['windspeed']=df.wspd_i
                df['relativehumidity']=df.rh_i
                df['temperature']=df.t_i
            else:
                df['airpressureminus1000']=df.p_u-1000
                df['winddirection']=df.wdir_u
                df['windspeed']=df.wspd_u
                df['relativehumidity']=df.rh_u
                df['temperature']=df.t_u

            # df['airpressureminus1000']=df.airpressure
            # df['airpressureminus1000'][df['airpressureminus1000']>1000]=np.nan
            # df.wdir_i[df.wdir_i.diff()==0]=np.nan
            # df.wspd_i[df.wspd_i.diff()==0]=np.nan
    
            # print(df.columns)
            
            # df['relativehumidity'][((df['relativehumidity'].diff()==0)&(df['relativehumidity']<30))]=np.nan
    
            # if meta.network[i]=='g':
            #     df.airpressureminus1000-=1000
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
            df['Lat_decimal']=df.gps_lat#.astype(float)
            df['Lon_decimal']=df.gps_lon#.astype(float)
            df['Altitude']=df.gps_alt#.astype(float)
            # print(df.elev)
    
            # df['Lat_decimal']=np.nan
            # df['Lon_decimal']=np.nan
            
            # v=np.where(np.isfinite(df.Latitude)) ; v=v[0]
            # df['lat_min']=np.nan
            # df['lon_min']=np.nan
            # df['Lat_decimal']=np.nan
            # df['Lon_decimal']=np.nan
            # df['lat_min'][v]=(df.Latitude[v]/100-(df.Latitude[v]/100).astype(int))*100
            # df['lon_min'][v]=(df.Longitude[v]/100-(df.Longitude[v]/100).astype(int))*100
            # df['Lat_decimal'][v]=(df.Latitude[v]/100).astype(int)+df['lat_min'][v]/60
            # df['Lon_decimal'][v]=(df.Longitude[v]/100).astype(int)+df['lon_min'][v]/60
    
            # if len(df['Lat_decimal'][df['Lat_decimal']<60])>0:
    
                # plt.close()
                # plt.plot(df['Lat_decimal'])
                # plt.title(site)
                # plt.show()
                # print('dropping test data')
                # df.drop(df['Lat_decimal'][df['Lat_decimal']<60].index, inplace=True)
                # df.reset_index(drop=True, inplace=True)
                
            # reject if more than 3 x std from mean
            # v=abs(df['Altitude']-np.nanmean(df['Altitude']))>(2*np.nanstd(df['Altitude']))
            # df['Altitude'][v]=np.nan
            # plt.plot(df['Altitude'])
            d_alt=df['Altitude']-np.nanmedian(df['Altitude'])
            dev_alt=np.nanstd(df['Altitude'])*2
            df['Altitude'][abs(d_alt)>dev_alt]=np.nan
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
            if site=='CP1':
                df=adjuster(site,df,['airpressureminus1000'],2022,7,1,'min_filter',2022,12,31,'xmit outlier?',-250)
                df=adjuster(site,df,['airpressureminus1000'],2022,7,1,'max_filter',2022,12,31,'xmit outlier?',-150)
                # print(np.nanmin(df['airpressureminus1000']))
                # plt.plot(df['airpressureminus1000'])

            if site=='HUM' and tx_or_standard=='tx':
                df=adjuster(site,df,['airpressureminus1000'],2022,11,1,'min_filter',2022,12,31,'xmit outlier?',-243)
                df=adjuster(site,df,['airpressureminus1000'],2022,11,1,'max_filter',2022,12,1,'xmit outlier?',-213)
            
            if site=='HUM' and tx_or_standard=='level_3':
                df=adjuster(site,df,['airpressureminus1000'],2022,11,1,'min_filter',2022,12,31,'xmit outlier?',-243)
                df=adjuster(site,df,['airpressureminus1000'],2022,11,1,'max_filter',2022,12,1,'xmit outlier?',-213)
                # df=adjuster(site,df,['airpressureminus1000'],2022,12,1,'min_filter',2022,12,31,'xmit outlier?',-243)
                df=adjuster(site,df,['airpressureminus1000'],2022,12,15,'max_filter',2022,12,31,'xmit outlier?',-200)                
                
            if site=='NAU':
                df=adjuster(site,df,['airpressureminus1000'],2022,12,23,'min_filter',2022,12,28,'xmit outlier?',-267)
                df=adjuster(site,df,['airpressureminus1000'],2022,12,23,'max_filter',2022,12,31,'xmit outlier?',-238)

            if site=='NEM':
                df=adjuster(site,df,['airpressureminus1000'],2022,7,1,'min_filter',2022,12,31,'xmit outlier?',-290)
                df=adjuster(site,df,['airpressureminus1000'],2022,7,1,'max_filter',2022,12,31,'xmit outlier?',-220)
                df=adjuster(site,df,['windspeed'],2022,11,1,'nan_var',2022,12,31,'instrument frozen?',0)

            if site=='NSE':
                df=adjuster(site,df,['airpressureminus1000'],2022,4,1,'min_filter',2022,5,31,'xmit outlier?',-350)
                # df=adjuster(site,df,['airpressureminus1000'],2022,5,31,'min_filter',2022,7,31,'xmit outlier?',-1300)
                # df=adjuster(site,df,['airpressureminus1000'],2022,5,31,'max_filter',2022,7,31,'xmit outlier?',-1100)
                df=adjuster(site,df,['temperature'],2022,4,1,'min_filter',2022,7,31,'xmit outlier?',-90)
                df=adjuster(site,df,['temperature'],2022,4,1,'max_filter',2022,7,31,'xmit outlier?',9)
                df=adjuster(site,df,['temperature'],2022,7,1,'min_filter',2022,12,31,'xmit outlier?',-50)
                df=adjuster(site,df,['relativehumidity'],2022,7,1,'min_filter',2022,12,31,'xmit outlier?',50)

                df.temperature[df.temperature<-90]=np.nan
                df.temperature[df.temperature>-9]=np.nan
    
            if site=='NUK_U':
                df=adjuster(site,df,['windspeed'],2022,4,1,'nan_var',2022,5,31,'propellor not secure?',0)
            
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

    
            if site=='SWC':
                df=adjuster(site,df,['winddirection'],2020,8,9,'rotate',2022,8,3,'wind dir box 180deg off',180)
    
            if do_plot:
    
                N=len(df)
                if N>0:
    
                    opath='./AWS_data_for_CARRA-TU/data_range_'+date0+'_to_'+date1+'/PROMICE_GC-Net_GEUS/'+tx_or_standard+'/'
                    os.system('mkdir -p '+opath)
    
                    fs=20
                    plt.close()
                    plt.rcParams["font.size"] = fs
                    plt.scatter(x,y,color='grey')
                    plt.plot((x[0],x[-1]),(x[0]*m+b,x[-1]*m+b),linewidth=th*3,color='r')
                    plt.ylabel('elevation, m')
                    plt.xlabel('year')
                    plt.title(site)
                    
                    if ly == 'p':
                        fig_path=opath+'Figs/'
                        os.system('mkdir -p '+fig_path)
                        fig_path=opath+'Figs/elev/'
                        os.system('mkdir -p '+fig_path)
                        plt.savefig(fig_path+site+'_elev.png', bbox_inches='tight', dpi=72)
    
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
                    df2.to_csv(opath+site+'.csv')
                    
    
                    plt.close()
                    n_rows=7
                    fig, ax = plt.subplots(n_rows,1,figsize=(10,16))
    
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
                    cc+=1                
                    
                    ax[cc].plot(df.batt_v,'.',label='Battery')
                    ax[cc].set_xlim(t0,t1)
                    ax[cc].legend()
                    
                    plt.setp(ax[cc].xaxis.get_majorticklabels(), rotation=90,ha='center' )
                    ax[cc].xaxis.set_major_formatter(mdates.DateFormatter('%Y %b %d'))
                    
                    if ly == 'p':
                        fig_path=opath+'Figs/'
                        os.system('mkdir -p '+fig_path)
                        plt.savefig(fig_path+site+'.png', bbox_inches='tight', dpi=72)
                # else:
                    #     df2.to_csv(opath+site+'.csv')
                
        
                    
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
    
                # ax[cc].set_title(site+' AWS transmissions until '+str(df.date[-1].strftime('%Y %b %d %HUTC')))
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
                #     plt.savefig(opath+site+'.png', bbox_inches='tight', dpi=300)
