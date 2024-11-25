#!/usr/bin/env python
# coding: utf-8

# In[27]:


import datetime

#import dask.dataframe as dd
import numpy as np
import pandas as pd
from xbbg import blp


# In[28]:


Ticker_dict={
    # US, Germany, Italy, France, GBP, Spain, JPY
    'currncy':{
        'tickers':[
            'RP','RY','RF','PLE','AN','NJ','PS','CA','EZ','HR','EA','PJ','SJ','ZQ','EN','AJ','ZK','KRP','EW','SWR',
            'KOL','YF','YK','EI','KRW','KGB','BDR','ZI','CY','KZY','KEL','HL','YA','AR','ZD','KR','EL','AC','AS','AC',
            'AS','XAY','AJ','XAYX','YAY','AN','AR','ASD','KA','BSR'
        ],

         'numbers':['1','2'],
        'fields':['PX_LAST','PX_BID','PX_ASK','PX_MID','FUT_NOTICE_FIRST','FUT_DLV_DT_LAST','FUT_DLV_DT_FIRST',
            'FUT_CUR_GEN_TICKER','VOLUME','OPEN_INT']           
    },
    'index': {
        'tickers':[
            'XB','ES','HWB','NO','HW','VG','NQ','XU','KM','IFD','AF','NZ','RTY','CA','MX','JAI','BC','HC','FFD','IFB',
            '0J','UX','HI','A5','DM','VNC','HU','FT','HWR','KST','BZ','KMS','FFB','HCT','SXO','HWI','MES','FNY','FDO',
            'Z ','QC','STE','VE','GX','TWT','NK','JGS','TP','QZ','XP','CF','KRS','MCY','DFW','FVS','MFS','FNP','WSP',
#             'NI','DED','NH','XMB','SM','FZY','PT','IUI','VXT','MZS','XMC','TMI','AI','VHO','ST','AXR','AXW','TFS','FA',
#             'EO','MHC','VEY','IB','FXY','ASD','NX','RNS','IK','QNT','MRO','JPW','VXL','DCP','DBE','FNS','RFC','ZWP',
#             'SX','HRT','KG'
        ],
        'numbers':['1','2','3'],
        'fields':['PX_LAST','PX_BID','PX_ASK','PX_MID','FUT_NOTICE_FIRST','FUT_DLV_DT_LAST','FUT_DLV_DT_FIRST',
            'FUT_CUR_GEN_TICKER','VOLUME','OPEN_INT']           
    },
    'bonds' :{
        'tickers':['WN','TWE','US','UXY','TY','FV','3Y','TU',  # Us ultra, 20y, Long, 10y Ultra, 10,5,3,2
                   'CN','XQ','CV',  # CAD 10,5,2,
                   'UB','RX','OE','DU', ## Buxl, Bund, Bobl, Schatz
                   'OAT','BTA', # FR OAT, mid-term
                   'G ', 'WB',  # Long Gilt, Short Gilt
                   'IK', 'BTS', #IT BTP, short BTP
                   'KOA', #SP 10y
                   'BUO','BTO','BTL', #SWED 10,5,2
                   'FB', #SW 10y
                   'JJA','JB', 'BJ',  #Mini JPN 20, 10y, Mini JGB 10y
                   'KAA', 'KE', #Korea 10y, 3y
                   'XM','VTA', 'YM' # AU 10y, 5y ,3y       
                  ],
        # from WBF = CN = CN10y, 
        'numbers':['1','2','3'],
        'fields':['PX_LAST','PX_BID','PX_ASK','PX_MID','FUT_NOTICE_FIRST','FUT_DLV_DT_LAST','FUT_DLV_DT_FIRST',
            'FUT_CUR_GEN_TICKER','VOLUME','OPEN_INT']
       },
    'stirt':{
        # need to find fixings for first contract carry
        'tickers':[
            'SFR','ER','SFI','FF','IR','SER','COR','BA','SSY','RA','ZB','IB','KTR','ORI','JO','TKY',
            'ED', 'L ',  'ES', 'YE'],  # some included for history
        
#         Eurodollar	ED
# Euroyen	EY
# Euroswiss	ES
# Euro	6E
# British Pound	6B
# Canadian Dollar	6C
# Australian Dollar	6A
# New Zealand Dollar	6N
# Japanese Yen	6J
# Swiss Franc	6S
# Mexican Peso	6M
# Brazilian Real	6L
# Russian Ruble	6R
# Chinese Renminbi	CNY
        
        'numbers':[str(x) for x in range(1,12)], # whites, reds, greens,  NO blues or golds
        'fields':['PX_LAST','PX_BID','PX_ASK','PX_MID','FUT_NOTICE_FIRST','FUT_DLV_DT_LAST','FUT_DLV_DT_FIRST',
             'FUT_CUR_GEN_TICKER','VOLUME','OPEN_INT']
        },
    'comdty':{
        'tickers': [
            'C ','AC','DCS','CRD','EP','CT', 'CA','W ','KW','MW','EB','XA','IAC','MFI','XW', 'GC','SI','PL','PA','LA',
            'HG','LX','LL','LN','ALE','LT', 'JN','LBO','PGP','UFD','PCW','QS','HO','XB','PG','NV','QST','BAP','Z0',
            'NG','FN','PNG'
        ],
           'numbers':['1','2','3'],
                'fields':['PX_LAST','PX_BID','PX_ASK','PX_MID','FUT_NOTICE_FIRST','FUT_DLV_DT_LAST','FUT_DLV_DT_FIRST',
             'FUT_CUR_GEN_TICKER','VOLUME','OPEN_INT']
    }
}


# In[29]:


API_TASKS={}
# use sets to ensure uniquenss
for k in Ticker_dict.keys():
    API_TASKS[k] = {
    'instruments':list({x+y+' Comdty' for y in Ticker_dict[k]['numbers'] for x in Ticker_dict[k]['tickers'] }),
    'fields' : Ticker_dict[k]['fields']}
    
    # 'Curncy'?


# In[30]:


API_TASKS.keys()


# In[31]:


comdty_fields = API_TASKS['comdty']['fields']
comdty_fields


# In[32]:


for i in ['comdty']: # API_TASKS.keys():
    print(i)   #API_TASKS[i].keys())
    print(API_TASKS[i]['instruments'])
    print(API_TASKS[i]['fields'])


# In[33]:


API_TASKS['comdty']['instruments'].sort()
API_TASKS['comdty']['instruments']


# In[34]:


start_date=datetime.date(2000,1,1)
end_date = datetime.date.today()


# In[35]:


all_data = pd.DataFrame()
df = {}


# In[36]:


# kwargs = {
# #         'timeout': 1000,
# #         'allow_missing': True,
#         'Fill':'NA'
#     }
blp.bdh(tickers=['W 3 Comdty'], flds=comdty_fields,start_date=datetime.date(2022,1,1), end_date=end_date)   #,**kwargs)
# blp.bdh(tickers=['EUSA10 Index'], flds=['PX_LAST'], start_dt=datetime.date(2022,1,1),end_dt=datetime.date.today())


# In[37]:


instrument =API_TASKS['comdty']['instruments']
fields = API_TASKS['comdty']['fields']
comdty_data = pd.DataFrame()

for i in instrument : #API_TASKS.keys():   
    
    kwargs = {
#         'timeout': 1000,
#         'allow_missing': True,
#         'Fill':'NA'
    }
    comdty_data = pd.concat([comdty_data,
                             blp.bdh(
                                 tickers=i,
                                 flds=fields,
                                 start_date=start_date,
                                 end_date=end_date,
                                 **kwargs,
                             )], axis=1)
    
    print(i)
#     all_data=pd.concat([comdty_data,df[k]],axis=1)
print(f'All data = {comdty_data.shape}')
# all_data.shape


# In[62]:


# for k in [x for x in API_TASKS.keys() if x != 'currncy']:
#     instrument =API_TASKS[k]['instruments']
#     fields = API_TASKS[k]['fields']
#     print(k)
#     kwargs = {
# #         'timeout': 1000,
# #         'allow_missing': True,
# #         'Fill':'NA'
#     }
#     df[k] = blp.bdh(
#             tickers=instrument,
#             flds=fields,
#             start_date=start_date,
#             end_date=end_date,
#                 **kwargs,
#         )
#     print(instrument)
#     all_data=pd.concat([all_data,df[k]],axis=1)
#     print(f'All data = {all_data.shape}')
# all_data.shape
    


# In[38]:


comdty_data.head()


# In[39]:


comdty_data.to_csv('NewComdty_data.csv')
comdty_data.to_parquet('Newcomdty_data.pqt')


# In[63]:


{x[1] for x in df['currncy'].columns}  #loc[:,'ER1 Curncy']['FUT_CUR_GEN_TICKER'].unique()


# In[64]:


# {x[0] for x in df['curncy'].columns}
[df[x].shape for x in df.keys()]


# In[66]:


df['currncy'].T


# In[11]:


'a'+'.csv'


# In[67]:


all_data.to_csv('all_futures_data_all2.csv')
all_data.to_parquet('all_futures_data_all2.pqt')


# In[68]:


for k in df.keys():
    df[k].to_csv(k+'_curncy_all2.csv')
    df[k].to_parquet(k+'_curncy_all2.pqt')


# In[23]:


df.keys()


# In[24]:


len(df.keys())


# In[25]:


pwd


# In[26]:


all_data.shape


# In[27]:


df['ez.economics.data']


# In[ ]:




