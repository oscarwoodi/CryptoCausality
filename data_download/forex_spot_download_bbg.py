import datetime

# import dask.dataframe as dd
import numpy as np
import pandas as pd
from xbbg import blp

forex_spot_list =['SEKUSD','NOKUSD','HKDUSD','NZDUSD','AUDUSD','CADUSD','CHFUSD','GBPUSD','JPYUSD',
    'EURUSD','SEKEUR','NOKEUR','HKDEUR','NZDEUR','AUDEUR','CADEUR','CHFEUR','GBPEUR','JPYEUR','SEKJPY',
    'NOKJPY','HKDJPY','NZDJPY','AUDJPY','CADJPY','CHFJPY','GBPJPY','SEKGBP','NOKGBP','HKDGBP','NZDGBP','AUDGBP',
    'CADGBP','CHFGBP','SEKCHF','NOKCHF','HKDCHF','NZDCHF','AUDCHF','CADCHF','SEKCAD','NOKCAD','HKDCAD','NZDCAD',
    'AUDCAD','SEKAUD','NOKAUD','HKDAUD','NZDAUD','SEKNZD','NOKNZD','HKDNZD','SEKHKD','NOKHKD','SEKNOK']



forex_tn_list = [x+'TN' for x in forex_spot_list]
#
# ['EURGBP', 'EURJPY', 'EURCHF', 'PLNEUR', 'AUDNZD', 'NOKSEK', 'GBPCHF', 'EURCAD', 'EURCZK', 'EURHUF',
#                     'EURAUD', 'GBPJPY', 'CHFJPY', 'NZDJPY', 'EURNOK', 'AUDJPY', 'GBPNOK', 'EURSEK', 'CHFZAR', 'GBPCHF',
#                     'GBPJPY', 'PLNEUR', 'GBPCAD', 'CADJPY', 'CADJPY', 'AUDJPY', 'AUDNZD', 'EURZAR', 'HUFEUR', 'AUDCAD',
#                     'AUDJPY', 'AUDUSD', 'BRLUSD'],

Ticker_dict = {
    # US, Germany, Italy, France, GBP, Spain, JPY
    'curncy_spot': {
        'tickers': forex_spot_list,
        'numbers': [''],
        'fields': ['PX_LAST', 'PX_BID', 'PX_ASK', 'PX_MID', 'PX_HIGH', 'PX_LOW',"PX_OPEN"],
        'sector': 'Curncy'
    },
    'curncy_tn': {
        'tickers': forex_tn_list,
        'numbers': [''],
        'fields': ['PX_LAST', 'PX_BID', 'PX_ASK', 'PX_MID', 'PX_HIGH', 'PX_LOW',"PX_OPEN"],
        'sector': 'Curncy'
    },

}

# In[8]:


API_TASKS = {}
# use sets to ensure uniquenss
for k in Ticker_dict.keys():
    if len(Ticker_dict[k]['numbers']) == 0:
        Ticker_dict[k]['numbers'] = ['']
    sector = Ticker_dict[k]['sector']
    API_TASKS[k] = {
        'instruments': list(
            {x + y + ' ' + sector for y in Ticker_dict[k]['numbers'] for x in Ticker_dict[k]['tickers']}),
        'fields': Ticker_dict[k]['fields']}
    API_TASKS[k]['instruments'].sort()
    # 'Curncy'?

start_date = datetime.date(2000, 1, 1)
end_date = datetime.date.today()

# In[12]:
#### PULL FROM BBG


all_data = pd.DataFrame()
df = {}

all_data = pd.DataFrame()
df = dict()
data_set = dict()
for k in [x for x in API_TASKS.keys()]:
    instruments = list({x for x in API_TASKS[k]['instruments']})
    fields = API_TASKS[k]['fields']
    print(k)
    df[k] = {}
    data_set[k] = pd.DataFrame()
    for i in instruments:
        kwargs = {
            #         'timeout': 1000,
            #         'allow_missing': True,
            #         'Fill':'NA'
        }
        df[k][i] = blp.bdh(
            tickers=i,
            flds=fields,
            start_date=start_date,
            end_date=end_date,
            **kwargs,
        )

        data_set[k] = pd.concat([data_set[k], df[k][i]], axis=1)

    data_set[k].to_csv('~/Dropbox/FX_alt/' +f'combo_FOREX_170424_{k}.csv')
    data_set[k].to_parquet('~/Dropbox/FX_alt/' + f'combo_FOREX_170424_{k}.pqt')
    print(f'Data {k} = {data_set[k].shape}')
print('fin')
# all_data.shape


# In[31]:


index = data_set['index_spot'].copy()
index.columns = index.columns.swaplevel(0, 1)
index = index.loc[:, 'PX_LAST']
index.isna().sum().sort_values()

# In[67]:


all_data.to_csv('all_futures_data_all3.csv')
all_data.to_parquet('all_futures_data_all3.pqt')

# In[68]:

# Missing Currency Data
missing_tickers = ['BDR1 Curncy', 'BDR2 Curncy', 'EZ2 Curncy', 'HR2 Curncy',
                   'NJ2 Curncy', 'PLE2 Curncy', 'RP2 Curncy',
                   'YA2 Curncy',
                   'YF2 Curncy', 'ZI2 Curncy']

all_data = pd.DataFrame()
df = {}

all_data = pd.DataFrame()
df = dict()
data_set = dict()
for k in ['curncy']:  # x for x in API_TASKS.keys()]:
    instruments = list({x for x in API_TASKS[k]['instruments']})
    fields = API_TASKS[k]['fields']
    print(k)
    df[k] = {}
    data_set[k] = pd.DataFrame()
    for i in instruments:
        kwargs = {
            #         'timeout': 1000,
            #         'allow_missing': True,
            #         'Fill':'NA'
        }
        df[k][i] = blp.bdh(
            tickers=i,
            flds=fields,
            start_date=start_date,
            end_date=end_date,
            **kwargs,
        )
        if i in missing_tickers:
            print(i)
            print(df[k][i].head())

        data_set[k] = pd.concat([data_set[k], df[k][i]], axis=1)

    data_set[k].to_csv(f'combo_data_220224_{k}.csv')
    data_set[k].to_parquet(f'combo_data_220224_{k}.pqt')
    print(f'Data {k} = {data_set[k].shape}')
print('fin')
# all_data.shape


for k in df.keys():
    df[k].to_csv(k + '_curncy_all2.csv')
    df[k].to_parquet(k + '_curncy_all2.pqt')
