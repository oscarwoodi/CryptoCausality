import datetime

#import dask.dataframe as dd
import numpy as np
import pandas as pd
from xbbg import blp



# fx_spot_dict={
#     'RP':'EURGBP','RY':	'EURJPY','RF':	'EURCHF','PLE':	'PLNEUR','AN':	'AUDNZD','NJ':	'NOKSEK','PS':	'GBPCHF',
#     'CA':	'EURCAD','EZ':	'EURCZK','HR':	'EURHUF','EA':	'EURAUD','PJ':	'GBPJPY','SJ':	'CHFJPY','ZQ':	'NZDJPY',
#     'EN':	'EURNOK','AJ':	'AUDJPY','ZK':	'GBPNOK','EW':	'EURSEK','SWR':	'CHFZAR','YF':	'GBPCHF','YK':	'GBPJPY',
#     'EI':	'PLNEUR','ZI':	'GBPCAD','CY':	'CADJPY','HL':	'CADJPY','YA':	'AUDJPY','AR':	'AUDNZD','ZD':	'EURZAR',
#     'EL':	'HUFEUR','AC':	'AUDCAD','XAY':	'AUDJPY','ASD':	'AUDUSD','BSR':	'BRLUSD',
# }
fx_spot_dict ={'EC':'EURUSD','JY':'JPYUSD','PE':'MXNUSD','BP':'GBPUSD','AD':'AUDUSD','CD':'CADUSD','BR':'BRLUSD','SF':'CHFUSD',
                'NV':'NZDUSD','RA':'ZARUSD','EE':'EURUSD','CHY':'CHYUSD','SIR':'INRUSD','NO':'NOKUSD','PP':'PLNUSD','SE':'SEKUSD',
                'KO':'KRWUSD','HE':'HUFUSD','TLC':'TRYUSD','IS':'ILSUSD','CC':'CZKUSD','CHL':'USDCLP'}



fx_spot_list = list(fx_spot_dict.values())
fx_currncy_list = list(fx_spot_dict.keys())

equities_spot_dict={    'XB': 'IBOV','ES': 'SPX','HWB': 'NDX','NO': 'NKY','HW': 'SPX','VG': 'SX5E','NQ': 'NDX','XU': 'XIN9I','KM': 'KOSPI2',
     'IFD': 'CSI1000','AF': 'NSEBANK','NZ': 'NIFTY','RTY': 'RTY','CA': 'SX7E','MX': 'TWSE','JAI': 'NKY','BC': 'SET50',
    'HC': 'HSCEI','FFD': 'SH000905','IFB': 'SHSN300','0J': 'SX5ED','UX': 'VIX','HI': 'HSI','A5': 'XU030','DM': 'INDU',
    'VNC': 'VN30','HU': 'HSI','FT': 'TWSE','HWR': 'RTY','KST': 'KOSDQ150','BZ': 'IBOV','KMS': 'KOSPI2','FFB': 'SSE50',
    'HCT': 'HSTECH','SXO': 'SXXP','HWI': 'INDU','MES': 'MXEF','FNY': 'NKY','FDO': 'DJI','Z ': 'UKX','QC': 'OMX','STE': 'SPX',
    'VE': 'RTSI$','GX': 'DAX','TWT': 'FTCRTWRP','NK': 'NKY','JGS': 'NIFTY','TP': 'TPX','QZ': 'SIMSCI','XP': 'AS51','CF': 'CAC',
    'KRS': 'WIG20','MCY': 'MXCNA50C','DFW': 'DAX','FVS': 'V2X','MFS': 'MXEA','FNP': 'NDX','WSP': 'SPX','NI': 'NKY',
    'DED': 'SX5ED','NH': 'NKY','XMB': 'IMOEX','SM': 'SMI','FZY': 'KOSPI2','PT': 'SPTSX60','IUI': 'SX7E','VXT': 'VIX',
    'MZS': 'DAX','XMC': 'IMOEX','TMI': 'TPX','AI': 'TOP40','VHO': 'SX5E','ST': 'FTSEMIB','AXR': 'SPXT','AXW': 'SPXT',
    'TFS': 'MXEA','FA': 'MID','EO': 'AEX','MHC': 'HSCEI','VEY': 'RTSI$','IB': 'IBEX','FXY': 'KOSPI2','ASD': 'SPXDIVAN',
    'NX': 'NKY','RNS': 'NMIDSELP','IK': 'FBMKLCI','QNT': 'NDX','MRO': 'TSEMOTHR','JPW': 'JPNK400','VXL': 'VIX','DCP': 'J430PR',
    'DBE': 'SX7EDA','FNS': 'NYFANG','RFC': 'XIN0I','ZWP': 'M1WO','SX': 'SXAP','HRT': 'MXIN','KG': 'SXEP','DJE': 'DJUSRE'}
eq_spot_dict2= {'XB': 'IBOV','ES': 'SPX','HWB': 'NDX','NO': 'NKY','HW': 'SPX','VG': 'SX5E',
    'NQ': 'NDX','XU': 'XIN9I','KM': 'KOSPI2','IFD': 'CSI1000','AF': 'NSEBANK',  'NZ': 'NIFTY',
    'RTY': 'RTY','CA': 'SX7E','MX': 'TWSE','JAI': 'NKY','BC': 'SET50','HC': 'HSCEI','FFD': 'SH000905',
    'IFB': 'SHSN300','0J': 'SX5ED','UX': 'VIX','HI': 'HSI','A5': 'XU030','DM': 'INDU','VNC': 'VN30',
    'HU': 'HSI','FT': 'TWSE','HWR': 'RTY','KST': 'KOSDQ150','BZ': 'IBOV','KMS': 'KOSPI2','FFB': 'SSE50',
    'HCT': 'HSTECH','SXO': 'SXXP','HWI': 'INDU','MES': 'MXEF','FNY': 'NKY','FDO': 'DJI','Z ': 'UKX',
    'QC': 'OMX','STE': 'SPX','VE': 'RTSI$','GX': 'DAX','TWT': 'FTCRTWRP','NK': 'NKY','JGS': 'NIFTY',
    'TP': 'TPX','QZ': 'SIMSCI','XP': 'AS51','CF': 'CAC','KRS': 'WIG20','MCY': 'MXCNA50C','DFW': 'DAX',
    'FVS': 'V2X','MFS': 'MXEA','FNP': 'NDX','WSP': 'SPX','NI': 'NKY','DED': 'SX5ED','NH': 'NKY',
    'XMB': 'IMOEX','SM': 'SMI','FZY': 'KOSPI2','PT': 'SPTSX60','IUI': 'SX7E','VXT': 'VIX',
    'MZS': 'DAX',   'XMC': 'IMOEX','TMI': 'TPX','AI': 'TOP40','VHO': 'SX5E','ST': 'FTSEMIB',
    'AXR': 'SPXT','AXW': 'SPXT','TFS': 'MXEA','FA': 'MID','EO': 'AEX','MHC': 'HSCEI','VEY': 'RTSI$',
    'IB': 'IBEX','FXY': 'KOSPI2','ASD': 'SPXDIVAN','NX': 'NKY','RNS': 'NMIDSELP','IK': 'FBMKLCI',
    'QNT': 'NDX','MRO': 'TSEMOTHR','JPW': 'JPNK400','VXL': 'VIX','DCP': 'J430PR',
    'DBE': 'SX7EDA','FNS': 'NYFANG','RFC': 'XIN0I','ZWP': 'M1WO','SX': 'SXAP','HRT': 'MXIN','KG': 'SXEP',

                }

equities_spot_dict.update(eq_spot_dict2)


# In[5]:


stir_spot_dict ={ 'SFR': 'SOFRRATE',
                  'ER': 'EUR003M',
                  'SFI': 'SONIO/N',
                  'FF': 'FEDL01',
                  'IR': 'BBSW3M',
                  'SER': 'SOFRRATE','COR': 'CAONREPO',
                  'BA': 'CDOR03', # CDRO CAD Bankers Acceptance
                  'SSY': 'SRFXON3',
                  'RA': 'SKF30001',#SW 3M STIBOR FRA
                  'ZB': '',  #NZFMA Bill Fixes (Subscription!!! )
                  'IB': 'RBACOR',
                  'KTR': 'ESTRON',
                  'ORI': '', # OMX RIBA SEK FIUTUREs. Don't understand
                  'JO': 'MUTKCALM',
                  'TKY': 'ESTRON',
                  'FP': 'EUR003M',
                  'JDB': 'ESTRON',
                  'TZR': 'USB3MTA','YPO': 'MUTKCALM','KUS': 'ESTRON','BSB': 'BSBY3M',
                  'INR': '',
                  'SZ': 'SRFXON3','SRL': 'SOFRRATE','ONS': 'SONIO/N',
}


# In[6]:



Ticker_dict={
    # US, Germany, Italy, France, GBP, Spain, JPY
    'curncy_spot':{
        'tickers': fx_spot_list,

            # ['EURGBP', 'EURJPY', 'EURCHF', 'PLNEUR', 'AUDNZD', 'NOKSEK', 'GBPCHF', 'EURCAD', 'EURCZK', 'EURHUF',
            #          'EURAUD', 'GBPJPY', 'CHFJPY', 'NZDJPY', 'EURNOK', 'AUDJPY', 'GBPNOK', 'EURSEK', 'CHFZAR', 'GBPCHF',
            #          'GBPJPY', 'PLNEUR', 'GBPCAD', 'CADJPY', 'CADJPY', 'AUDJPY', 'AUDNZD', 'EURZAR', 'HUFEUR', 'AUDCAD',
            #          'AUDJPY', 'AUDUSD', 'BRLUSD'],
        'numbers':[''],
        'fields':['PX_LAST','PX_BID','PX_ASK','PX_MID','PX_HIGH','PX_LOW', 'PX_OPEN', 'VOLUME'],
        'sector':'Curncy'
    },
    'index_spot':{
        'tickers':list({ 'IBOV', 'SPX', 'NDX', 'NKY', 'SPX', 'SX5E', 'NDX', 'XIN9I', 'KOSPI2', 'CSI1000',
                 'NSEBANK', 'NIFTY', 'RTY', 'SX7E', 'TWSE', 'NKY', 'SET50', 'HSCEI', 'SH000905', 'SHSN300',
                 'SX5ED', 'VIX', 'HSI', 'XU030', 'INDU', 'VN30', 'HSI', 'TWSE', 'RTY', 'KOSDQ150', 'IBOV', 'KOSPI2',
                 'SSE50', 'HSTECH', 'SXXP', 'INDU', 'MXEF', 'NKY', 'DJI', 'UKX', 'OMX', 'SPX', 'RTSI$', 'DAX', 'FTCRTWRP',
                 'NKY', 'NIFTY', 'TPX', 'SIMSCI', 'AS51', 'CAC', 'WIG20', 'MXCNA50C', 'DAX', 'V2X', 'MXEA', 'NDX', 'SPX',
                 'NKY', 'SX5ED', 'NKY', 'IMOEX', 'SMI', 'KOSPI2', 'SPTSX60', 'SX7E', 'VIX', 'DAX', 'IMOEX', 'TPX', 'TOP40',
                 'SX5E', 'FTSEMIB', 'SPXT', 'SPXT', 'MXEA', 'MID', 'AEX', 'HSCEI', 'RTSI$', 'IBEX', 'KOSPI2', 'SPXDIVAN',
                 'NKY', 'NMIDSELP', 'FBMKLCI', 'NDX', 'TSEMOTHR', 'JPNK400', 'VIX', 'J430PR', 'SX7EDA', 'NYFANG', 'XIN0I',
                 'M1WO', 'SXAP', 'MXIN', 'SXEP', 'DJUSRE'}),
          'numbers':[''],
        'fields':['PX_LAST','PX_MID','PX_HIGH','PX_LOW','PX_OPEN'], # no bid / ask prices for index
        'sector':'Index'
    },
        'stir_spot':{
        'tickers':list({ 'SOFRRATE', 'EUR003M', 'SONIO/N', 'FEDL01', 'BBSW3M', 'CAONREPO',
                          'SRFXON3',  'ESTRON', 'MUTKCALM', 'USB3MTA',
                              'BSBY3M', 'SRFXON3'}),
            
          'numbers':[''],
        'fields':['PX_LAST','PX_BID','PX_ASK','PX_MID','PX_HIGH','PX_LOW','PX_OPEN'],
        'sector':'Index'
    },
    
    'curncy':{
        'tickers':  fx_currncy_list,

            # list({'RP','RY','RF','PLE','AN','NJ','PS','CA','EZ','HR','EA','PJ','SJ','ZQ','EN','AJ',
            #     'ZK', 'SWR','YF','YK','EI','BDR','ZI','CY','HL','YA','AR','ZD','EL','AC','XAY','ASD','BSR'
            #              }),

        'numbers':['1','2'],
        'fields':['PX_LAST','PX_BID','PX_ASK','PX_MID','PX_HIGH','PX_LOW', 'PX_OPEN',
                  'FUT_NOTICE_FIRST','FUT_DLV_DT_LAST','FUT_DLV_DT_FIRST',
            'FUT_CUR_GEN_TICKER','VOLUME','OPEN_INT'],
        'sector':'Curncy'
    },
    'index': {
        'tickers':list({
            'XB','ES','HWB','NO','HW','VG','NQ','XU','KM','IFD','AF','NZ','RTY','CA','MX','JAI','BC','HC','FFD','IFB',
            '0J','UX','HI','A5','DM','VNC','HU','FT','HWR','KST','BZ','KMS','FFB','HCT','SXO','HWI','MES','FNY','FDO',
            'Z ','QC','STE','VE','GX','TWT','NK','JGS','TP','QZ','XP','CF','KRS','MCY','DFW','FVS','MFS','FNP','WSP',

        }),
        'numbers':['1','2'],
        'fields':['PX_LAST','PX_BID','PX_ASK','PX_MID','PX_HIGH','PX_LOW','PX_OPEN',
                  'FUT_NOTICE_FIRST','FUT_DLV_DT_LAST','FUT_DLV_DT_FIRST',
            'FUT_CUR_GEN_TICKER','VOLUME','OPEN_INT'],
        'sector':'Index'
        },
    'bonds' :{
        'tickers':list({'WN','TWE','US','UXY','TY','FV','3Y','TU',  # Us ultra, 20y, Long, 10y Ultra, 10,5,3,2
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
                      }),
        # from WBF = CN = CN10y,
        'numbers': ['1', '2'],
        'fields': ['PX_LAST', 'PX_BID', 'PX_ASK', 'PX_MID', 'PX_HIGH','PX_LOW','PX_OPEN',
                   'FUT_NOTICE_FIRST', 'FUT_DLV_DT_LAST', 'FUT_DLV_DT_FIRST',
                   'FUT_CUR_GEN_TICKER', 'VOLUME', 'OPEN_INT',  # and some bond specific ones
                   'FUT_IMPLIED_REPO_RT',
                   'FUT_CTD_CNV_YIELD',  # yield of CTD computed according to bonds conventional basis`
                   'FUT_CTD_CPN',
                   'CTD_FORWARD_YTM_LAST',
                   'FUT_CTD_NET_BASIS',  # gross basis adjusted for net carry
                   'FUT_CTD_GROSS_BASIS',  # CTD price - delivery price
                   'FUT_NOTL_CNV_YIELD'  # conventional yield of the 'notional' bond for the future
                   ],
        'sector':'Curncy'
       },


    'stirt':{
        # need to find fixings for first contract carry
        'tickers':list({

        'SFR','ER','SFI','FF','IR','SER','COR','BA','SSY','RA','ZB','IB','KTR','ORI','JO','TKY','FP','JDB',
        'TZR','YPO','KUS','BSB','INR','SZ','SRL','ONS',
        'ES','YE', 'L ', 'ED'}), # some just for historics
        
        
        'numbers':[str(x) for x in [1,2,3,4]], # whites reds, greens,  NO blues or golds
        'fields':['PX_LAST','PX_BID','PX_ASK','PX_MID','PX_HIGH', 'PX_LOW','PX_OPEN',
                  'FUT_NOTICE_FIRST','FUT_DLV_DT_LAST','FUT_DLV_DT_FIRST',
             'FUT_CUR_GEN_TICKER','VOLUME','OPEN_INT'],
        'sector':'Curncy'
        },
    'comdty':{
        'tickers': list({
            'C ','AC','DCS','CRD','EP','CT', 'CA','W ','KW','MW','EB','XA','IAC','MFI','XW', 'GC','SI','PL','PA','LA',
            'HG','LX','LL','LN','ALE','LT', 'JN','LBO','PGP','UFD','PCW','QS','HO','XB','PG','NV','QST','BAP','Z0',
            'NG','FN','PNG'
        }),
           'numbers':['1','2','3'],
        'fields': ['PX_LAST', 'PX_BID', 'PX_ASK', 'PX_MID', 'PX_HIGH', 'PX_LOW', 'PX_OPEN',
                   'FUT_NOTICE_FIRST', 'FUT_DLV_DT_LAST', 'FUT_DLV_DT_FIRST',
                   'FUT_CUR_GEN_TICKER', 'VOLUME', 'OPEN_INT'],

        # 'fields':['PX_LAST','PX_BID','PX_ASK','PX_MID','PX_OPEN',
        #                   'FUT_NOTICE_FIRST','FUT_DLV_DT_LAST','FUT_DLV_DT_FIRST',
        #      'FUT_CUR_GEN_TICKER','VOLUME','OPEN_INT'],
        'sector':'Comdty'
    }
}


# In[8]:


API_TASKS={}
# use sets to ensure uniquenss
for k in Ticker_dict.keys():
    if len(Ticker_dict[k]['numbers']) == 0:
        Ticker_dict[k]['numbers'] = ['']
    sector = Ticker_dict[k]['sector']
    API_TASKS[k] = {
    'instruments':list({x+y+' '+ sector for y in Ticker_dict[k]['numbers'] for x in Ticker_dict[k]['tickers'] }),
    'fields' : Ticker_dict[k]['fields']}
    API_TASKS[k]['instruments'].sort()
    # 'Curncy'?



start_date=datetime.date(2000,1,1)
end_date = datetime.date.today()


# In[12]:
#### PULL FROM BBG


# all_data = pd.DataFrame()
# df = {}
#
#
# all_data = pd.DataFrame()
# df =dict()
# data_set = dict()
# for k in [x for x in API_TASKS.keys()]: #API_TASKS.keys()]:
#     instruments =list({x for x in API_TASKS[k]['instruments']})
#     fields = API_TASKS[k]['fields']
#     print(k)
#     df[k] = {}
#     data_set[k] = pd.DataFrame()
#     for i in instruments:
#         kwargs = {
#                 #         'timeout': 1000,
#                 #         'allow_missing': True,
#                 #         'Fill':'NA'
#         }
#         df[k][i] = blp.bdh(
#                 tickers=i,
#                 flds=fields,
#                 start_date=start_date,
#                 end_date=end_date,
#                     **kwargs,
#                 )
#
#         data_set[k]=pd.concat([data_set[k],df[k][i]],axis=1)
#
#     data_set[k].to_csv('~/Dropbox/CCA/futures_data_raw/' + f'combo_data_170424_{k}_new.csv')
#     data_set[k].to_parquet('~/Dropbox/CCA/futures_data_raw/' + f'combo_data_170424_{k}_new.pqt')
#     print(f'Data {k} = {data_set[k].shape}')
# print('fin')
# # all_data.shape


# PULL UP INFO ABOUT EACH TICKER
# get FUT_CUR_GEN_TICKER and get NAME from that!

desc_data = pd.DataFrame()
df_desc =dict()
desc_data_set = dict()

for k in [x for x in API_TASKS.keys() if x not in ['curncy_spot','index_spot','stir_spot']]:  # API_TASKS.keys()]:
    instruments = list({x for x in API_TASKS[k]['instruments']})
    fields = ["SHORT_NAME", "LONG_NAME", "EXCH_CODE", "FUT_EXCH_NAME_LONG",
              "FUT_CONT_SIZE", "FUT_TICK_SIZE","FUT_TICK_VAL","FUT_VAL_PT", "FUT_CUR_GEN_TICKER"]
    print(k)
    df_desc[k] = {}
    desc_data_set[k] = pd.DataFrame()
    for i in instruments:
        kwargs = {
            #         'timeout': 1000,
            #         'allow_missing': True,
            #         'Fill':'NA'
        }
        df_desc[k][i] = blp.bdp(
            tickers=i,
            flds=fields,
            **kwargs,
        )
        if df_desc[k][i].empty:
            continue
        for x in [x.lower() for x in fields]:
            if x not in df_desc[k][i].columns:
                df_desc[k][i][x] = np.nan
            if df_desc[k][i].loc[:,x].empty:
                df_desc[k][i][x] = np.nan

        # if 'fut_cur_gen_ticker' in df_desc[k][i].columns:
        flds = ['NAME', 'SHORT_NAME', 'FUT_INIT_SPEC_ML', 'FUT_SEC_SPEC_ML', 'UNDL_SPOT_TICKER', 'TIME_ZONE_NUM',
                'TRADING_DAY_START_TIME_EOD', 'TRADING_DAY_END_TIME_EOD',
                    'DERIVATIVE_DELIVERY_TYPE', 'NOTIONAL_CURRENCY_1']  #,   'EXCHANGE_TRADING_SESSION_HOURS']  # messes it up,'FUT_TRADING_HRS']
        if not df_desc[k][i].loc[:,'fut_cur_gen_ticker'].isna().iloc[0]:
            non_gen_ticker = df_desc[k][i].loc[:,'fut_cur_gen_ticker'].iloc[0]
            bloomberg_sector = i.split(' ')[1]
            non_gen_ticker = non_gen_ticker +' ' + bloomberg_sector
            non_gen_fields = ''
            extended_fields = blp.bdp(
                tickers=non_gen_ticker,
                flds = flds
            )
            extended_fields = extended_fields.rename(columns={'name':'fut_name',
                                                              'short_name':'fut_short_name'})
            extended_fields.index = df_desc[k][i].index

        else:
            extended_fields = pd.DataFrame(index=df_desc[k][i].index,
                                           columns = [x.lower() for x in flds])
        df_desc[k][i] = pd.concat([df_desc[k][i], extended_fields], axis=1)
        df_desc[k][i].loc[i,'sector'] = k


total=pd.DataFrame()
for kk in df_desc.keys():
    total = pd.concat([total,pd.concat(df_desc[kk], axis=0) ] ,axis=0)


total.to_csv('~/Dropbox/CCA/futures_data_raw/combo_data_futures_descriptions4.csv')
print('fin')
# all_data.shape


#
# # In[31]:
#
#
# index = data_set['index_spot'].copy()
# index.columns = index.columns.swaplevel(0,1)
# index = index.loc[:,'PX_LAST']
# index.isna().sum().sort_values()
#
#
# # In[67]:
#
#
# all_data.to_csv('all_futures_data_all2.csv')
# all_data.to_parquet('all_futures_data_all2.pqt')
#
#
# # In[68]:
#
# # Missing Currency Data
# missing_tickers = ['BDR1 Curncy','BDR2 Curncy','EZ2 Curncy','HR2 Curncy',
#                    'NJ2 Curncy', 'PLE2 Curncy', 'RP2 Curncy',
#                    'YA2 Curncy',
#                    'YF2 Curncy', 'ZI2 Curncy']
#
#
#
# all_data = pd.DataFrame()
# df = {}
#
# all_data = pd.DataFrame()
# df = dict()
# data_set = dict()
# for k in ['curncy']: #x for x in API_TASKS.keys()]:
#     instruments = list({x for x in API_TASKS[k]['instruments']})
#     fields = API_TASKS[k]['fields']
#     print(k)
#     df[k] = {}
#     data_set[k] = pd.DataFrame()
#     for i in instruments:
#         kwargs = {
#             #         'timeout': 1000,
#             #         'allow_missing': True,
#             #         'Fill':'NA'
#         }
#         df[k][i] = blp.bdh(
#             tickers=i,
#             flds=fields,
#             start_date=start_date,
#             end_date=end_date,
#             **kwargs,
#         )
#         if i in missing_tickers:
#             print(i)
#             print(df[k][i].head())
#
#         data_set[k] = pd.concat([data_set[k], df[k][i]], axis=1)
#
#     data_set[k].to_csv('~/Dropbox/CCA/Futures_data_raw/'+f'combo_data_280224_{k}.csv')
#     data_set[k].to_parquet('~/Dropbox/CCA/Futures_data_raw/'+ f'combo_data_280224_{k}.pqt')
#     print(f'Data {k} = {data_set[k].shape}')
# print('fin')
# # all_data.shape
#
#
# # for k in df.keys():
# #     df[k].to_csv(k+'_curncy_all2.csv')
# #     df[k].to_parquet(k+'_curncy_all2.pqt')
#
#
#
#
# #         [ EXTRA CURRENCY FUTURES TICKERS
# #             'RP','RY','RF','PLE','AN','NJ','PS','CA','EZ','HR','EA','PJ','SJ','ZQ','EN','AJ','ZK','KRP','EW','SWR',
# #             'KOL','YF','YK','EI','KRW','KGB','BDR','ZI','CY','KZY','KEL','HL','YA','AR','ZD','KR','EL','AC','AS','AC',
# #             'AS','XAY','AJ','XAYX','YAY','AN','AR','ASD','KA','BSR'
# #         ],
# # EXTRA INDEX FUTURES TICKERS
# #             'NI','DED','NH','XMB','SM','FZY','PT','IUI','VXT','MZS','XMC','TMI','AI','VHO','ST','AXR','AXW','TFS','FA',
# #             'EO','MHC','VEY','IB','FXY','ASD','NX','RNS','IK','QNT','MRO','JPW','VXL','DCP','DBE','FNS','RFC','ZWP',
# #             'SX','HRT','KG'
# # OTHER STIRT FUTURES - some onyl historic
# #             'SFR','ER','SFI','FF','IR','SER','COR','BA','SSY','RA','ZB','IB','KTR','ORI','JO','TKY',
# #             'ED', 'L ',  'ES', 'YE'  HISTORICS ONLY
# #         ],  # some included for history
#
#
#
# # In[16]:
