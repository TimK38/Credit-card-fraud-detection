import pandas as pd
import numpy as np

def preprocess_special_features(df):

    df = diff_with_first_fraud_locdt(df) 
    df = black_white_list(df) 

    return df

def black_white_list(df):

    df[['mchno','conam']] = df[['mchno','conam']].astype(str)
    df['normal_mchno'] = df.apply(lambda x : x['mchno'] if x['fraud_ind']==0 else -999,axis = 1)
    df['fraud_mchno'] = df.apply(lambda x : x['mchno'] if x['fraud_ind']==1 else -999,axis = 1)
    df['fraud_conam'] = df.apply(lambda x : x['conam'] if x['fraud_ind']==1 else -999,axis = 1)

    rolling_list = ['normal_mchno',
                    'fraud_mchno',
                    'fraud_conam'
                     ]

    for feature in rolling_list:
        # 創建列表形式的特徵
        df[feature] = df[feature].apply(lambda x : [x])
        
        # 使用更穩健的方法來計算累積和
        # 先重置索引確保連續性
        df_temp = df.reset_index(drop=True)
        rolling_result = df_temp.groupby('bacno')[feature].apply(lambda x : x.cumsum()).reset_index(level=0, drop=True)
        
        # 確保索引對齊
        df['rolling_{}'.format(feature)] = rolling_result.reindex(df.index)

        df_tem = df.drop_duplicates(subset = ['cano','locdt'],keep = 'last')
        df_tem['last_rolling_{}_cano'.format(feature)] = df_tem.groupby(['cano'])['rolling_{}'.format(feature)].shift(1)
        df = pd.merge(df, df_tem[['cano','locdt','last_rolling_{}_cano'.format(feature)]], how = 'left', on = ['cano','locdt'])
        df['last_rolling_{}_cano'.format(feature)] = df['last_rolling_{}_cano'.format(feature)].fillna('NA')

        # 安全地檢查元素是否在列表中
        def check_in_list(row):
            try:
                feature_value = row[feature[-5:]]
                rolling_list_value = row['last_rolling_{}_cano'.format(feature)]
                if rolling_list_value == 'NA' or pd.isna(rolling_list_value):
                    return 0
                return 1 if feature_value in rolling_list_value else 0
            except:
                return 0
        
        df['{}_in_{}_list'.format(feature[-5:],feature)] = df.apply(check_in_list, axis=1)        

    df['conam'] = df['conam'].astype(float)

    return df

def diff_with_first_fraud_locdt(df):

    df_fraud = df[df['fraud_ind']==1].drop_duplicates(subset = ['cano'],keep = 'first')
    df_fraud = df_fraud.rename(columns = {'locdt':'first_fraud_locdt'})
    df = pd.merge(df, df_fraud[['cano','first_fraud_locdt']], how = 'left', on = ['cano'])
    df['diff_with_first_fraud_locdt'] = df['locdt'] - df['first_fraud_locdt']
    df['diff_with_first_fraud_locdt'] = np.where(df['diff_with_first_fraud_locdt']<=0, np.nan, df['diff_with_first_fraud_locdt'])

    return df
