import pandas as pd
import numpy as np


def preprocess_init(df_train, df_test, bool_features):
    """
    初始化預處理函數
    
    功能說明：
    1. 合併訓練集和測試集
    2. 創建時間週期特徵 (days)
    3. 處理布林型特徵
    4. 按照指定順序排序數據
    
    參數：
        df_train (DataFrame): 訓練集數據
        df_test (DataFrame): 測試集數據  
        bool_features (list): 需要處理的布林型特徵列表
    
    返回：
        DataFrame: 預處理後的合併數據集
    """
    
    # 合併訓練集和測試集，sort=True 確保列順序一致
    df = pd.concat([df_train, df_test], sort=True)

    # 創建時間週期特徵：根據授權日期 (locdt) 劃分為不同的時間段
    # 1-30天 -> 30, 31-60天 -> 60, 61-90天 -> 90, 91-120天 -> 120
    df['days'] = np.select([df['locdt']<=30, 
                           [(df['locdt']>30) & (df['locdt']<=60)], 
                           [(df['locdt']>60) & (df['locdt']<=90)], 
                           [(df['locdt']>90) & (df['locdt']<=120)]],
                          [30,60,90,120])[0]

    # 處理布林型特徵：將 'Y'/'N' 轉換為 1/0
    df = preprocess_bool(df, bool_features)

    # 按照業務邏輯排序：歸戶帳號 -> 交易卡號 -> 授權日期 -> 授權時間
    # 這樣排序有助於後續的時間序列特徵工程
    df = df.sort_values(by = ['bacno','cano','locdt','loctm']).reset_index(drop = True)

    return df
                
def preprocess_bool(df, bool_features):
    """
    布林型特徵預處理函數
    
    功能說明：
    將字符型的布林值 ('Y'/'N') 轉換為數值型 (1/0)
    這樣便於機器學習模型處理
    
    參數：
        df (DataFrame): 輸入數據框
        bool_features (list): 需要轉換的布林型特徵列表
                             包括：ecfg, flbmk, flg_3dsmk, insfg, ovrlt
    
    返回：
        DataFrame: 轉換後的數據框
    """
    
    for feature in bool_features:
        # 使用 np.select 進行條件轉換：'Y' -> 1, 'N' -> 0
        # 其他值保持不變（如 NaN）
        df[feature] = np.select([df[feature]=='Y',df[feature]=='N'],[1,0]) 
    
    return df
