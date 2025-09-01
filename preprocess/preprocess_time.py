import pandas as pd
import numpy as np
from preprocess.util import generic_groupby
    
def preprocess_time(df):
    """
    時間特徵預處理函數
    
    功能說明：
    創建基於時間的特徵，用於捕捉交易的時間模式和異常行為
    時間特徵在詐騙檢測中非常重要，因為詐騙交易往往有特殊的時間模式
    
    生成的特徵：
    1. global_time: 全局時間戳 (結合日期和時間的絕對時間)
    2. last_time_days: 與前一筆交易的時間間隔 (同卡號同週期內)
    3. next_time_days: 與後一筆交易的時間間隔 (同卡號同週期內)  
    4. cano_locdt_global_time_std: 同卡號同日期內交易時間的標準差
    
    業務意義：
    - 交易時間間隔異常可能表示盜刷行為
    - 同一天內交易時間分散度反映消費習慣
    - 全局時間戳便於進行時間序列分析
    
    參數：
        df (DataFrame): 輸入的交易數據，必須包含以下欄位：
                       - loctm: 授權時間 (原始格式)
                       - locdt: 授權日期
                       - cano: 交易卡號
                       - days: 時間週期
    
    返回：
        DataFrame: 添加了時間特徵的數據框
    """

    # 創建全局時間戳：將日期和時間合併為統一的時間戳
    df['global_time'] = loctm_to_global_time(df)
    
    # 計算與前一筆交易的時間間隔 (按卡號和週期分組)
    # 正值表示當前交易晚於前一筆交易的秒數
    df['last_time_days'] = df.groupby(['cano','days'])['global_time'].diff(periods = 1)
    
    # 計算與後一筆交易的時間間隔 (按卡號和週期分組)  
    # 負值表示當前交易早於後一筆交易的秒數
    df['next_time_days'] = df.groupby(['cano','days'])['global_time'].diff(periods = -1)

    # 計算同卡號同日期內交易時間的標準差
    # 標準差大表示交易時間分散，標準差小表示交易時間集中
    groups = ['cano','locdt']
    feature = 'global_time'
    agg_list = [np.std]
    df = generic_groupby(df, groups, feature, agg_list)
    
    return df

def loctm_to_global_time(df):
    """
    時間格式轉換函數
    
    功能說明：
    將原始的授權時間 (loctm) 轉換為全局時間戳
    結合授權日期 (locdt) 創建統一的時間表示方式
    
    轉換過程：
    1. 解析原始時間格式，提取時、分、秒
    2. 將時間轉換為秒數
    3. 結合日期創建全局時間戳 (以秒為單位)
    
    時間格式說明：
    - 原始格式：HHMMSSXX (最後兩位可能是毫秒或其他標識)
    - 目標格式：全局秒數 = 日期秒數 + 當日秒數
    
    業務價值：
    - 統一時間表示，便於時間序列分析
    - 支持跨日期的時間間隔計算
    - 為時間相關特徵工程提供基礎
    
    參數：
        df (DataFrame): 包含 loctm (授權時間) 和 locdt (授權日期) 的數據框
    
    返回：
        Series: 全局時間戳序列 (以秒為單位)
    """
    
    # 創建數據副本，避免修改原始數據
    df = df.copy()
    
    # 將授權時間轉換為字符串格式
    df['loctm'] = df['loctm'].astype(str)
    
    # 移除最後兩位數字 (可能是毫秒或其他標識)
    df['loctm'] = df['loctm'].str[:-2]
    
    # 提取小時部分 (倒數第6-4位)
    df['hours'] = df['loctm'].str[-6:-4]
    df['hours'] = np.where(df['hours']=='', '0', df['hours']).astype(int)
    
    # 提取分鐘部分 (倒數第4-2位)
    df['minutes'] = df['loctm'].str[-4:-2]
    df['minutes'] = np.where(df['minutes']=='', '0', df['minutes']).astype(int)
    
    # 提取秒數部分 (最後2位)
    df['second'] = df['loctm'].str[-2:].astype(int)
    
    # 將時間轉換為當日總秒數
    df['loctm'] = df['hours']*60*60 + df['minutes']*60 + df['second']
    
    # 創建全局時間戳：日期秒數 + 當日秒數
    # locdt * 24*60*60 將日期轉換為秒數 (假設 locdt 是從某個起始點開始的天數)
    df['global_time'] = df['locdt']*24*60*60 + df['hours']*60*60+df['minutes']*60+df['second']
                        
    return df['global_time']