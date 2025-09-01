import pandas as pd
import numpy as np
from preprocess.util import generic_groupby
from preprocess.util import applyParallel


def preprocess_conam(df):
    """
    交易金額特徵預處理主函數
    
    功能說明：
    對交易金額 (conam) 進行全面的特徵工程，創建與金額相關的統計特徵
    交易金額是詐騙檢測中最重要的特徵之一，異常的金額模式往往是詐騙的重要指標
    
    處理流程：
    1. 計算同卡號同日期內的金額統計特徵 (最大值、最小值)
    2. 分析零金額交易的時間模式特徵
    
    業務價值：
    - 金額異常檢測：識別異常高額或異常低額的交易
    - 零金額交易分析：零金額交易往往有特殊的業務含義
    - 同日金額模式：分析用戶在同一天的消費行為模式
    
    詐騙檢測意義：
    - 盜刷者可能會先進行小額測試交易
    - 異常高額交易需要特別關注
    - 零金額交易可能是系統測試或異常操作
    
    參數：
        df (DataFrame): 輸入的交易數據，必須包含以下欄位：
                       - conam: 交易金額-台幣 (經過轉換)
                       - cano: 交易卡號
                       - locdt: 授權日期
                       - global_time: 全局時間戳 (由 preprocess_time 創建)
    
    返回：
        DataFrame: 添加了交易金額相關特徵的數據框
    """

    # 步驟1: 計算同卡號同日期內的金額統計特徵
    df = preprocess_global_conam_max_min(df)
    
    # 步驟2: 分析零金額交易的時間模式
    df = diff_with_zero_conam_time(df)

    return df

def diff_with_zero_conam_time(df):
    """
    零金額交易時間差異分析函數
    
    功能說明：
    分析每筆交易與同卡號同日期內零金額交易的時間差異
    零金額交易在信用卡系統中有特殊含義，可能表示：
    - 授權測試交易
    - 預授權交易
    - 系統驗證交易
    - 異常或錯誤交易
    
    特徵創建邏輯：
    1. 識別所有零金額交易 (conam == 0)
    2. 提取每個卡號每日的首次零金額交易時間
    3. 計算其他交易與零金額交易的時間差異
    
    業務意義：
    - 零金額交易往往是後續正常交易的前置操作
    - 時間差異可以反映交易的關聯性
    - 異常的零金額交易模式可能表示系統攻擊
    
    詐騙檢測價值：
    - 盜刷者可能先進行零金額測試
    - 異常的零金額交易時間模式需要關注
    - 與零金額交易的時間關係可作為風險指標
    
    參數：
        df (DataFrame): 包含交易數據的數據框，需要有：
                       - conam: 交易金額
                       - cano: 交易卡號
                       - locdt: 授權日期
                       - global_time: 全局時間戳
    
    返回：
        DataFrame: 添加零金額交易時間差異特徵的數據框
                  新增特徵：diff_gtime_with_conam_zero_trans_locdt
    """

    # 創建數據副本，避免修改原始數據
    df = df.copy()
    
    # 步驟1: 篩選零金額交易，並去除重複
    # 每個卡號每日只保留第一筆零金額交易 (keep='first')
    df_tem = df[df['conam']==0].drop_duplicates(subset = ['cano','locdt'],keep = 'first')
    
    # 步驟2: 重新命名時間欄位，使其更具描述性
    df_tem = df_tem.rename(columns = {'global_time' : 'conam_zero_trans_global_time'})
    
    # 步驟3: 將零金額交易時間資訊合併回原始數據
    # 使用 left join 確保所有原始交易都保留
    df = pd.merge(df, df_tem[['cano','locdt','conam_zero_trans_global_time']], how = 'left' , on = ['cano','locdt'])
    
    # 步驟4: 計算每筆交易與零金額交易的時間差異
    # 正值：當前交易晚於零金額交易
    # 負值：當前交易早於零金額交易  
    # NaN：該日期沒有零金額交易
    df['diff_gtime_with_conam_zero_trans_locdt'] = df['global_time'] - df['conam_zero_trans_global_time']
            
    return df

def preprocess_global_conam_max_min(df):
    """
    交易金額統計特徵創建函數
    
    功能說明：
    計算同卡號同日期內交易金額的統計特徵 (最大值、最小值)
    這些統計特徵能夠反映用戶在特定日期的消費行為模式
    
    統計特徵說明：
    - cano_locdt_conam_min: 同卡號同日期內的最小交易金額
    - cano_locdt_conam_max: 同卡號同日期內的最大交易金額
    
    業務意義：
    - 消費範圍分析：了解用戶單日消費的金額範圍
    - 異常檢測：識別異常高額或低額交易
    - 行為模式：分析用戶的消費習慣和偏好
    
    詐騙檢測應用：
    - 異常高額交易：可能是盜刷行為
    - 金額模式突變：與歷史消費模式不符
    - 測試性小額交易：盜刷者的常見手法
    
    技術實現：
    使用 generic_groupby 工具函數進行高效的分組聚合運算
    按卡號和日期分組，對交易金額進行 min 和 max 聚合
    
    參數：
        df (DataFrame): 包含交易數據的數據框，需要有：
                       - cano: 交易卡號
                       - locdt: 授權日期  
                       - conam: 交易金額-台幣
    
    返回：
        DataFrame: 添加金額統計特徵的數據框
                  新增特徵：
                  - cano_locdt_conam_min: 同卡號同日期最小金額
                  - cano_locdt_conam_max: 同卡號同日期最大金額
    """
    
    # 定義分組維度：按卡號和授權日期分組
    group = ['cano','locdt']
    
    # 定義聚合函數：計算最小值和最大值
    agg_list = ['min','max']
    
    # 定義目標特徵：交易金額
    feature = 'conam'
    
    # 使用通用分組函數進行特徵工程
    # 會自動生成 cano_locdt_conam_min 和 cano_locdt_conam_max 特徵
    df = generic_groupby(df, group, feature, agg_list)
    
    return df
