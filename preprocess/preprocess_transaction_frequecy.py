import pandas as pd
import numpy as np
from preprocess.util import generic_groupby
    
def preprocess_transaction_frequency(df):
    """
    交易頻率特徵預處理函數
    
    功能說明：
    創建基於不同維度的交易頻率統計特徵，用於捕捉用戶的交易行為模式
    這些特徵有助於識別異常的交易頻率，是詐騙檢測的重要指標
    
    生成的特徵：
    1. cano_days_txkey_count: 同卡號在同一週期(30/60/90/120天)內的交易次數
    2. cano_locdt_txkey_count: 同卡號在同一天內的交易次數  
    3. bacno_locdt_mchno_txkey_count: 同帳號在同一天同一特店的交易次數
    
    業務意義：
    - 高頻交易可能表示正常的消費習慣或異常的盜刷行為
    - 同一天多次交易需要特別關注
    - 同一特店的重複交易模式分析
    
    參數：
        df (DataFrame): 輸入的交易數據，必須包含以下欄位：
                       - txkey: 交易序號 (用於計數)
                       - cano: 交易卡號
                       - days: 時間週期 (由 preprocess_init 創建)
                       - locdt: 授權日期
                       - bacno: 歸戶帳號
                       - mchno: 特店代號
    
    返回：
        DataFrame: 添加了交易頻率特徵的數據框
    """

    # 設定要統計的目標特徵：交易序號 (txkey)
    feature = 'txkey'
    
    # 設定聚合函數：計數 (count)
    agg_list = ['count']
    
    # 定義不同的分組維度，用於創建多種頻率特徵
    groups_list = [
        ['cano','days'],           # 卡號 + 時間週期：捕捉週期性交易模式
        ['cano','locdt'],          # 卡號 + 授權日期：捕捉每日交易頻率
        ['bacno','locdt','mchno']  # 帳號 + 日期 + 特店：捕捉特定場所的交易模式
    ]
    
    # 遍歷每個分組維度，創建對應的頻率特徵
    for groups in groups_list:
        # 使用通用的 groupby 函數進行特徵工程
        # 會自動生成格式為 "{groups}_{feature}_{agg}" 的新特徵名稱
        df = generic_groupby(df, groups, feature, agg_list)

    return df
