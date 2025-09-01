import pandas as pd
import numpy as np


def preprocess_change_card(df):
    """
    換卡行為特徵預處理函數
    
    功能說明：
    分析同一帳戶下不同卡號的使用模式，創建換卡相關的特徵
    這些特徵對於詐騙檢測非常重要，因為異常的換卡行為往往與詐騙活動相關
    
    核心概念：
    - 換卡行為：同一帳戶(bacno)在同一週期(days)內使用不同卡號(cano)進行交易
    - 時間間隔：分析卡號之間的使用時間差異
    - 交易模式：識別異常的卡號切換模式
    
    生成的特徵：
    1. cano_last_trans_locdt: 該卡號在該週期內的最後交易日期
    2. min: 該卡號在該週期內的首次交易日期  
    3. next_card_min: 下一張卡的首次交易日期
    4. diff_locdt_of_two_card: 兩張卡之間的時間間隔
    5. diff_locdt_with_last_trans_cano: 當前交易與該卡最後交易的時間差
    6. diff_locdt_with_last_trans_days_cano: 當前週期與該卡最後交易週期的差異
    
    業務意義：
    - 快速換卡可能表示卡片遺失或被盜用
    - 異常的卡號使用時間間隔需要特別關注
    - 同帳戶多卡交替使用模式分析
    
    詐騙檢測價值：
    - 盜刷者可能會在短時間內使用多張卡片
    - 正常用戶的換卡行為通常有規律可循
    - 時間間隔異常可能表示異常活動
    
    參數：
        df (DataFrame): 輸入的交易數據，必須包含以下欄位：
                       - bacno: 歸戶帳號 (同一用戶的唯一識別)
                       - cano: 交易卡號 (信用卡號)
                       - days: 時間週期 (由 preprocess_init 創建)
                       - locdt: 授權日期
    
    返回：
        DataFrame: 添加了換卡行為特徵的數據框
    """

    # 步驟1: 計算每張卡在每個週期內的交易日期範圍
    # 按帳號、卡號、週期分組，計算該卡在該週期的首次和最後交易日期
    df_tem = df.groupby(['bacno','cano','days']).agg(['max','min'])['locdt'].reset_index().sort_values(by = ['cano','max'])
    
    # 步驟2: 找出同帳戶下一張卡的首次交易日期
    # 按帳號和週期分組，將下一張卡的首次交易日期向前移動
    df_tem['next_card_min'] = df_tem.groupby(['bacno','days'])['min'].shift(-1)
    
    # 步驟3: 處理時間邏輯，確保換卡的時間順序合理
    # 如果當前卡的最後交易日期 >= 下一張卡的首次交易日期，則設為 NaN (表示沒有明確的換卡行為)
    df_tem['next_card_min'] = np.where(df_tem['max'] - df_tem['next_card_min']>=0, np.nan, df_tem['next_card_min'])
    
    # 步驟4: 計算兩張卡之間的時間間隔
    # 負值表示有重疊使用期間，正值表示有空檔期間
    df_tem['diff_locdt_of_two_card'] = df_tem['max'] - df_tem['next_card_min']
    
    # 步驟5: 重新命名欄位，使其更具描述性
    df_tem = df_tem.rename(columns = {'max':'cano_last_trans_locdt'})
    
    # 步驟6: 選擇需要的欄位 (排除原始的 bacno 欄位，避免重複)
    df_tem = df_tem.iloc[:,list(range(1,7))]
    
    # 步驟7: 將換卡特徵合併回原始數據
    df = pd.merge(df,df_tem,how = 'left', on = ['cano','days'])
    
    # 步驟8: 計算當前交易與該卡最後交易的時間差
    # 正值表示當前交易晚於該卡的最後交易（可能是重新使用舊卡）
    df['diff_locdt_with_last_trans_cano'] = df['locdt'] - df['cano_last_trans_locdt']
    
    # 步驟9: 計算當前週期與該卡最後交易週期的差異
    # 用於識別長期未使用後重新啟用的卡片
    df['diff_locdt_with_last_trans_days_cano'] = df['days'] - df['cano_last_trans_locdt']

    return df

