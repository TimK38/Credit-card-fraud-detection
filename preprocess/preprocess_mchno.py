import pandas as pd
import numpy as np


def preprocess_mchno(df):
    """
    特店代號特徵預處理主函數
    
    功能說明：
    對特店代號 (mchno) 進行特徵工程，創建與商戶相關的行為模式特徵
    特店代號是詐騙檢測中的重要維度，異常的商戶交易模式往往是詐騙的重要指標
    
    處理流程：
    1. 分析同帳戶在同特店的交易時間跨度特徵
    2. 計算同卡號在同特店的交易序列索引特徵
    
    業務價值：
    - 商戶行為分析：了解用戶在特定商戶的消費模式
    - 異常檢測：識別異常的商戶交易行為
    - 時間模式分析：分析用戶在商戶的消費時間規律
    
    詐騙檢測意義：
    - 盜刷者可能會在特定類型的商戶進行集中交易
    - 異常的商戶交易頻率和時間跨度需要關注
    - 商戶交易序列可以反映交易的合理性
    
    參數：
        df (DataFrame): 輸入的交易數據，必須包含以下欄位：
                       - bacno: 歸戶帳號
                       - mchno: 特店代號
                       - cano: 交易卡號
                       - days: 時間週期
                       - locdt: 授權日期
    
    返回：
        DataFrame: 添加了特店相關特徵的數據框
    """
    
    # 步驟1: 計算同帳戶在同特店的交易時間跨度
    df = bacno_mchno_locdt_head_tail_diff(df)
    
    # 步驟2: 計算同卡號在同特店的交易序列索引
    df = cano_days_mchno_index(df)

    return df

def bacno_mchno_locdt_head_tail_diff(df):
    """
    帳戶特店交易時間跨度分析函數
    
    功能說明：
    計算同一帳戶在同一特店同一週期內的交易時間跨度
    通過分析首次和最後一次交易的時間差，了解用戶在特定商戶的消費時間模式
    
    核心概念：
    - 交易時間跨度：同帳戶在同特店的首次交易到最後交易的天數差異
    - 消費持續性：反映用戶在特定商戶的消費持續時間
    - 行為規律性：正常用戶在熟悉商戶的消費通常有時間規律
    
    業務意義：
    - 商戶忠誠度：時間跨度長可能表示用戶對該商戶的忠誠度高
    - 消費習慣：反映用戶在特定商戶的消費頻率和規律
    - 異常檢測：異常的時間跨度可能表示異常行為
    
    詐騙檢測價值：
    - 集中交易：盜刷者可能在短時間內在同一商戶進行多筆交易
    - 時間異常：與用戶歷史在該商戶的消費模式不符
    - 商戶類型：某些商戶類型的異常交易模式需要特別關注
    
    技術實現：
    1. 使用 groupby().head(1) 獲取每組的第一筆交易（首次交易）
    2. 使用 groupby().tail(1) 獲取每組的最後一筆交易（最後交易）
    3. 計算首次和最後交易的日期差異
    
    參數：
        df (DataFrame): 包含交易數據的數據框，需要有：
                       - bacno: 歸戶帳號
                       - mchno: 特店代號
                       - days: 時間週期
                       - locdt: 授權日期
    
    返回：
        DataFrame: 添加交易時間跨度特徵的數據框
                  新增特徵：
                  - locdt_head: 該帳戶在該特店該週期的首次交易日期
                  - locdt_tail: 該帳戶在該特店該週期的最後交易日期
                  - bacno_mchno_locdt_head_tail_diff: 首次與最後交易的時間差
    """

    # 步驟1: 獲取每個帳戶在每個特店每個週期的首次交易
    # groupby().head(1) 取得每組的第一筆記錄
    df_head = df.groupby(['bacno','mchno','days']).head(1)[['bacno','mchno','days','locdt']]
    df_head = df_head.rename(columns = {'locdt' : 'locdt_head'})
    
    # 步驟2: 獲取每個帳戶在每個特店每個週期的最後交易
    # groupby().tail(1) 取得每組的最後一筆記錄
    df_tail = df.groupby(['bacno','mchno','days']).tail(1)[['bacno','mchno','days','locdt']]
    df_tail = df_tail.rename(columns = {'locdt' : 'locdt_tail'})
    
    # 步驟3: 合併首次和最後交易資訊
    df_head = pd.merge(df_head, df_tail, how = 'left', on = ['bacno','mchno','days'])
    
    # 步驟4: 計算交易時間跨度
    # 正值：表示在該商戶有多天的交易記錄
    # 0：表示只在同一天有交易記錄
    df_head['bacno_mchno_locdt_head_tail_diff'] = df_head['locdt_tail'] - df_head['locdt_head']
    
    # 步驟5: 將時間跨度特徵合併回原始數據
    df = pd.merge(df,df_head, how = 'left', on =['bacno','mchno','days'])

    return df

def cano_days_mchno_index(df):
    """
    卡號特店交易序列索引函數
    
    功能說明：
    為每張卡在每個週期內每個特店的交易創建序列索引
    這個索引反映了該卡在該特店的交易順序和頻率
    
    核心概念：
    - 交易序列：同卡號在同特店的交易按時間順序編號
    - 交易頻率：索引數值反映在該商戶的交易次數
    - 行為模式：序列索引可以幫助識別異常的交易模式
    
    業務意義：
    - 商戶熟悉度：高索引值可能表示用戶對該商戶很熟悉
    - 消費頻率：索引增長速度反映消費頻率
    - 交易合理性：序列索引有助於判斷交易的合理性
    
    詐騙檢測價值：
    - 異常頻率：短時間內高索引值可能表示異常交易
    - 新商戶風險：首次在某商戶交易（索引=1）的風險評估
    - 模式識別：結合其他特徵識別異常的交易序列模式
    
    應用場景：
    - 風險評分：將交易序列索引作為風險評分的因子
    - 異常檢測：識別異常的商戶交易頻率
    - 用戶畫像：了解用戶的商戶偏好和消費習慣
    
    技術實現：
    1. 初始化索引值為1
    2. 按卡號、週期、特店分組進行累積求和
    3. 結果為每筆交易在該組合中的序列位置
    
    參數：
        df (DataFrame): 包含交易數據的數據框，需要有：
                       - cano: 交易卡號
                       - days: 時間週期
                       - mchno: 特店代號
    
    返回：
        DataFrame: 添加交易序列索引特徵的數據框
                  新增特徵：
                  - cano_days_mchno_index: 該卡在該週期該特店的交易序列索引
                    (1表示首次交易，2表示第二次交易，以此類推)
    """

    # 步驟1: 初始化序列索引，每筆交易都設為1
    df['cano_days_mchno_index'] = 1
    
    # 步驟2: 按卡號、週期、特店分組，進行累積求和
    # cumsum() 會將每組內的1進行累積相加，形成序列索引
    # 第1筆交易索引為1，第2筆為2，第3筆為3，以此類推
    df['cano_days_mchno_index'] = df.groupby(['cano','days','mchno'])['cano_days_mchno_index'].cumsum()
    
    return df
