import numpy as np
import pandas as pd
import math
from joblib import Parallel, delayed


def roundup(x):
    """
    數值向上取整到最近的10的倍數函數
    
    功能說明：
    將輸入的數值向上取整到最近的10的倍數
    這在數據分析中常用於創建分組區間或標準化數值範圍
    
    計算邏輯：
    1. 將數值除以10.0
    2. 使用 math.ceil() 向上取整
    3. 再乘以10得到最終結果
    
    應用場景：
    - 金額分組：將交易金額歸類到標準區間
    - 數據分桶：創建統一的數值分組
    - 特徵工程：標準化連續型變數
    
    範例：
    - roundup(23) → 30
    - roundup(45.6) → 50  
    - roundup(100) → 100
    
    參數：
        x (float/int): 需要向上取整的數值
    
    返回：
        int: 向上取整到10的倍數的結果
    """
    return int(math.ceil(x / 10.0)) * 10

def generic_groupby(df, group, feature, agg_list):
    """
    通用分組聚合特徵工程函數
    
    功能說明：
    這是整個預處理流程中最核心的工具函數，用於創建基於分組的統計特徵
    通過靈活的分組維度和聚合函數組合，自動生成具有業務意義的特徵名稱
    
    核心價值：
    - 代碼復用：避免重複編寫相似的分組聚合邏輯
    - 命名規範：自動生成標準化的特徵名稱
    - 擴展性強：支持任意分組維度和聚合函數組合
    - 性能優化：使用高效的 pandas 分組操作
    
    特徵命名規則：
    生成的特徵名稱格式為：{group1}_{group2}_{feature}_{agg}
    例如：cano_locdt_conam_max (卡號_日期_金額_最大值)
    
    聚合函數支持：
    - 'count': 計數
    - 'sum': 求和
    - 'mean': 平均值
    - 'min': 最小值
    - 'max': 最大值
    - np.std: 標準差 (會自動轉換為 'std')
    - 其他 pandas 支持的聚合函數
    
    業務應用場景：
    - 交易頻率統計：按卡號、日期分組計算交易次數
    - 金額模式分析：按不同維度分組計算金額統計特徵
    - 時間模式特徵：按時間維度分組創建時間相關特徵
    - 商戶行為特徵：按商戶維度分組分析交易模式
    
    詐騙檢測價值：
    - 異常檢測：統計特徵能夠捕捉異常的交易模式
    - 行為建模：為機器學習模型提供豐富的統計特徵
    - 規則引擎：統計特徵可用於設定風控規則閾值
    
    技術實現：
    1. 按指定維度分組並進行聚合運算
    2. 標準化聚合函數名稱 (np.std → 'std')
    3. 自動生成描述性的特徵名稱
    4. 將新特徵合併回原始數據框
    
    參數：
        df (DataFrame): 輸入的數據框
        group (list): 分組欄位列表，例如 ['cano', 'locdt']
        feature (str): 要進行聚合的目標特徵，例如 'conam'
        agg_list (list): 聚合函數列表，例如 ['count', 'sum', np.std]
    
    返回：
        DataFrame: 添加了新統計特徵的數據框
    
    使用範例：
        # 創建卡號日期維度的金額統計特徵
        df = generic_groupby(df, ['cano', 'locdt'], 'conam', ['min', 'max', 'mean'])
        # 生成特徵：cano_locdt_conam_min, cano_locdt_conam_max, cano_locdt_conam_mean
    """
    
    # 步驟1: 按指定分組進行聚合運算
    df_tem = df.groupby(group)[feature].agg(agg_list).reset_index()
    
    # 步驟2: 標準化聚合函數名稱，將 np.std 轉換為 'std'
    # 這確保特徵名稱的一致性和可讀性
    agg_list = ['std' if x==np.std else x for x in agg_list]
    
    # 步驟3: 創建特徵重命名字典
    # 格式：{原聚合函數名: 新特徵名}
    # 新特徵名格式：{分組欄位}_{目標特徵}_{聚合函數}
    rename_dict = dict([(x,'{}_{}_{}'.format('_'.join(group), feature, x)) for x in agg_list])
    
    # 步驟4: 重命名聚合結果的欄位
    df_tem = df_tem.rename(columns = rename_dict)
    
    # 步驟5: 將新特徵合併回原始數據框
    # 使用 left join 確保原始數據的完整性
    df = pd.merge(df, df_tem, how = 'left', on = group)

    return df

def applyParallel(dfGrouped, func):
    """
    並行處理分組數據函數
    
    功能說明：
    對已分組的 DataFrame 進行並行處理，顯著提升大數據集的處理效率
    特別適用於需要對每個分組進行複雜運算的場景
    
    並行處理優勢：
    - 效能提升：利用多核心 CPU 同時處理多個分組
    - 記憶體效率：分組處理避免一次性載入過大數據
    - 擴展性：可處理超大規模數據集
    - 穩定性：單個分組出錯不影響其他分組的處理
    
    適用場景：
    - 大規模特徵工程：對每個用戶/卡號進行複雜的特徵計算
    - 時間序列處理：對每個時間序列進行獨立的統計分析
    - 模型訓練：對不同分組進行獨立的模型訓練或預測
    - 數據清洗：對每個分組進行獨立的數據清洗操作
    
    技術實現：
    - 使用 joblib 的 Parallel 和 delayed 進行並行處理
    - 設定 n_jobs=8 使用8個並行進程
    - 自動將處理結果合併為單一 DataFrame
    
    效能考量：
    - CPU 密集型任務：適合計算密集的操作
    - I/O 操作：對於 I/O 密集型操作效果有限
    - 記憶體使用：需要考慮並行進程的記憶體消耗
    
    詐騙檢測應用：
    - 用戶行為分析：並行分析每個用戶的交易行為模式
    - 特徵工程：並行計算複雜的統計特徵
    - 異常檢測：並行對每個用戶進行異常檢測分析
    
    注意事項：
    - 函數必須是純函數，不依賴外部狀態
    - 分組大小要合理，避免過小的分組導致並行開銷過大
    - 需要足夠的 CPU 核心數才能發揮並行優勢
    
    參數：
        dfGrouped: pandas GroupBy 對象，已經分組的數據
        func (function): 要應用到每個分組的函數
                        函數應該接受一個 DataFrame 並返回一個 DataFrame
    
    返回：
        DataFrame: 所有分組處理結果合併後的數據框
    
    使用範例：
        # 對每個卡號並行計算複雜特徵
        grouped = df.groupby('cano')
        result = applyParallel(grouped, complex_feature_function)
    """
    
    # 使用 joblib 進行並行處理
    # n_jobs=8: 使用8個並行進程
    # delayed(func): 將函數包裝為延遲執行的任務
    # (group for name, group in dfGrouped): 遍歷每個分組
    retLst = Parallel(n_jobs=8)(delayed(func)(group) for name, group in dfGrouped)
    
    # 將所有分組的處理結果合併為單一 DataFrame
    return pd.concat(retLst)