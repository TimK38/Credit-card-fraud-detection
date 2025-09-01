import pandas as pd
import numpy as np


def preprocess_train_test_split(df, cat_features):
    """
    訓練測試集分割與類別特徵對齊預處理函數
    
    功能說明：
    這是機器學習建模前的關鍵預處理步驟，解決訓練集和測試集之間類別特徵不一致的問題
    確保模型在訓練和預測階段使用相同的特徵空間，避免因未見過的類別值導致的模型錯誤
    
    核心問題解決：
    在真實的機器學習場景中，訓練集和測試集可能存在以下問題：
    1. 訓練集中存在測試集沒有的類別值 (會導致模型過擬合到不相關的特徵)
    2. 測試集中存在訓練集沒有的類別值 (會導致預測失敗)
    3. 類別特徵的數據類型不一致 (會影響模型性能)
    
    解決策略：
    - 選擇性對齊：只對非關鍵業務特徵進行對齊處理
    - 業務特徵保護：保留重要業務標識符的原始值和備份
    - 類型統一：將所有類別特徵轉換為 category 類型
    
    重要改進：
    與標準對齊策略不同，本函數保護重要的業務標識符 (bacno, cano, mchno)
    這些欄位不進行對齊處理，因為：
    1. 它們是特徵工程的基礎，不能設為 NaN
    2. 它們用於分組交叉驗證，需要保持原始值
    3. 它們在業務分析中具有重要意義
    
    業務意義：
    - 模型穩定性：確保模型在生產環境中的穩定預測
    - 特徵一致性：保證訓練和預測使用相同的特徵空間
    - 業務連續性：保留關鍵業務標識符用於追溯和分析
    
    詐騙檢測應用：
    - 新商戶類型處理：測試期間出現的新商戶類型在訓練集中可能不存在
    - 新地區處理：新開通服務的地區在訓練數據中可能缺失
    - 新交易類型：新的交易型態需要合理處理
    
    參數：
        df (DataFrame): 包含訓練和測試數據的完整數據框
                       必須包含 fraud_ind 欄位用於區分訓練/測試集
        cat_features (list): 需要進行處理的類別特徵列表
                            例如：['contp', 'stscd', 'etymd', 'stocn', 'mcc', 'csmcu', 'hcefg', 'bacno', 'cano', 'mchno', 'acqic', 'scity']
    
    返回：
        tuple: (df_train, df_test)
            - df_train: 處理後的訓練集，非關鍵類別特徵已對齊，業務標識符保持原值
            - df_test: 處理後的測試集，類別特徵已轉換為 category 類型
    
    使用範例：
        # 定義類別特徵（包含業務標識符）
        categorical_features = ['contp', 'stscd', 'etymd', 'mcc', 'bacno', 'cano', 'mchno']
        
        # 執行訓練測試集分割和特徵對齊
        train_df, test_df = preprocess_train_test_split(df, categorical_features)
        
        # 結果：
        # - bacno, cano, mchno 保持原始值
        # - 其他類別特徵進行對齊處理
        # - 所有特徵都有 _original 備份
    """

    # 步驟1: 根據 fraud_ind 欄位分離訓練集和測試集
    # 訓練集：有 fraud_ind 標籤的數據 (不為 NaN)
    # 測試集：沒有 fraud_ind 標籤的數據 (為 NaN)
    df_train = df[~df['fraud_ind'].isna()]
    df_test = df[df['fraud_ind'].isna()]

    # 步驟2: 定義需要保護的重要業務標識符
    # 這些欄位不進行對齊處理，因為它們是：
    # - 特徵工程的基礎 (用於創建統計特徵)
    # - 分組交叉驗證的依據 (避免數據洩漏)
    # - 業務分析的關鍵 (用戶、卡號、商戶追溯)
    business_identifiers = ['cano',    # 交易卡號 - 用於用戶行為分析和分組驗證
                           'bacno',   # 歸戶帳號 - 用於帳戶關聯分析和 GroupKFold
                           'mchno'    # 特店代號 - 用於商戶風險分析
                           ]
    
    # 步驟3: 為所有業務標識符創建原始值備份
    # 即使這些欄位不會被修改，也創建備份以保持一致性
    # 這樣在後續處理中可以統一使用 _original 欄位
    for identifier in business_identifiers:
        df_train[identifier + '_original'] = df_train[identifier].copy()
    
    # 步驟4: 對非業務標識符的類別特徵進行對齊處理
    # 只處理不在 business_identifiers 中的特徵
    # 這確保了業務關鍵欄位不會被設為 NaN
    features_to_align = [feature for feature in cat_features if feature not in business_identifiers]
    
    for feature in features_to_align:
        # 獲取測試集中該特徵的所有唯一值
        test_unique_values = df_test[feature].unique()
        
        # 將訓練集中不在測試集的類別值替換為 NaN
        # 這確保訓練集只包含測試集中存在的類別值
        # 邏輯：如果訓練集的值在測試集中存在，保留原值；否則設為 NaN
        df_train[feature] = np.where(
            df_train[feature].isin(test_unique_values), 
            df_train[feature], 
            np.nan
        )

    # 步驟5: 統一轉換類別特徵的數據類型
    # 將所有類別特徵轉換為 pandas 的 category 類型
    # 優點：節省內存、提高處理速度、確保類型一致性
    # 注意：包括業務標識符在內的所有類別特徵都轉換類型
    df_test[cat_features] = df_test[cat_features].astype('category')
    df_train[cat_features] = df_train[cat_features].astype('category')

    return df_train, df_test
