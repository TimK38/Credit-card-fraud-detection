def generate_statistic(df):
    """
    生成交叉驗證預測統計特徵函數
    
    功能說明：
    對交叉驗證的多個 fold 預測結果計算統計特徵，用於後續的異常值檢測和集成
    這些統計特徵幫助識別和處理異常的 fold 預測，提高最終預測的穩定性
    
    生成的統計特徵：
    1. max: 所有 fold 預測值的最大值
    2. min: 所有 fold 預測值的最小值  
    3. std: 所有 fold 預測值的標準差
    4. mean: 所有 fold 預測值的平均值
    5. upper_bound_1std: 上界 (mean + 1*std)
    6. lower_bound_1std: 下界 (mean - 1*std)
    
    業務價值：
    - 異常檢測：識別偏離正常範圍的 fold 預測
    - 集成學習：為多個模型預測提供統計基礎
    - 穩定性評估：通過標準差評估預測的一致性
    
    統計邊界的作用：
    - 1 標準差範圍涵蓋約 68% 的正常預測值
    - 超出此範圍的預測被視為異常值
    - 用於 remove_outlier 函數的異常值過濾
    
    參數：
        df (DataFrame): 包含多個 fold 預測結果的數據框
                       欄位格式：fold_0, fold_1, fold_2, ..., fold_n
    
    返回：
        DataFrame: 添加了統計特徵的數據框
                  新增欄位：max, min, std, mean, upper_bound_1std, lower_bound_1std
    
    使用範例：
        # 假設有 10 fold 的預測結果
        df_predictions = pd.DataFrame({
            'fold_0': [0.1, 0.8, 0.3],
            'fold_1': [0.2, 0.7, 0.4],
            ...
        })
        
        # 生成統計特徵
        df_with_stats = generate_statistic(df_predictions)
        
        # 結果包含原始預測 + 統計特徵
        # df_with_stats.columns: ['fold_0', 'fold_1', ..., 'max', 'min', 'std', 'mean', 'upper_bound_1std', 'lower_bound_1std']
    """
    
    # 創建原始數據的副本用於統計計算
    # 避免修改原始 fold 預測值
    df_folds = df.copy()
    
    # 計算所有 fold 預測值的基本統計量
    df['max'] = df_folds.max(axis=1)    # 每行的最大預測值
    df['min'] = df_folds.min(axis=1)    # 每行的最小預測值
    
    # 計算標準差和平均值
    # ddof=0 表示使用總體標準差（除以 N）而不是樣本標準差（除以 N-1）
    # 這在交叉驗證場景中更合適，因為我們有完整的 fold 預測集合
    df['std'] = df_folds.std(ddof=0, axis=1)   # 每行預測值的標準差
    df['mean'] = df_folds.mean(axis=1)         # 每行預測值的平均值
    
    # 創建異常值檢測的邊界
    # 使用 1 個標準差作為正常範圍的邊界
    # 這是一個相對寬鬆的標準，適合保留大部分有效預測
    df['upper_bound_1std'] = df['mean'] + df['std'] * 1  # 上邊界
    df['lower_bound_1std'] = df['mean'] - df['std'] * 1  # 下邊界

    return df
