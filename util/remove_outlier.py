def remove_outlier(df):
    """
    移除異常值後計算平均值函數
    
    功能說明：
    對於每一行的預測結果，移除超出 1 個標準差範圍的異常值後計算平均值
    這有助於提高集成模型的穩定性，減少異常 fold 對最終預測的影響
    
    處理邏輯：
    1. 遍歷所有 fold 的預測值
    2. 只保留在 [mean - std, mean + std] 範圍內的值
    3. 計算篩選後值的平均值
    4. 如果所有值都被過濾掉，則返回原始平均值
    
    業務價值：
    - 提高預測穩定性：移除異常的 fold 預測
    - 減少噪音影響：避免個別 fold 的異常影響整體結果
    - 保持魯棒性：即使有異常值也能給出合理預測
    
    參數：
        df (Series): 包含各 fold 預測值和統計信息的 pandas Series
                    必須包含：fold_0, fold_1, ..., fold_n, upper_bound_1std, lower_bound_1std
    
    返回：
        float: 移除異常值後的平均預測值
    """

    proba_list = []
    
    # 動態獲取 fold 數量，而不是硬編碼為 10
    # 查找所有以 'fold_' 開頭的欄位
    fold_columns = [col for col in df.index if str(col).startswith('fold_')]
    
    # 遍歷所有 fold 的預測值
    for fold_col in fold_columns:
        fold_value = df[fold_col]
        
        # 檢查該 fold 的預測值是否在合理範圍內
        # 只保留在 [mean - std, mean + std] 範圍內的值
        if (fold_value < df['upper_bound_1std'] and 
            fold_value > df['lower_bound_1std']):
            proba_list.append(fold_value)
    
    # 計算篩選後的平均值
    if len(proba_list) > 0:
        mean = sum(proba_list) / len(proba_list)
    else:
        # 如果所有值都被過濾掉，返回原始平均值作為備用方案
        mean = df['mean']
    
    return mean