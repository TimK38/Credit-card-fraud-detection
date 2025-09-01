from random import randrange

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    recall_score,
    roc_auc_score
)
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold

from lightgbm import LGBMClassifier, early_stopping, log_evaluation


class LGBM_Model:
    """
    LightGBM 模型封裝類別
    
    這個類別封裝了 LightGBM 分類器，提供了完整的訓練、驗證和預測流程
    特別針對信用卡詐騙檢測任務進行了優化
    
    屬性：
        features (list): 用於訓練的特徵列表
        clf (LGBMClassifier): LightGBM 分類器實例
        df_feature_importance (DataFrame): 特徵重要性統計結果
    
    主要功能：
        - 自動化模型建構和參數設定
        - GroupKFold 交叉驗證訓練
        - Out-of-fold 預測生成
        - 特徵重要性分析
        - 多種評估指標計算
    """
    
    def __init__(self, features):
        """
        初始化 LightGBM 模型
        
        參數：
            features (list): 用於模型訓練的特徵欄位列表
                           例如：['conam', 'loctm', 'cano_locdt_txkey_count', ...]
        
        初始化內容：
            - 儲存特徵列表用於後續訓練和重要性分析
            - 初始化分類器為 None，將在需要時建構
            - 準備特徵重要性統計容器
        
        使用範例：
            features = ['conam', 'loctm', 'mcc', 'cano_locdt_txkey_count']
            model = LGBM_Model(features)
        """
        self.features = features  # 儲存特徵列表
        self.clf = None          # 分類器實例，延遲初始化
        self.df_feature_importance = None  # 特徵重要性統計結果

    def build_clf(self, n_estimators=1000, learning_rate=0.1, num_leaves=16, 
                  reg_alpha=10, reg_lambda=7):
        """
        建構 LightGBM 分類器
        
        功能說明：
        根據指定參數建立 LightGBM 分類器實例，參數已針對信用卡詐騙檢測任務進行調優
        
        參數說明：
            n_estimators (int): 樹的數量，默認 1000
                               更多的樹通常能提高性能，但會增加訓練時間
            learning_rate (float): 學習率，默認 0.1
                                  控制每棵樹的貢獻，較小值需要更多樹但通常更穩定
            num_leaves (int): 每棵樹的葉子數量，默認 16
                             控制模型複雜度，過大容易過擬合
            reg_alpha (float): L1 正則化係數，默認 10
                              有助於特徵選擇和防止過擬合
            reg_lambda (float): L2 正則化係數，默認 7
                               平滑權重，防止過擬合
        
        模型配置說明：
            - boosting_type='gbdt': 使用梯度提升決策樹
            - verbosity=1: 顯示訓練信息（新版本參數）
            - metric='None': 不使用內建評估指標，使用自定義指標
            - n_jobs=-1: 使用所有 CPU 核心進行並行訓練
            - random_state=10: 固定隨機種子確保結果可重現
            - max_depth=-1: 不限制樹的深度，由 num_leaves 控制複雜度
            - min_child_samples=200: 葉子節點最小樣本數，防止過擬合
        
        詐騙檢測優化：
            - 參數組合針對不平衡數據集進行了調優
            - 正則化參數有助於處理高維稀疏特徵
            - 保守的 num_leaves 設定避免過擬合
        
        使用範例：
            model = LGBM_Model(features)
            model.build_clf(n_estimators=1500, learning_rate=0.05)
        """
        
        self.clf = LGBMClassifier(
            # 核心算法參數
            boosting_type='gbdt',        # 梯度提升決策樹
            n_estimators=n_estimators,   # 樹的數量
            learning_rate=learning_rate, # 學習率
            num_leaves=num_leaves,       # 葉子數量
            max_depth=-1,                # 不限制深度，由 num_leaves 控制
            
            # 正則化參數
            reg_alpha=reg_alpha,         # L1 正則化
            reg_lambda=reg_lambda,       # L2 正則化
            min_child_samples=200,       # 葉子節點最小樣本數
            
            # 系統參數
            verbosity=1,                 # 顯示訓練信息（新版本參數）
            n_jobs=-1,                   # 使用所有 CPU 核心
            random_state=10,             # 固定隨機種子
            metric='None',               # 使用自定義評估指標
            
            # 註釋掉的參數：可選的進階調優參數
            # is_unbalance='True',       # 自動處理不平衡數據
            # subsample=1,               # 樣本採樣比例
            # colsample_bytree=1,        # 特徵採樣比例
            # min_child_weight=1,        # 葉子節點最小權重和
            # min_split_gain=0.0,        # 分裂最小增益
            # objective='regression_l1', # 目標函數
            # subsample_for_bin=240000,  # 建構直方圖的樣本數
            # subsample_freq=1,          # 採樣頻率
            # class_weight='balanced',   # 類別權重平衡
            # scale_pos_weight=2,        # 正樣本權重縮放
        )


    def run(self, data, y, groups, test, eval_metric, n_splits=10, early_stopping_rounds=100):
        """
        執行 LightGBM 交叉驗證訓練
        
        功能說明：
        使用 GroupKFold 交叉驗證訓練 LightGBM 模型，生成 out-of-fold 預測和測試集預測
        這是整個模型訓練的核心方法，確保模型評估的可靠性和泛化能力
        
        參數：
            data (DataFrame): 訓練數據特徵矩陣
            y (Series): 訓練數據標籤向量 (0/1)
            groups (Series): 分組標識，用於 GroupKFold（通常是 bacno_transfer）
            test (DataFrame): 測試數據特徵矩陣
            eval_metric (function): 評估指標函數，例如 lgbm_averge_precision
            n_splits (int): 交叉驗證折數，默認 10
            early_stopping_rounds (int): 早停輪數，默認 100
        
        返回：
            tuple: (oof_preds_LGBM, df_sub_preds_LGBM, self.clf)
                - oof_preds_LGBM: Out-of-fold 預測結果
                - df_sub_preds_LGBM: 測試集各 fold 預測結果
                - self.clf: 最後一個 fold 訓練的模型實例
        
        GroupKFold 的重要性：
            在信用卡詐騙檢測中，同一用戶的交易存在強相關性
            GroupKFold 確保同一用戶的所有交易都在同一個 fold 中
            這避免了數據洩漏，提供更真實的模型性能評估
        
        Out-of-fold 預測：
            每個樣本都會被分配到驗證集一次，獲得無偏的預測結果
            這些預測可用於：
            - 模型性能評估
            - 特徵選擇
            - 模型融合
            - 閾值優化
        
        特徵重要性統計：
            每個 fold 都會記錄特徵重要性
            最終可以分析特徵的穩定性和平均重要性
        
        使用範例：
            model = LGBM_Model(features)
            oof_preds, test_preds, clf = model.run(
                X_train, y_train, groups, X_test, 
                model.lgbm_averge_precision, n_splits=10
            )
        """
        
        # 初始化預測結果容器
        oof_preds_LGBM = np.zeros((data.shape[0]))      # Out-of-fold 預測
        sub_preds_LGBM = np.zeros((test.shape[0]))       # 測試集預測（未使用）
        df_sub_preds_LGBM = pd.DataFrame()               # 各 fold 測試集預測
        self.df_feature_importance = pd.DataFrame()      # 特徵重要性統計
        
        # 如果分類器未建構，使用默認參數建構
        if not self.clf: 
            self.build_clf()

        # 建立 GroupKFold 交叉驗證器
        # 確保同一組（通常是同一用戶）的數據不會同時出現在訓練和驗證集中
        folds = GroupKFold(n_splits=n_splits)
        
        # 開始交叉驗證訓練
        for n_fold, (train_idx, valid_idx) in enumerate(folds.split(data, y, groups)):
            # 分割當前 fold 的訓練和驗證數據
            train_x, train_y = data.iloc[train_idx], y.iloc[train_idx]
            valid_x, valid_y = data.iloc[valid_idx], y.iloc[valid_idx]
            
            print("Starting LightGBM. Fold {},Train shape: {}, test shape: {}".format(
                n_fold+1, data.shape, test.shape))

            # 設置 LightGBM callbacks（新版本兼容性）
            # 在新版本的 LightGBM 中，verbose 和 early_stopping_rounds 參數
            # 都被移到 callbacks 中，提供更靈活的控制
            callbacks_list = [
                log_evaluation(period=100),  # 每 100 輪輸出一次評估結果
                early_stopping(stopping_rounds=early_stopping_rounds)  # early stopping 機制
            ]
            
            # 訓練當前 fold 的模型
            self.clf.fit(
                train_x, train_y,
                eval_set=[(train_x, train_y), (valid_x, valid_y)],  # 評估集
                eval_metric=eval_metric,        # 自定義評估指標
                callbacks=callbacks_list,       # 訓練控制 callbacks
                categorical_feature='auto',     # 自動檢測類別特徵
            )

            # 生成 out-of-fold 預測
            # 只取正類概率（[:, 1]），因為這是二分類問題
            oof_preds_LGBM[valid_idx] += self.clf.predict_proba(valid_x)[:, 1]
            
            # 生成測試集預測（每個 fold 單獨記錄）
            df_sub_preds_LGBM['fold_{}'.format(n_fold)] = self.clf.predict_proba(test)[:, 1]
            
            # 記錄當前 fold 的特徵重要性
            df_fold_importance = pd.DataFrame()
            df_fold_importance["feature"] = self.features
            df_fold_importance["importance"] = self.clf.feature_importances_
            df_fold_importance["fold"] = n_fold + 1

            # 累積特徵重要性統計
            self.df_feature_importance = pd.concat([
                self.df_feature_importance, df_fold_importance
            ], axis=0)
        
        # 輸出最終評估結果
        print('Summary:')            
        print('LGBM Testing_Set average_precision_score %.6f' % 
              average_precision_score(y, oof_preds_LGBM))

        return oof_preds_LGBM, df_sub_preds_LGBM, self.clf

    @staticmethod 
    def lgb_f1(truth, predictions):
        """
        LightGBM F1 評估指標函數
        
        功能說明：
        計算 F1 分數作為 LightGBM 的自定義評估指標
        使用固定閾值 0.275 將概率預測轉換為二分類標籤
        
        參數：
            truth (array-like): 真實標籤 (0/1)
            predictions (array-like): 模型預測概率 (0-1)
        
        返回：
            tuple: (指標名稱, F1分數, 是否越大越好)
                - "F1": 指標名稱
                - f1: F1 分數值
                - True: 表示越大越好
        
        閾值說明：
            0.275 是經過調優的閾值，針對信用卡詐騙檢測任務
            在精確率和召回率之間取得平衡
            
        F1 分數意義：
            F1 = 2 * (precision * recall) / (precision + recall)
            綜合考慮精確率和召回率的調和平均數
            特別適合不平衡數據集的評估
        
        使用範例：
            # 在 LightGBM 訓練中使用
            model.run(X_train, y_train, groups, X_test, 
                     LGBM_Model.lgb_f1, n_splits=10)
        """
        # 使用固定閾值將概率轉換為二分類標籤
        pred_labels = np.where(predictions >= 0.275, 1, 0)
        
        # 計算 F1 分數
        f1 = f1_score(truth, pred_labels)
        
        # 返回 LightGBM 期望的格式：(名稱, 分數, 是否越大越好)
        return ("F1", f1, True)
    
    @staticmethod
    def lgbm_averge_precision(truth, predictions):
        """
        LightGBM 平均精確率評估指標函數
        
        功能說明：
        計算平均精確率（Average Precision）作為 LightGBM 的自定義評估指標
        這是信用卡詐騙檢測中常用的評估指標，特別適合不平衡數據集
        
        參數：
            truth (array-like): 真實標籤 (0/1)
            predictions (array-like): 模型預測概率 (0-1)
        
        返回：
            tuple: (指標名稱, 平均精確率, 是否越大越好)
                - "Averge Precision": 指標名稱（注意：原始拼寫保持一致）
                - aps: 平均精確率分數值
                - True: 表示越大越好
        
        平均精確率意義：
            AP 是 Precision-Recall 曲線下的面積
            綜合評估模型在不同閾值下的性能
            對不平衡數據集比 ROC AUC 更敏感
            
        詐騙檢測優勢：
            - 重點關注少數類（詐騙交易）的檢測能力
            - 不受大量正常交易的影響
            - 提供更真實的模型性能評估
        
        使用範例：
            # 在 LightGBM 訓練中使用（推薦）
            model.run(X_train, y_train, groups, X_test, 
                     LGBM_Model.lgbm_averge_precision, n_splits=10)
        """
        # 計算平均精確率
        aps = average_precision_score(truth, predictions)
        
        # 返回 LightGBM 期望的格式：(名稱, 分數, 是否越大越好)
        # 注意：保持原始拼寫 "Averge" 以維持向後兼容性
        return ("Averge Precision", aps, True)



