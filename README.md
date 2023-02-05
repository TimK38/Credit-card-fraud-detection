# 玉山人工智慧公開挑戰賽2019秋季賽 真相只有一個『信用卡盜刷偵測』
## 競賽說明
本比賽目的為建立信用卡反盜刷偵測模型，為此參賽者必須建立二分類模型，判斷該筆信用卡交易是否為盜刷交易
## 參賽名次
![image](https://github.com/TimKuo38/Credit-card-fraud-detection/blob/master/%E5%8F%83%E8%B3%BD%E5%90%8D%E6%AC%A1.png](https://github.com/TimK38/Credit-card-fraud-detection/blob/main/%E5%8F%83%E8%B3%BD%E5%90%8D%E6%AC%A1.png)
## 資料說明
* 本次競賽提供訓練集(train.csv)和測試集(test.csv)兩張資料表,訓練集共
1,521,787 筆,測試集共 421,665 筆。
* 訓練集的授權日期為 1 ~ 90, 共 90 天的信用卡授權交易紀錄,請參賽者預測授權日期在 91 ~ 120 的各筆交易是否為盜刷。
<br>[欄位 說明]
<br>bacno 歸戶帳號
<br>txkey 交易序號
<br>locdt 授權日期
<br>loctm 授權時間
<br>cano 交易卡號
<br>contp 交易類別
<br>etymd 交易型態
<br>mchno 特店代號
<br>acqic 收單行代碼
<br>mcc MCC_CODE
<br>conam 交易金額-台幣(經過轉換)
<br>ecfg 網路交易註記
<br>insfg 分期交易註記
<br>iterm 分期期數
<br>stocn 消費地國別
<br>scity 消費城市
<br>stscd 狀態碼
<br>ovrlt 超額註記碼
<br>flbmk Fallback 註記
<br>hcefg 支付形態
<br>csmcu 消費地幣別
<br>flg_3dsmk 3DS 交易註記
<br>fraud_ind 盜刷註記 
## 執行說明
感謝 第三名 阿罵我要吃糖果 有效程式供我參考與調整，下方為這次盜刷偵測模型的流程
1. 執行new_features.ipynb，此程式會依raw_data產出進階的頻率變數，來描繪每筆消費紀錄與過去消費紀錄間的差異 <br> 
2. 執行Tim_main_lgbm.ipynb，該程式有四個步驟，
  <br>2-1:用raw_data產進階變數。
  <br>2-2:將資料切分成兩群，分別是過去有被盜刷過的與過去沒被盜刷的紀錄，
  <br>2-3:分別對兩群人建立ensemble lightgbm model
  <br>2-4:產出預測結果供後續上傳評分
