# 資料說明文件（Data Dictionary）
## 臺北市道路交通事故熱點風險預測與照相設備設置建議系統

| 項目 | 內容 |
|---|---|
| 文件版本 | v1.0 |
| 最後更新 | 2026-06-13 |
| 對應模組 | s01_clean.py、s02_hotspot.py、s03_series.py |

---

## 目錄

1. [原始資料（data/raw/）](#1-原始資料)
2. [accidents.parquet — 事故主檔](#2-accidentsparquet--事故主檔)
3. [cameras.parquet — 照相設備表](#3-camerasparquet--照相設備表)
4. [hotspots.csv — 熱點摘要](#4-hotspotscsv--熱點摘要)
5. [daily_series.csv — 全市日事故數序列](#5-daily_seriescsv--全市日事故數序列)
6. [hotspot_monthly.csv — 熱點月事故數 Panel](#6-hotspot_monthlycsv--熱點月事故數-panel)
7. [dist_vectors.csv — 時間指紋向量](#7-dist_vectorscsv--時間指紋向量)
8. [anomalies.csv — 清洗異常紀錄](#8-anomaliescsv--清洗異常紀錄)

---

## 1. 原始資料

**位置**：`data/raw/`（唯讀，不入版控）

| 檔名 | 編碼 | 筆數 | 說明 |
|---|---|---|---|
| 110年臺北市道路交通事故斑點圖(改A1A2).csv | cp950 | 24,157 | 2021 年事故資料（SRS IN-1） |
| 111年臺北市道路交通事故斑點圖(改A1A2).csv | cp950 | 25,424 | 2022 年事故資料（SRS IN-2） |
| 112年臺北市道路交通事故斑點圖.csv | cp950 | 24,272 | 2023 年事故資料（SRS IN-3） |
| 113年臺北市道路交通事故斑點圖 .csv | cp950 | 22,368 | 2024 年事故資料（SRS IN-4） |
| 114年臺北市道路交通事故斑點圖.csv | cp950 | 22,762 | 2025 年事故資料（SRS IN-5） |
| 臺北市政府警察局固定式違規照相設備及區間測速裝置設置地點一覽表.csv | Big5 | 143 | 照相設備現況（SRS IN-6） |

**原始欄位（事故斑點圖）**：`發生時間、處理別、肇事地點、座標-X、座標-Y`

---

## 2. accidents.parquet — 事故主檔

**位置**：`data/processed/accidents.parquet`
**產生模組**：s01_clean.py（基礎欄位）、s02_hotspot.py（新增 cluster）
**筆數**：118,980
**編碼**：UTF-8（Parquet 格式）

| 欄位 | 型別 | 空值數 | 說明 |
|---|---|---|---|
| 發生時間 | datetime64[ns] | 0 | 事故發生的日期與時間，精度為分鐘。範圍：2021-01-01 ~ 2025-12-31 |
| 處理別 | int8 | 0 | 事故嚴重程度：`1` = A1（死亡）、`2` = A2（受傷） |
| 肇事地點 | object (str) | 0 | 事故發生路段之文字描述，如「大安區新生南路與信義路口」 |
| longitude | float64 | 0 | 事故地點**經度**（WGS84）。合法範圍：121.4–121.7。kepler.gl 相容欄名 |
| latitude | float64 | 0 | 事故地點**緯度**（WGS84）。合法範圍：24.9–25.25。kepler.gl 相容欄名 |
| cluster | int16 | 0 | DBSCAN 熱點標籤。`-1` = 噪音點（未歸入任何熱點），`0–144` = 熱點編號 |

**備註**
- A1 事故約占 0.3%（全部 A2 主導）
- cluster = -1 的噪音點共 96,122 筆（占 80.8%），保留於主檔供完整分析使用
- longitude/latitude 已過濾座標缺漏與超出範圍之異常值（見 anomalies.csv）

---

## 3. cameras.parquet — 照相設備表

**位置**：`data/processed/cameras.parquet`
**產生模組**：s01_clean.py
**筆數**：143
**編碼**：UTF-8（Parquet 格式）

| 欄位 | 型別 | 空值數 | 空值率 | 說明 |
|---|---|---|---|---|
| 編號 | int64 | 0 | 0% | 設備流水編號（1–143） |
| 功能 | object (str) | 0 | 0% | 設備功能類型，主要值：`測速`、`闖紅燈`、`闖紅燈、測速` |
| 設置路段 | object (str) | 0 | 0% | 設備所在路段名稱，如「信義路4段」 |
| 設置地點 | object (str) | 31 | **21.7%** | 更精確的設置位置描述（部分為空，原始資料即缺漏） |
| latitude | float64 | 0 | 0% | 設備位置**緯度**（WGS84）。kepler.gl 相容欄名 |
| longitude | float64 | 0 | 0% | 設備位置**經度**（WGS84）。kepler.gl 相容欄名 |
| 速限 | object (str) | 0 | 0% | 該路段速限（km/h），部分路段有雙向不同速限（如「50(外環)\n60(內環)」） |
| 方向 | object (str) | 0 | 0% | 拍攝方向，如「北向南」、「南向北」 |
| 型式-型號規格 | object (str) | 0 | 0% | 設備廠牌型號 |
| 廠商 | object (str) | 0 | 0% | 設備廠商名稱，目前全部為「是拓科」 |
| 廠商代碼 | int64 | 0 | 0% | 廠商代碼，目前全部為 63000 |
| is_speed | bool | 0 | 0% | 衍生欄位：`True` = 具測速功能（功能欄含「測速」），`False` = 純闖紅燈照相 |

**備註**
- 測速設備：98 台（68.5%）；純闖紅燈：45 台（31.5%）
- `設置地點` 空值率 21.7% 為原始政府資料本身缺漏，不影響距離計算（使用 latitude/longitude）

---

## 4. hotspots.csv — 熱點摘要

**位置**：`data/processed/hotspots.csv`
**產生模組**：s02_hotspot.py
**筆數**：145（每列代表一個 DBSCAN 熱點）
**編碼**：UTF-8 with BOM（utf-8-sig，Excel 可直接開啟）

| 欄位 | 型別 | 空值數 | 說明 |
|---|---|---|---|
| cluster | int64 | 0 | 熱點編號（0–144），對應 accidents.parquet 的 cluster 欄位 |
| 件數 | int64 | 0 | 五年（2021–2025）熱點內總事故數。範圍：54–539 件，均值：157.6 |
| A1 | int64 | 0 | 五年內 A1（死亡）事故數。範圍：0–3，均值：0.38 |
| latitude | float64 | 0 | 熱點中心**緯度**（熱點內所有事故緯度的平均值）。kepler.gl 相容欄名 |
| longitude | float64 | 0 | 熱點中心**經度**（熱點內所有事故經度的平均值）。kepler.gl 相容欄名 |
| 地點 | object (str) | 0 | 熱點代表地點（熱點內 `肇事地點` 欄位的眾數） |
| dist_any | float64 | 0 | 熱點中心到**最近任意照相設備**的 haversine 距離（公尺）。範圍：16–1,839m，中位：359m |
| dist_speed | float64 | 0 | 熱點中心到**最近測速照相設備**的距離（公尺）。範圍：44–1,981m，中位：465m |
| cluster_type | float64 | 145（100%） | 熱點型態標籤（待 s05_pca_kmeans.py 執行後填入，如「通勤尖峰型」、「夜間型」等） |
| 件數_2021 | int64 | 0 | 2021 年熱點內事故數（範圍：9–102，均值：30.7） |
| 件數_2022 | int64 | 0 | 2022 年熱點內事故數（範圍：9–125，均值：33.5） |
| 件數_2023 | int64 | 0 | 2023 年熱點內事故數（範圍：10–121，均值：32.9） |
| 件數_2024 | int64 | 0 | 2024 年熱點內事故數（範圍：8–114，均值：29.6） |
| 件數_2025 | int64 | 0 | 2025 年熱點內事故數（範圍：7–95，均值：30.9） |

**備註**
- `cluster_type` 全部空值為預期行為，將由 s05 的 K-means 分群結果回填
- `dist_any` < 300m 之熱點視為「已覆蓋」，> 300m 則為「覆蓋缺口（Gap）」，供 s07 建議清單篩選使用
- Top 1 熱點：cluster=6，中正區羅斯福路4段與基隆路4段口，539 件，dist_any=915m（SRS AC-08 驗收基準）

---

## 5. daily_series.csv — 全市日事故數序列

**位置**：`data/processed/daily_series.csv`
**產生模組**：s03_series.py
**筆數**：1,826（2021-01-01 ~ 2025-12-31，每日一筆，無缺日）
**編碼**：UTF-8 with BOM

| 欄位 | 型別 | 空值數 | 說明 |
|---|---|---|---|
| date | object (YYYY-MM-DD) | 0 | 日期，完整涵蓋分析期間，無缺日（缺事故之日補 0） |
| count | int64 | 0 | 當日全市事故總數（A1+A2 合計）。範圍：7–124 件，日均：65.2 件 |

**備註**
- 1,826 天 = 365×4 + 366（2024 閏年）= 2021+2022+2023+2024+2025
- 最小值 7 件（非 0，整個分析期間每日皆有事故發生）
- 2021-05-19 起 COVID-19 三級警戒期間（至 2021-07-26）事故數明顯下降，s06 訓練時需以虛擬變數控制（SRS A-4）
- 供 s04_features.py 建立 lag/滾動特徵矩陣使用

---

## 6. hotspot_monthly.csv — 熱點月事故數 Panel

**位置**：`data/processed/hotspot_monthly.csv`
**產生模組**：s03_series.py
**Shape**：145 列（熱點）× 61 欄（1 個 cluster 欄 + 60 個月份欄）
**編碼**：UTF-8 with BOM

| 欄位 | 型別 | 說明 |
|---|---|---|
| cluster | int64 | 熱點編號（列索引），範圍 0–144 |
| 2021-01 ~ 2025-12 | int64（各 60 欄） | 各年月的熱點內事故數。缺事故之月補 0 |

**備註**
- 月欄格式為 `YYYY-MM`，完整覆蓋 60 個月（2021-01 到 2025-12）
- 各熱點月均值 2.6 件/月，最大單月 21 件
- 此 panel 為「全域 XGBoost 熱點月模型」（s06）的目標變數來源
- 單一熱點月均僅 4–9 件，樣本數不足以逐點建模，故採全域模型（SDD 4.6）

---

## 7. dist_vectors.csv — 時間指紋向量

**位置**：`data/processed/dist_vectors.csv`
**產生模組**：s03_series.py
**Shape**：145 列（熱點）× 44 欄（1 個 cluster 欄 + 43 個特徵欄）
**編碼**：UTF-8 with BOM

| 欄位群組 | 欄位名稱 | 維度 | 說明 |
|---|---|---|---|
| 索引 | cluster | 1 | 熱點編號 0–144 |
| 小時分布 | hour_0 ~ hour_23 | 24 | 各熱點在 0–23 時發生事故的**比例**（逐列正規化，加總=1.0 於各群組間） |
| 星期分布 | weekday_0 ~ weekday_6 | 7 | 各熱點在週一(0)至週日(6)發生事故的**比例** |
| 月份分布 | month_1 ~ month_12 | 12 | 各熱點在 1–12 月發生事故的**比例** |

**數值範圍**：0.0000 – 0.0868（比例值）

**備註**
- 43 維向量 = 24（小時）+ 7（星期）+ 12（月份）
- 逐列正規化：每列（每個熱點）的所有 43 維加總 = 1.0（已驗證）
- 此向量為 s05_pca_kmeans.py 的輸入，用於 PCA 降維後進行 K-means 分群，找出「通勤尖峰型」、「夜間型」等熱點型態
- 命名為「時間指紋（Time Fingerprint）」，代表各熱點獨特的時間發生分布模式

---

## 8. anomalies.csv — 清洗異常紀錄

**位置**：`outputs/anomalies.csv`
**產生模組**：s01_clean.py
**筆數**：3
**編碼**：UTF-8 with BOM（Excel 可直接開啟）

| 欄位 | 型別 | 說明 |
|---|---|---|
| 發生時間 | object (str) | 原始事故時間字串（尚未解析） |
| 處理別 | int64 | 事故嚴重程度（1=A1, 2=A2） |
| 肇事地點 | object (str) | 原始地點描述 |
| longitude | float64 | 原始經度值（本表全部為 NaN） |
| latitude | float64 | 原始緯度值（本表全部為 NaN） |
| 異常原因 | object (str) | 被剔除的具體原因說明 |

**異常紀錄明細**

| 發生時間 | 肇事地點 | 異常原因 |
|---|---|---|
| 2023/12/13 20:51 | 文山區木柵路3段34號 | 座標欄位為空值（NaN） |
| 2025/3/15 13:22 | 大同區太原路50號 | 座標欄位為空值（NaN） |
| 2025/12/13 14:27 | 大安區信義路3段與復興南路口三角公園 | 座標欄位為空值（NaN） |

**備註**
- 三筆均為原始資料座標欄位完全缺漏（longitude 與 latitude 皆為 NaN）
- 無法透過補值修復（無法判斷正確座標），直接剔除
- 清洗流程設計：Step A（座標 NaN）→ Step B（座標超出範圍）→ Step C（時間解析失敗）
- 本次資料僅觸發 Step A，Step B 與 Step C 無異常紀錄

---

## 附錄：資料血緣圖（Data Lineage）

```
data/raw/*.csv（原始政府開放資料）
      │
      ▼ s01_clean.py（FR-01）
      ├─ data/processed/accidents.parquet   （118,980 筆，5 欄）
      ├─ data/processed/cameras.parquet     （143 筆，12 欄）
      └─ outputs/anomalies.csv              （3 筆，6 欄）
                    │
                    ▼ s02_hotspot.py（FR-02）
                    ├─ data/processed/accidents.parquet  （更新：加入 cluster 欄）
                    └─ data/processed/hotspots.csv       （145 筆，14 欄）
                                  │
                                  ▼ s03_series.py（FR-03）
                                  ├─ data/processed/daily_series.csv      （1,826 筆）
                                  ├─ data/processed/hotspot_monthly.csv   （145×61）
                                  └─ data/processed/dist_vectors.csv      （145×44）
```
