# 軟體設計文件(SDD)
## 臺北市道路交通事故熱點風險預測與照相設備設置建議系統

| 項目 | 內容 |
|---|---|
| 文件版本 | v1.0 |
| 文件性質 | Software Design Document(參考 IEEE 1016 架構) |
| 對應需求文件 | SRS.md v1.0 |
| 開發語言 | Python 3.10+ |

---

## 1. 簡介

### 1.1 目的
本文件描述系統之架構設計、模組設計、資料設計與演算法設計,作為 SRS.md 各需求(FR-01~FR-10)之實作藍圖。每一設計元件均標注其對應之需求編號以確保可追溯性。

### 1.2 設計總覽
系統為單機批次資料分析管線,採「腳本式模組 + 中間檔案交換」架構:各模組(s01~s08)以檔案為介面依序執行,任一模組可獨立重跑而不影響上游產物,便於分工與除錯。

---

## 2. 系統架構設計

### 2.1 分層架構

```
┌─────────────────────────────────────────────────────┐
│ 輸出層  s07_recommend.py / s08_visualize.py          │
│         建議清單 CSV、圖表 PNG/HTML                   │
├─────────────────────────────────────────────────────┤
│ 模型層  s05_pca_kmeans.py / s06_forecast.py          │
│         PCA、K-means、SARIMA/XGBoost/LSTM、MAE/RMSE  │
├─────────────────────────────────────────────────────┤
│ 特徵層  s02_hotspot.py / s03_series.py /             │
│         s04_features.py                              │
│         DBSCAN 熱點、時間序列化、特徵矩陣              │
├─────────────────────────────────────────────────────┤
│ 資料層  s01_clean.py(+ 第二階段 s04 內之外部資料合併)│
│         cp950/Big5 讀取、清理、合併、標準化            │
└─────────────────────────────────────────────────────┘
```

### 2.2 資料流(Pipeline DFD)

```
IN-1..5 斑點圖 ──┐
                 ├─ s01_clean ──► accidents.parquet(118,980 筆)
IN-6 設備表 ─────┘                    │
                                      ▼
                              s02_hotspot ──► hotspots.csv(145 處)
                                      │              │
                ┌─────────────────────┤              │
                ▼                     ▼              │
        s03_series            s03_series             │
        daily_series.csv      hotspot_monthly.csv    │
        (1,826 天)            + dist_vectors.csv     │
                │             (145×43)               │
                ▼                     │              │
        s04_features                  ▼              │
        feature_matrix.csv    s05_pca_kmeans         │
        〔二階:+IN-7..9 join〕 pca_coords / clusters  │
                │                     │              │
                ▼                     │              │
        s06_forecast                  │              │
        metrics.csv、pred.csv         │              │
                └─────────┬───────────┴──────────────┘
                          ▼
                   s07_recommend ──► recommendations.csv(前10名)
                          │
                          ▼
                   s08_visualize ──► figures/F1..F9.png
```

### 2.3 目錄結構

```
project/
├─ data/raw/             # IN-1..IN-11 原始檔(唯讀)
├─ data/processed/       # accidents.parquet, hotspots.csv,
│                        # daily_series.csv, hotspot_monthly.csv,
│                        # dist_vectors.csv, feature_matrix.csv
├─ src/
│   ├─ config.py
│   ├─ utils.py          # haversine、log、IO 共用函式
│   ├─ s01_clean.py      # FR-01
│   ├─ s02_hotspot.py    # FR-02
│   ├─ s03_series.py     # FR-03
│   ├─ s04_features.py   # FR-04(第二階段擴充 FR-10)
│   ├─ s05_pca_kmeans.py # FR-05, FR-06
│   ├─ s06_forecast.py   # FR-07
│   ├─ s07_recommend.py  # FR-08
│   └─ s08_visualize.py  # FR-09
├─ figures/              # OUT-5
├─ outputs/              # OUT-2..OUT-4, logs/
├─ requirements.txt
└─ README.md             # 執行順序與環境說明
```

---

## 3. 設定設計(config.py)

```python
# 路徑
RAW_DIR        = "data/raw"
PROC_DIR       = "data/processed"
OUT_DIR        = "outputs"
FIG_DIR        = "figures"

# 清理(FR-01)
ENCODING_ACC   = "cp950"        # encoding_errors="replace"
ENCODING_CAM   = "big5"
LON_RANGE      = (121.4, 121.7)
LAT_RANGE      = (24.9, 25.25)

# 熱點(FR-02)
DBSCAN_EPS_M   = 50             # 公尺
DBSCAN_MIN_PTS = 80
EARTH_R_M      = 6_371_000

# 缺口(FR-08)
GAP_RADIUS_M   = 300
TOP_N          = 10

# 切分(FR-07)
TRAIN_END      = "2024-12-31"   # 前 80%
VALID_START    = "2025-01-01"   # 後 20%

# 第二階段(FR-10)
COVID_LV3      = ("2021-05-19", "2021-07-26")
HEAVY_RAIN_MM  = 40
CWA_STATION    = "466920"       # 臺北測站

RANDOM_STATE   = 42             # NFR-1
```

---

## 4. 模組詳細設計

### 4.1 s01_clean.py(FR-01)

| 函式 | 簽章 | 說明 |
|---|---|---|
| `load_accidents` | `(raw_dir) -> DataFrame` | glob 五份斑點圖,cp950 容錯讀取,統一欄名 `[發生時間, 處理別, 肇事地點, X, Y]`,縱向合併 |
| `clean_accidents` | `(df) -> DataFrame` | dropna 座標 → 範圍過濾 → `pd.to_datetime(errors="coerce")` → dropna 時間;每步寫 log |
| `load_cameras` | `(path) -> DataFrame` | Big5 讀取設備表;衍生布林欄 `is_speed = 功能.str.contains("測速")` |

輸出:`accidents.parquet`(UTF-8、datetime64 型別)、`cameras.parquet`。
錯誤處理:讀檔失敗即中止並回報檔名;剔除筆數 > 1% 時警告。

### 4.2 s02_hotspot.py(FR-02)

演算法:
```python
coords_rad = np.radians(acc[["Y", "X"]].values)
db = DBSCAN(eps=DBSCAN_EPS_M / EARTH_R_M,
            min_samples=DBSCAN_MIN_PTS,
            metric="haversine",
            algorithm="ball_tree").fit(coords_rad)
acc["cluster"] = db.labels_          # -1 = 噪音,保留
```

熱點彙總:groupby(cluster≥0)計算 `件數、A1 數、lat/lon 平均、地點眾數`。
設計理由:haversine + ball_tree 避免投影轉換;eps=50 m 約一個路口之涵蓋半徑;min_samples=80 即五年平均每年 16 件,過濾偶發點。
複雜度:O(n log n),n≈1.2×10⁵,單機數秒。

### 4.3 s03_series.py(FR-03)

| 輸出 | 產生方式 |
|---|---|
| `daily_series.csv` | `acc.set_index(發生時間).resample("D").size()`,reindex 完整日曆補 0(1,826 天) |
| `hotspot_monthly.csv` | cluster≥0 之 `pivot_table(index=cluster, columns=月, aggfunc=size)`,補 0(145×60) |
| `dist_vectors.csv` | 各熱點之 hour(24)+weekday(7)+month(12)計數,**逐列除以列總和**正規化 → 43 維比例向量 |

### 4.4 s04_features.py(FR-04;第二階段 FR-10)

第一階段特徵(對 daily_series):

| 特徵 | 實作 | 防洩漏 |
|---|---|---|
| lag_1/7/14/28 | `s.shift(k)` | shift 保證僅用過去 |
| ma_7/28, std_7/28 | `s.shift(1).rolling(w).mean()/.std()` | 先 shift(1) 再 rolling |
| weekday | one-hot(7) | 日曆屬性,無洩漏 |
| month | 整數或 one-hot | 同上 |

第二階段擴充(以日期 left join):

| 特徵 | 來源 | 衍生 |
|---|---|---|
| rain_mm, is_rain, heavy_rain, rain_streak | IN-7 CWA | heavy_rain = rain_mm > 40;rain_streak = 連續雨日計數 |
| is_holiday, is_makeup_work, holiday_day_n | IN-8 辦公日曆 | 補班日獨立旗標(不可僅用 weekday 推斷) |
| covid_lv3 | config 常數 | 2021/5/19–7/26 = 1 |
| mrt_volume | IN-9 | 標準化後使用;缺值以前後線性插補 |
| night_hours | astral 計算 | 依日期算臺北日出日落 |

熱點靜態特徵:`dist_to_speed_cam, dist_to_any_cam, district, speed_limit(IN-10), junction_volume(IN-11)`。
暖機期處理:特徵矩陣 dropna(前 28 天截斷),訓練/驗證切分於截斷後執行。

### 4.5 s05_pca_kmeans.py(FR-05, FR-06)

```
dist_vectors(145×43)
  → StandardScaler
  → PCA(n_components=10)         # 先看 scree,報告取 2–3 維
  → KMeans(k=2..8, random_state=42, n_init=10)
  → elbow(inertia)+ silhouette 選 k
  → 各群平均 24hr 曲線 → 人工命名型態
```

輸出:`pca_coords.csv`、`explained_variance.csv`、`loadings.csv`、`hotspot_clusters.csv(熱點→群)`、`cluster_profiles.csv(群→平均曲線)`。
型態→設備類型對映表(供 s07 使用):

| 型態(依群曲線命名) | 特徵 | 建議設備類型 |
|---|---|---|
| 夜間型 | 22:00–04:00 占比高 | 測速照相(深夜超速情境) |
| 通勤尖峰型 | 07–09 / 17–19 雙峰 | 闖紅燈照相 / 路口科技執法 |
| 全日型 | 分布平坦 | 綜合評估(以 A1 數加權) |
| 假日型 | 週六日占比高 | 行人安全導向科技執法 |

### 4.6 s06_forecast.py(FR-07)

切分(嚴格時間序):
```python
train = X[X.index <= TRAIN_END]      # 2021-01-01 ~ 2024-12-31
valid = X[X.index >= VALID_START]    # 2025-01-01 ~ 2025-12-31
```

模型設計:

| 模型 | 套件 | 設定 | 角色 |
|---|---|---|---|
| Naive(lag-7) | — | ŷ(t) = y(t−7) | 下限 baseline |
| SARIMA | statsmodels | s=7;(p,d,q)(P,D,Q) 以訓練期 AIC 網格搜尋(p,q≤2) | 統計 baseline |
| XGBoost | xgboost | n_estimators=500, lr=0.05, max_depth=5, early_stopping(訓練期尾端 10% 作內部驗證) | 主要挑戰者 |
| LSTM(選用) | keras | 輸入窗 28、單層 64 units、MinMaxScaler 僅以訓練期 fit | 深度學習比較 |
| 全域 XGBoost | xgboost | 樣本 =(熱點, 月);特徵 = 過去 12 個月 + 靜態特徵 | 熱點月模型(供排序) |

指標:`MAE、RMSE`,另計相對 Naive 改善率;第二階段重訓後輸出「加入外部特徵前/後」對照(供 F9)。
介面:`evaluate(y_true, y_pred) -> {"mae":…, "rmse":…}`;所有結果累加寫入 `outputs/metrics.csv`。

### 4.7 s07_recommend.py(FR-08)

```python
hot["dist_speed"] = min_haversine(hot, cam[cam.is_speed])
hot["dist_any"]   = min_haversine(hot, cam)
gap  = hot[hot.dist_any > GAP_RADIUS_M]
gap["risk"] = 預測風險(二階)或五年事故數(一階)
gap = gap.sort_values(["risk", "A1"], ascending=False).head(TOP_N)
gap["建議設備類型"] = gap.cluster_type.map(TYPE_MAPPING)   # 4.5 對映表
```

`min_haversine` 為向量化實作(numpy 廣播),置於 utils.py。
驗收基準(T-5):第 1 名 = 中正區羅斯福路4段與基隆路4段口(539 件、最近設備 915 m)。

### 4.8 s08_visualize.py(FR-09)

| 圖 | 實作 | 備註 |
|---|---|---|
| F1 密度 + 熱點 | matplotlib hexbin / folium HeatMap + 熱點標記 | 全市範圍 |
| F2 熱點 vs 設備疊圖 | folium:熱點圓(半徑∝事故數)+ 設備標記 + 缺口前 10 高亮 | HTML 互動地圖 |
| F3 A1/A2 比例 | 長條 + 百分比標注 | 呼應資料不平衡之視覺化觀察 |
| F4 日序列 | 折線 + COVID 期間灰底 axvspan | 標注三級警戒 |
| F5 PCA 散佈 + 群曲線 | 散佈(色=群)+ 各群平均 24hr 子圖 | |
| F6 預測 vs 實際 | 2025 驗證期折線疊圖 + 指標文字框 | 各模型一條 |
| F7 雨日 vs 非雨日箱型圖 | seaborn boxplot(二階) | |
| F8 假日 vs 平日箱型圖 | 同上,補班日獨立一類(二階) | |
| F9 指標前後對照 | 分組長條(二階) | |

中文字型:matplotlib 設 `rcParams["font.family"]`(Microsoft JhengHei / Noto Sans CJK),避免豆腐字。

---

## 5. 資料設計(中間檔 Schema)

### accidents.parquet
| 欄位 | 型別 | 說明 |
|---|---|---|
| 發生時間 | datetime64 | 分鐘精度 |
| 處理別 | int8 | 1=A1, 2=A2 |
| 肇事地點 | str | 行政區+路口描述 |
| X / Y | float64 | 經度 / 緯度(WGS84) |
| cluster | int16 | DBSCAN 標籤,−1=噪音(s02 後新增) |

### hotspots.csv
`cluster, 件數, A1, lat, lon, 地點, dist_speed, dist_any, cluster_type`

### feature_matrix.csv(日)
`date(index), y, lag_1, lag_7, lag_14, lag_28, ma_7, ma_28, std_7, std_28, weekday_0..6, month,`
`〔二階〕rain_mm, is_rain, heavy_rain, rain_streak, is_holiday, is_makeup_work, holiday_day_n, covid_lv3, mrt_volume, night_hours`

### recommendations.csv
`rank, 地點, lat, lon, 件數, A1, dist_any, trend_2021..2025, cluster_type, 建議設備類型, risk_score`

---

## 6. 設計決策與理由(Design Rationale)

| 決策 | 理由 | 替代方案(未採用原因) |
|---|---|---|
| DBSCAN 而非網格計數 | 不受網格切割邊界效應影響;噪音點自動排除;本身即課程要求之 clustering | KDE(無離散熱點清單)、固定網格(熱點被格線切開) |
| haversine 距離 | 避免座標投影(TWD97 轉換)額外相依 | 投影至公尺座標(增加 pyproj 相依) |
| 檔案交換式管線 | 組員可平行分工、單模組可重跑、中間結果可直接進報告 | 單一 notebook(難分工、易隱藏洩漏) |
| XGBoost 為主挑戰者 | 表格特徵 + 中等資料量下穩定優於 RNN;訓練快可多次迭代 | 僅 LSTM(訓練慢、調參成本高,列為選用) |
| 全域熱點月模型 | 單熱點月均僅 4–9 件,逐點建模過稀疏 | 145 個獨立模型(過擬合) |
| COVID dummy | 2021 三級警戒為已知結構斷點,不控制將汙染訓練 | 刪除該期間資料(損失樣本且破壞序列連續性) |
| 以 shift→rolling 造特徵 | 機械性防止未來資訊洩漏(SRS T-4) | 事後檢查(易遺漏) |

---

## 7. 可追溯矩陣(Requirements Traceability)

| SRS 需求 | 設計章節 | 模組 | 主要輸出 |
|---|---|---|---|
| FR-01 | 4.1 | s01_clean.py | accidents.parquet |
| FR-02 | 4.2 | s02_hotspot.py | hotspots.csv |
| FR-03 | 4.3 | s03_series.py | daily/monthly/dist_vectors |
| FR-04 | 4.4 | s04_features.py | feature_matrix.csv |
| FR-05 | 4.5 | s05_pca_kmeans.py | pca_coords, loadings |
| FR-06 | 4.5 | s05_pca_kmeans.py | hotspot_clusters.csv |
| FR-07 | 4.6 | s06_forecast.py | metrics.csv, pred.csv |
| FR-08 | 4.7 | s07_recommend.py | recommendations.csv |
| FR-09 | 4.8 | s08_visualize.py | figures/F1–F9 |
| FR-10 | 4.4(擴充) | s04_features.py | feature_matrix(二階版) |
| NFR-1 | 3 | config.py | RANDOM_STATE=42 |
| NFR-3 | 2.3, 3 | 全部 | 模組化 + config 集中 |
| NFR-4 | 4.1–4.8 | utils.py | outputs/logs/ |

---

## 8. 執行與部署

```bash
pip install -r requirements.txt
python src/s01_clean.py
python src/s02_hotspot.py
python src/s03_series.py
python src/s04_features.py        # 二階:--phase2 旗標啟用外部資料合併
python src/s05_pca_kmeans.py
python src/s06_forecast.py
python src/s07_recommend.py
python src/s08_visualize.py
```

requirements.txt(核心):`pandas, numpy, scikit-learn, statsmodels, xgboost, matplotlib, seaborn, folium, pyarrow, astral`;選用:`tensorflow`。
