from pathlib import Path

# ── 根目錄（此檔案在 src/，往上一層即專案根）
ROOT_DIR = Path(__file__).resolve().parent.parent

# ── 路徑
RAW_DIR  = ROOT_DIR / "data" / "raw"
PROC_DIR = ROOT_DIR / "data" / "processed"
OUT_DIR  = ROOT_DIR / "outputs"
FIG_DIR  = ROOT_DIR / "figures"
LOG_DIR  = ROOT_DIR / "outputs" / "logs"

# ── 原始檔名（FR-01）
ACC_FILES = [
    "110年臺北市道路交通事故斑點圖(改A1A2).csv",
    "111年臺北市道路交通事故斑點圖(改A1A2).csv",
    "112年臺北市道路交通事故斑點圖.csv",
    "113年臺北市道路交通事故斑點圖 .csv",
    "114年臺北市道路交通事故斑點圖.csv",
]
CAM_FILE = "臺北市政府警察局固定式違規照相設備及區間測速裝置設置地點一覽表.csv"

# ── 編碼（FR-01）
ENCODING_ACC = "cp950"
ENCODING_CAM = "big5"

# ── 座標範圍過濾（FR-01，WGS84）
LON_RANGE = (121.4, 121.7)
LAT_RANGE = (24.9,  25.25)

# ── DBSCAN 熱點（FR-02）
DBSCAN_EPS_M   = 50
DBSCAN_MIN_PTS = 80
EARTH_R_M      = 6_371_000

# ── 覆蓋缺口分析（FR-08）
GAP_RADIUS_M = 300
TOP_N        = 10

# ── 時間序列切分（FR-07，80/20 嚴格時序）
TRAIN_END   = "2024-12-31"
VALID_START = "2025-01-01"

# ── 第二階段外部資料（FR-10）
COVID_LV3_START = "2021-05-19"
COVID_LV3_END   = "2021-07-26"
HEAVY_RAIN_MM   = 40
CWA_STATION     = "466920"

# ── 可重現性（NFR-1）
RANDOM_STATE = 42
