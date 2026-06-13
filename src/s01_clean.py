"""
s01_clean.py — 資料載入與清理（FR-01）

輸出：
  data/processed/accidents.parquet   清理後事故主檔（118,980 筆）
  data/processed/cameras.parquet     照相設備表
  outputs/anomalies.csv              被剔除紀錄 + 異常原因
  outputs/logs/s01_clean.log         執行 log
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── 讓 src/ 內各模組互相 import 時找得到 config / utils
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    RAW_DIR, PROC_DIR, OUT_DIR,
    ACC_FILES, CAM_FILE,
    ENCODING_ACC, ENCODING_CAM,
    LON_RANGE, LAT_RANGE,
)
from utils import get_logger, timer

logger = get_logger("s01_clean")
start, stop = timer(logger)


# ─────────────────────────────────────────────
# 1. 載入事故斑點圖
# ─────────────────────────────────────────────

def load_accidents(raw_dir: Path) -> pd.DataFrame:
    """讀取五份斑點圖並縱向合併，統一欄名為標準格式。"""
    frames = []
    for fname in ACC_FILES:
        path = raw_dir / fname
        if not path.exists():
            logger.error(f"找不到檔案：{path}")
            raise FileNotFoundError(path)
        df = pd.read_csv(path, encoding=ENCODING_ACC, encoding_errors="replace")
        logger.info(f"  讀入 {fname}：{len(df):,} 筆，欄位={df.columns.tolist()}")
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    logger.info(f"合併後共 {len(combined):,} 筆")

    # 統一欄名：原始欄名 → 標準欄名
    col_map = {
        combined.columns[0]: "發生時間",
        combined.columns[1]: "處理別",
        combined.columns[2]: "肇事地點",
        combined.columns[3]: "longitude",   # 座標-X = 經度（kepler.gl 相容）
        combined.columns[4]: "latitude",    # 座標-Y = 緯度（kepler.gl 相容）
    }
    combined = combined.rename(columns=col_map)
    return combined


# ─────────────────────────────────────────────
# 2. 清理事故資料，同步記錄異常紀錄
# ─────────────────────────────────────────────

def clean_accidents(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    清理流程（依序）：
      Step A — 座標欄位 NaN
      Step B — 座標超出臺北市範圍
      Step C — 發生時間解析失敗

    回傳 (clean_df, anomalies_df)
    """
    anomaly_rows = []

    def mark_anomaly(subset: pd.DataFrame, reason: str):
        tmp = subset.copy()
        tmp["異常原因"] = reason
        anomaly_rows.append(tmp)

    # Step A：座標缺漏
    mask_coord_null = df["longitude"].isna() | df["latitude"].isna()
    if mask_coord_null.sum() > 0:
        mark_anomaly(df[mask_coord_null], "座標欄位為空值（NaN）")
        logger.info(f"  [Step A] 座標缺漏剔除：{mask_coord_null.sum():,} 筆")
    df = df[~mask_coord_null].copy()

    # Step B：座標超出範圍
    mask_out_of_range = ~(
        df["longitude"].between(*LON_RANGE) &
        df["latitude"].between(*LAT_RANGE)
    )
    if mask_out_of_range.sum() > 0:
        detail = df[mask_out_of_range][["發生時間", "肇事地點", "longitude", "latitude"]].copy()
        detail["異常說明"] = (
            "longitude=" + df[mask_out_of_range]["longitude"].astype(str) +
            "（合法範圍 " + f"{LON_RANGE[0]}–{LON_RANGE[1]}" + "），"
            "latitude=" + df[mask_out_of_range]["latitude"].astype(str) +
            "（合法範圍 " + f"{LAT_RANGE[0]}–{LAT_RANGE[1]}" + "）"
        )
        mark_anomaly(df[mask_out_of_range].assign(異常說明=detail["異常說明"]), "座標超出臺北市合法範圍")
        logger.info(f"  [Step B] 座標超出範圍剔除：{mask_out_of_range.sum():,} 筆")
    df = df[~mask_out_of_range].copy()

    # Step C：時間解析失敗
    df["發生時間"] = pd.to_datetime(df["發生時間"], errors="coerce")
    mask_bad_time = df["發生時間"].isna()
    if mask_bad_time.sum() > 0:
        mark_anomaly(df[mask_bad_time], "發生時間欄位無法解析為日期時間格式")
        logger.info(f"  [Step C] 時間解析失敗剔除：{mask_bad_time.sum():,} 筆")
    df = df[~mask_bad_time].copy()

    # 確保型別正確
    df["處理別"] = df["處理別"].astype("int8")
    df["longitude"] = df["longitude"].astype("float64")
    df["latitude"] = df["latitude"].astype("float64")

    logger.info(f"清理完成，保留 {len(df):,} 筆")

    # 合併所有異常紀錄
    if anomaly_rows:
        anomalies = pd.concat(anomaly_rows, ignore_index=True)
        # 統一欄位順序
        base_cols = ["發生時間", "處理別", "肇事地點", "longitude", "latitude", "異常原因"]
        extra_cols = [c for c in anomalies.columns if c not in base_cols]
        anomalies = anomalies[base_cols + extra_cols]
    else:
        anomalies = pd.DataFrame(columns=["發生時間", "處理別", "肇事地點",
                                          "longitude", "latitude", "異常原因"])

    return df, anomalies


# ─────────────────────────────────────────────
# 3. 載入照相設備表
# ─────────────────────────────────────────────

def load_cameras(raw_dir: Path) -> pd.DataFrame:
    """讀取固定式照相設備一覽表，欄名統一為英文（kepler.gl 相容）。"""
    path = raw_dir / CAM_FILE
    df = pd.read_csv(path, encoding=ENCODING_CAM, encoding_errors="replace")
    logger.info(f"設備表讀入：{len(df):,} 筆，欄位={df.columns.tolist()}")

    # 統一欄名
    col_map = {
        df.columns[0]: "編號",
        df.columns[1]: "功能",
        df.columns[2]: "設置路段",
        df.columns[3]: "設置地點",
        df.columns[4]: "latitude",    # 緯度 → kepler.gl 相容
        df.columns[5]: "longitude",   # 經度 → kepler.gl 相容
        df.columns[6]: "速限",
    }
    df = df.rename(columns=col_map)

    # 衍生欄位：是否為測速照相
    df["is_speed"] = df["功能"].str.contains("測速", na=False)
    logger.info(f"  測速設備：{df['is_speed'].sum()} 台 / 總計 {len(df)} 台")

    return df


# ─────────────────────────────────────────────
# 主程式
# ─────────────────────────────────────────────

def main():
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 事故資料
    start("載入五份事故斑點圖")
    raw_acc = load_accidents(RAW_DIR)
    stop()

    start("清理事故資料")
    clean_acc, anomalies = clean_accidents(raw_acc)
    stop()

    # 驗收 AC-01a
    logger.info(f"[驗收 AC-01a] 清理後筆數：{len(clean_acc):,}（預期 118,980）")
    if len(clean_acc) != 118_980:
        logger.warning(f"  ！筆數與預期不符，請檢查原始資料")

    # 輸出 accidents.parquet
    out_acc = PROC_DIR / "accidents.parquet"
    clean_acc.to_parquet(out_acc, index=False)
    logger.info(f"輸出：{out_acc}")

    # 輸出 anomalies.csv（AC-01b）
    out_anom = OUT_DIR / "anomalies.csv"
    anomalies.to_csv(out_anom, index=False, encoding="utf-8-sig")
    logger.info(f"輸出異常紀錄：{out_anom}（{len(anomalies):,} 筆）")

    # ── 照相設備
    start("載入照相設備表")
    cameras = load_cameras(RAW_DIR)
    stop()

    out_cam = PROC_DIR / "cameras.parquet"
    cameras.to_parquet(out_cam, index=False)
    logger.info(f"輸出：{out_cam}")

    logger.info("=== s01_clean 完成 ===")


if __name__ == "__main__":
    main()
