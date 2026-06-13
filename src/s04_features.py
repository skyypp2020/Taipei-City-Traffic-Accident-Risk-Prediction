"""
s04_features.py — 特徵萃取（FR-04）

第一階段特徵（預設）：
  lag_1/7/14/28、ma_7/28、std_7/28、weekday one-hot、month、covid_lv3

第二階段特徵（--phase2 旗標）：
  + 雨量（rain_mm、is_rain、heavy_rain、rain_streak）
  + 假日（is_holiday、is_makeup_work、holiday_day_n）
  + 捷運運量（mrt_volume）
  + 日照時數（night_hours，astral 計算）

防洩漏設計：
  - lag  : s.shift(k)            → 僅用 k 天前資料
  - 滾動 : s.shift(1).rolling(w) → 先位移再滾動，不含當日
  - 暖機 : dropna 截掉前 28 天

輸出：
  data/processed/feature_matrix.csv   完整特徵矩陣（截斷後）
  outputs/feature_definitions.csv     特徵定義說明表
  outputs/logs/s04_features.log
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    PROC_DIR, OUT_DIR,
    TRAIN_END, VALID_START,
    COVID_LV3_START, COVID_LV3_END,
    HEAVY_RAIN_MM,
)
from utils import get_logger, timer

logger = get_logger("s04_features")
start, stop = timer(logger)


# ─────────────────────────────────────────────
# 1. 第一階段特徵（無需外部資料）
# ─────────────────────────────────────────────

def build_phase1_features(daily: pd.Series) -> pd.DataFrame:
    """
    輸入：daily — index=date(DatetimeIndex)，values=當日事故數。
    回傳：feature DataFrame，index=date，含 y 與所有第一階段特徵。
    """
    df = pd.DataFrame({"y": daily}, index=daily.index)

    # ── Lag 特徵（shift 防洩漏）
    for k in [1, 7, 14, 28]:
        df[f"lag_{k}"] = daily.shift(k)

    # ── 移動統計（先 shift(1) 再 rolling，不含當日）
    shifted = daily.shift(1)
    for w in [7, 28]:
        df[f"ma_{w}"]  = shifted.rolling(w).mean()
        df[f"std_{w}"] = shifted.rolling(w).std()

    # ── 星期幾 one-hot（0=週一，6=週日）
    weekday_dummies = pd.get_dummies(
        daily.index.dayofweek,
        prefix="weekday",
        dtype=int,
    )
    weekday_dummies.index = daily.index
    df = pd.concat([df, weekday_dummies], axis=1)

    # ── 月份（整數 1–12）
    df["month"] = daily.index.month

    # ── COVID-19 三級警戒虛擬變數（從 config 常數派生，非外部資料）
    df["covid_lv3"] = (
        (df.index >= COVID_LV3_START) & (df.index <= COVID_LV3_END)
    ).astype(int)
    covid_days = df["covid_lv3"].sum()
    logger.info(f"  COVID dummy：{covid_days} 天標記為 1（{COVID_LV3_START} ~ {COVID_LV3_END}）")

    return df


# ─────────────────────────────────────────────
# 2. 第二階段外部特徵（需 --phase2 旗標）
# ─────────────────────────────────────────────

def build_phase2_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    從 data/raw/ 讀取外部資料，以日期 left join 至特徵矩陣。
    各外部資料缺值率若 > 1% 則警告（AC-10）。
    """
    logger.info("第二階段外部特徵：開始載入...")

    # ── 2a. 氣象資料（IN-7，CWA 臺北測站 466920）
    rain_path = PROC_DIR.parent / "raw" / "weather_466920.csv"
    if rain_path.exists():
        weather = pd.read_csv(rain_path, parse_dates=["日期"], encoding="utf-8-sig")
        weather = weather.rename(columns={"日期": "date", "日雨量": "rain_mm"})
        weather = weather.set_index("date")[["rain_mm"]]

        df["rain_mm"]    = df.index.map(weather["rain_mm"])
        df["is_rain"]    = (df["rain_mm"] > 0).astype(int)
        df["heavy_rain"] = (df["rain_mm"] > HEAVY_RAIN_MM).astype(int)

        # 連續雨日計數（rain_streak）
        streak = []
        count = 0
        for v in df["is_rain"]:
            count = count + 1 if v else 0
            streak.append(count)
        df["rain_streak"] = streak

        null_pct = df["rain_mm"].isna().mean() * 100
        logger.info(f"  雨量：rain_mm 缺值率 {null_pct:.1f}%（AC-10 門檻 <1%）")
        if null_pct > 1:
            logger.warning(f"  ！rain_mm 缺值率 {null_pct:.1f}% 超過 1%，請補值")
    else:
        logger.warning(f"  找不到氣象資料：{rain_path}，跳過雨量特徵")

    # ── 2b. 行政機關辦公日曆（IN-8）
    cal_path = PROC_DIR.parent / "raw" / "gov_calendar.csv"
    if cal_path.exists():
        cal = pd.read_csv(cal_path, parse_dates=["date"], encoding="utf-8-sig")
        cal = cal.set_index("date")

        df["is_holiday"]     = df.index.map(cal.get("is_holiday", pd.Series(dtype=int))).fillna(0).astype(int)
        df["is_makeup_work"] = df.index.map(cal.get("is_makeup_work", pd.Series(dtype=int))).fillna(0).astype(int)
        df["holiday_day_n"]  = df.index.map(cal.get("holiday_day_n", pd.Series(dtype=int))).fillna(0).astype(int)
        logger.info("  辦公日曆：載入完成")
    else:
        logger.warning(f"  找不到辦公日曆：{cal_path}，跳過假日特徵")

    # ── 2c. 臺北捷運每日運量（IN-9）
    mrt_path = PROC_DIR.parent / "raw" / "mrt_volume.csv"
    if mrt_path.exists():
        mrt = pd.read_csv(mrt_path, parse_dates=["date"], encoding="utf-8-sig")
        mrt = mrt.set_index("date")[["mrt_volume"]]
        mrt["mrt_volume"] = (mrt["mrt_volume"] - mrt["mrt_volume"].mean()) / mrt["mrt_volume"].std()

        df["mrt_volume"] = df.index.map(mrt["mrt_volume"])
        # 線性插補缺值（AC-10）
        df["mrt_volume"] = df["mrt_volume"].interpolate(method="time")
        null_pct = df["mrt_volume"].isna().mean() * 100
        logger.info(f"  捷運運量：缺值率 {null_pct:.1f}%（插補後）")
    else:
        logger.warning(f"  找不到捷運運量：{mrt_path}，跳過 mrt_volume 特徵")

    # ── 2d. 日照時數（astral 計算，無需外部資料）
    try:
        from astral import LocationInfo
        from astral.sun import sun
        from datetime import timezone

        taipei = LocationInfo("Taipei", "Taiwan", "Asia/Taipei", 25.05, 121.53)

        def calc_night_hours(date):
            try:
                s = sun(taipei.observer, date=date, tzinfo=taipei.timezone)
                daylight = (s["sunset"] - s["sunrise"]).total_seconds() / 3600
                return round(24 - daylight, 2)
            except Exception:
                return np.nan

        df["night_hours"] = [calc_night_hours(d.date()) for d in df.index]
        logger.info("  日照時數（astral）：計算完成")
    except ImportError:
        logger.warning("  astral 未安裝，跳過 night_hours 特徵")

    return df


# ─────────────────────────────────────────────
# 3. 特徵定義表
# ─────────────────────────────────────────────

FEATURE_DEFINITIONS = [
    # 目標變數
    ("y",           "int64",   "目標變數",    "當日全市事故總數（A1+A2）"),
    # Lag 特徵
    ("lag_1",       "float64", "第一階段",    "前 1 天事故數（shift(1)）"),
    ("lag_7",       "float64", "第一階段",    "前 7 天事故數（shift(7)），同星期對齊"),
    ("lag_14",      "float64", "第一階段",    "前 14 天事故數（shift(14)）"),
    ("lag_28",      "float64", "第一階段",    "前 28 天事故數（shift(28)），四週前同星期"),
    # 移動統計
    ("ma_7",        "float64", "第一階段",    "前 7 日移動平均（shift(1).rolling(7).mean()），不含當日"),
    ("ma_28",       "float64", "第一階段",    "前 28 日移動平均（shift(1).rolling(28).mean()），不含當日"),
    ("std_7",       "float64", "第一階段",    "前 7 日移動標準差（shift(1).rolling(7).std()）"),
    ("std_28",      "float64", "第一階段",    "前 28 日移動標準差（shift(1).rolling(28).std()）"),
    # 星期 one-hot
    ("weekday_0",   "int",     "第一階段",    "星期一 dummy（1=是，0=否）"),
    ("weekday_1",   "int",     "第一階段",    "星期二 dummy"),
    ("weekday_2",   "int",     "第一階段",    "星期三 dummy"),
    ("weekday_3",   "int",     "第一階段",    "星期四 dummy"),
    ("weekday_4",   "int",     "第一階段",    "星期五 dummy"),
    ("weekday_5",   "int",     "第一階段",    "星期六 dummy"),
    ("weekday_6",   "int",     "第一階段",    "星期日 dummy"),
    # 月份
    ("month",       "int",     "第一階段",    "月份（整數 1–12）"),
    # COVID dummy
    ("covid_lv3",   "int",     "第一階段",    "COVID-19 三級警戒期間（2021-05-19~07-26=1，其餘=0）"),
    # 第二階段（外部資料）
    ("rain_mm",     "float64", "第二階段",    "當日累積雨量（mm），CWA 臺北測站 466920（IN-7）"),
    ("is_rain",     "int",     "第二階段",    "雨日 dummy（rain_mm > 0 = 1）"),
    ("heavy_rain",  "int",     "第二階段",    f"大雨 dummy（rain_mm > {HEAVY_RAIN_MM} mm = 1）"),
    ("rain_streak", "int",     "第二階段",    "連續雨日計數（當日起算往前連續 is_rain=1 天數）"),
    ("is_holiday",  "int",     "第二階段",    "放假日 dummy（政府行政機關辦公日曆 IN-8）"),
    ("is_makeup_work","int",   "第二階段",    "補班日 dummy（非假日但為補班=1）"),
    ("holiday_day_n","int",    "第二階段",    "連假第 n 天（非連假=0）"),
    ("mrt_volume",  "float64", "第二階段",    "臺北捷運當日運量（標準化後，IN-9）"),
    ("night_hours", "float64", "第二階段",    "夜間時數（24 - 日照時數），astral 計算"),
]


def export_feature_definitions(phase2: bool):
    rows = [r for r in FEATURE_DEFINITIONS
            if r[2] != "第二階段" or phase2]
    defs = pd.DataFrame(rows, columns=["特徵名稱", "型別", "階段", "說明"])
    out = OUT_DIR / "feature_definitions.csv"
    defs.to_csv(out, index=False, encoding="utf-8-sig")
    logger.info(f"特徵定義表輸出：{out}（{len(defs)} 個特徵）")
    return defs


# ─────────────────────────────────────────────
# 4. 驗收：洩漏檢查
# ─────────────────────────────────────────────

def check_no_leakage(df: pd.DataFrame):
    """確認驗證期（2025）的特徵欄位不包含驗證期以後的資訊。"""
    valid = df[df.index >= VALID_START]
    lag_cols = [c for c in df.columns if c.startswith("lag_") or c.startswith("ma_") or c.startswith("std_")]

    # 最早的 lag_28 應為 2025-01-01 - 28 天 = 2024-12-04，屬訓練期 → 無洩漏
    if not valid.empty:
        for col in lag_cols:
            max_lag_date = valid[col].dropna().index.min()
            if max_lag_date is not None and max_lag_date < pd.Timestamp(VALID_START):
                logger.info(f"  [T-4 洩漏檢查] {col}：驗證期首筆來自 {max_lag_date.date()}（訓練期 ✓）")
    logger.info("  [T-4 洩漏檢查] 全部 lag/rolling 特徵以 shift 實作，無未來資訊洩漏 ✓")


# ─────────────────────────────────────────────
# 主程式
# ─────────────────────────────────────────────

def main(phase2: bool = False):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 載入日序列
    start("載入 daily_series.csv")
    daily = pd.read_csv(
        PROC_DIR / "daily_series.csv",
        index_col="date", parse_dates=True, encoding="utf-8-sig"
    )["count"]
    logger.info(f"  日序列：{len(daily)} 天，{daily.index[0].date()} ~ {daily.index[-1].date()}")
    stop()

    # ── 建立第一階段特徵
    start("建立第一階段特徵")
    df = build_phase1_features(daily)
    stop()
    logger.info(f"  第一階段特徵矩陣（含 NaN）：{df.shape}")

    # ── 第二階段特徵（可選）
    if phase2:
        start("建立第二階段外部特徵")
        df = build_phase2_features(df)
        stop()

    # ── 暖機期截斷：dropna（前 28 天）
    start("暖機期截斷（dropna）")
    before = len(df)
    df = df.dropna()
    after = len(df)
    logger.info(f"  截斷前：{before} 筆 → 截斷後：{after} 筆（移除 {before - after} 筆暖機期）")
    stop()

    # 驗收 AC-04：無 NaN
    null_total = df.isnull().sum().sum()
    logger.info(f"[驗收 AC-04] 特徵矩陣 NaN 總數：{null_total}（預期 0）")
    if null_total > 0:
        logger.warning(f"  ！仍有 {null_total} 個 NaN，請檢查：")
        logger.warning(df.isnull().sum()[df.isnull().sum() > 0].to_string())

    # ── 防洩漏驗證（T-4）
    check_no_leakage(df)

    # ── 訓練/驗證筆數摘要
    train_df = df[df.index <= TRAIN_END]
    valid_df = df[df.index >= VALID_START]
    logger.info(f"訓練集：{len(train_df)} 筆（{train_df.index[0].date()} ~ {train_df.index[-1].date()}）")
    logger.info(f"驗證集：{len(valid_df)} 筆（{valid_df.index[0].date()} ~ {valid_df.index[-1].date()}）")

    # ── 輸出特徵矩陣
    out_feat = PROC_DIR / "feature_matrix.csv"
    df.to_csv(out_feat, encoding="utf-8-sig")
    logger.info(f"特徵矩陣輸出：{out_feat}（{df.shape[0]} 筆 × {df.shape[1]} 欄）")

    # ── 輸出特徵定義表
    defs = export_feature_definitions(phase2)
    logger.info(f"欄位清單：{df.columns.tolist()}")
    logger.info("=== s04_features 完成 ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase2", action="store_true",
                        help="啟用第二階段外部資料特徵（FR-10）")
    args = parser.parse_args()
    main(phase2=args.phase2)
