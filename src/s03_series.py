"""
s03_series.py — 時間序列化（FR-03）

輸出：
  data/processed/daily_series.csv       全市日事故數（1,826 天）
  data/processed/hotspot_monthly.csv    熱點×月事故數 panel（145×60）
  data/processed/dist_vectors.csv       各熱點時間指紋向量（145×43）
  outputs/logs/s03_series.log
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from config import PROC_DIR
from utils import get_logger, timer

logger = get_logger("s03_series")
start, stop = timer(logger)

# 分析期完整日曆範圍
SERIES_START = "2021-01-01"
SERIES_END   = "2025-12-31"


# ─────────────────────────────────────────────
# 1. 全市日事故數序列
# ─────────────────────────────────────────────

def build_daily_series(acc: pd.DataFrame) -> pd.Series:
    """
    全市所有事故（含噪音點）依日期聚合，
    reindex 補 0 確保 1,826 天無缺日。
    回傳 index=date(DatetimeIndex)、name='count' 的 Series。
    """
    daily = (
        acc.set_index("發生時間")
        .resample("D")
        .size()
        .rename("count")
    )

    full_idx = pd.date_range(SERIES_START, SERIES_END, freq="D")
    daily = daily.reindex(full_idx, fill_value=0)
    daily.index.name = "date"

    logger.info(f"日序列長度：{len(daily)}（預期 1,826）")
    logger.info(f"  日均事故：{daily.mean():.1f} 件，最大：{daily.max()} 件，零件日數：{(daily==0).sum()} 天")
    return daily


# ─────────────────────────────────────────────
# 2. 熱點 × 月事故數 panel
# ─────────────────────────────────────────────

def build_hotspot_monthly(acc: pd.DataFrame) -> pd.DataFrame:
    """
    僅使用熱點內事故（cluster >= 0）。
    pivot_table：rows=cluster、columns=年月(YYYY-MM)，補 0。
    輸出 shape：145 rows × 60 cols（2021-01 ~ 2025-12）。
    """
    hot = acc[acc["cluster"] >= 0].copy()
    hot["年月"] = hot["發生時間"].dt.to_period("M").astype(str)

    panel = (
        hot.groupby(["cluster", "年月"])
        .size()
        .unstack(fill_value=0)
    )

    # 補齊所有月份（60 個月）
    all_months = pd.period_range(SERIES_START, SERIES_END, freq="M").astype(str)
    panel = panel.reindex(columns=all_months, fill_value=0)

    # 補齊所有熱點（145 個）
    all_clusters = sorted(acc[acc["cluster"] >= 0]["cluster"].unique())
    panel = panel.reindex(index=all_clusters, fill_value=0)

    logger.info(f"熱點月序列 shape：{panel.shape}（預期 145×60）")
    logger.info(f"  月均事故(各熱點)：{panel.values.mean():.1f} 件/月，最大：{panel.values.max()} 件")
    return panel


# ─────────────────────────────────────────────
# 3. 時間指紋向量（43 維）
# ─────────────────────────────────────────────

def build_dist_vectors(acc: pd.DataFrame) -> pd.DataFrame:
    """
    各熱點之 24hr + 7 weekday + 12 month 計數向量，逐列正規化為比例。
    欄位命名：hour_0..23、weekday_0..6（0=週一）、month_1..12。
    輸出 shape：145 rows × 43 cols，index = cluster。
    """
    hot = acc[acc["cluster"] >= 0].copy()
    hot["hour"]    = hot["發生時間"].dt.hour
    hot["weekday"] = hot["發生時間"].dt.dayofweek   # 0=週一
    hot["month"]   = hot["發生時間"].dt.month

    frames = {}

    # 24 小時分布
    hour_pivot = (
        hot.groupby(["cluster", "hour"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=range(24), fill_value=0)
    )
    hour_pivot.columns = [f"hour_{h}" for h in range(24)]
    frames["hour"] = hour_pivot

    # 7 weekday 分布
    wd_pivot = (
        hot.groupby(["cluster", "weekday"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=range(7), fill_value=0)
    )
    wd_pivot.columns = [f"weekday_{d}" for d in range(7)]
    frames["weekday"] = wd_pivot

    # 12 month 分布
    mo_pivot = (
        hot.groupby(["cluster", "month"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=range(1, 13), fill_value=0)
    )
    mo_pivot.columns = [f"month_{m}" for m in range(1, 13)]
    frames["month"] = mo_pivot

    # 合併成 43 維向量
    vectors = pd.concat([frames["hour"], frames["weekday"], frames["month"]], axis=1)

    # 逐列正規化（各列總和 = 1）
    row_sums = vectors.sum(axis=1)
    vectors = vectors.div(row_sums, axis=0)

    # 補齊所有熱點
    all_clusters = sorted(acc[acc["cluster"] >= 0]["cluster"].unique())
    vectors = vectors.reindex(index=all_clusters, fill_value=0)
    vectors.index.name = "cluster"

    logger.info(f"時間指紋向量 shape：{vectors.shape}（預期 145×43）")
    logger.info(f"  列加總驗證（應全為 1.0）：min={vectors.sum(axis=1).min():.4f}，max={vectors.sum(axis=1).max():.4f}")
    return vectors


# ─────────────────────────────────────────────
# 主程式
# ─────────────────────────────────────────────

def main():
    start("載入 accidents.parquet")
    acc = pd.read_parquet(PROC_DIR / "accidents.parquet")
    logger.info(f"  事故：{len(acc):,} 筆，熱點內：{(acc['cluster']>=0).sum():,} 筆")
    stop()

    # ── 全市日序列
    start("建立全市日事故數序列")
    daily = build_daily_series(acc)
    stop()

    out_daily = PROC_DIR / "daily_series.csv"
    daily.to_csv(out_daily, header=True, encoding="utf-8-sig")
    logger.info(f"輸出：{out_daily}")

    # ── 熱點月 panel
    start("建立熱點×月事故數 panel")
    panel = build_hotspot_monthly(acc)
    stop()

    out_panel = PROC_DIR / "hotspot_monthly.csv"
    panel.to_csv(out_panel, encoding="utf-8-sig")
    logger.info(f"輸出：{out_panel}")

    # ── 時間指紋向量
    start("建立時間指紋向量（43 維）")
    vectors = build_dist_vectors(acc)
    stop()

    out_vec = PROC_DIR / "dist_vectors.csv"
    vectors.to_csv(out_vec, encoding="utf-8-sig")
    logger.info(f"輸出：{out_vec}")

    # ── 驗收摘要
    logger.info("=== 驗收 AC-03 ===")
    logger.info(f"  日序列長度：{len(daily)} == 1826? {len(daily)==1826}")
    logger.info(f"  日序列缺日：{(daily==0).sum()} 天補零（非缺日，為真實零件天）")
    logger.info(f"  Panel shape：{panel.shape} == (145, 60)? {panel.shape==(145,60)}")
    logger.info(f"  向量 shape ：{vectors.shape} == (145, 43)? {vectors.shape==(145,43)}")
    logger.info("=== s03_series 完成 ===")


if __name__ == "__main__":
    main()
