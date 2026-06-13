"""
s02_hotspot.py — 空間熱點辨識（FR-02）

輸出：
  data/processed/accidents.parquet   加入 cluster 欄位（-1 = 噪音）
  data/processed/hotspots.csv        145 個熱點摘要 + 設備距離
  outputs/logs/s02_hotspot.log
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    PROC_DIR, OUT_DIR,
    DBSCAN_EPS_M, DBSCAN_MIN_PTS, EARTH_R_M,
)
from utils import get_logger, min_haversine, timer

logger = get_logger("s02_hotspot")
start, stop = timer(logger)


# ─────────────────────────────────────────────
# 1. DBSCAN 熱點辨識
# ─────────────────────────────────────────────

def run_dbscan(acc: pd.DataFrame) -> pd.DataFrame:
    """
    對事故座標執行 DBSCAN（haversine + ball_tree）。
    注意：haversine 要求輸入為 [latitude, longitude] 弧度順序。
    回傳加上 cluster 欄位的 DataFrame（-1 = 噪音）。
    """
    # haversine 要求 [lat, lon] 弧度
    coords_rad = np.radians(acc[["latitude", "longitude"]].values)

    db = DBSCAN(
        eps=DBSCAN_EPS_M / EARTH_R_M,   # 公尺 → 弧度
        min_samples=DBSCAN_MIN_PTS,
        metric="haversine",
        algorithm="ball_tree",
        n_jobs=-1,
    ).fit(coords_rad)

    acc = acc.copy()
    acc["cluster"] = db.labels_.astype("int16")

    n_hotspots = (acc["cluster"] >= 0).sum()
    n_clusters = acc["cluster"].nunique() - (1 if -1 in acc["cluster"].values else 0)
    noise_ratio = (acc["cluster"] == -1).mean()

    logger.info(f"DBSCAN 完成：{n_clusters} 個熱點，熱點內事故 {n_hotspots:,} 筆")
    logger.info(f"  熱點內占比：{n_hotspots / len(acc) * 100:.1f}%（預期 19.2%）")
    logger.info(f"  噪音點：{(acc['cluster'] == -1).sum():,} 筆（占 {noise_ratio * 100:.1f}%）")

    return acc


# ─────────────────────────────────────────────
# 2. 熱點摘要統計
# ─────────────────────────────────────────────

def build_hotspot_summary(acc: pd.DataFrame) -> pd.DataFrame:
    """
    對每個熱點（cluster >= 0）計算：
      件數、A1 數、中心 latitude/longitude、代表地點（眾數）
    """
    hot = acc[acc["cluster"] >= 0].copy()

    summary = hot.groupby("cluster").agg(
        件數=("cluster", "count"),
        A1=("處理別", lambda x: (x == 1).sum()),
        latitude=("latitude", "mean"),
        longitude=("longitude", "mean"),
        地點=("肇事地點", lambda x: x.mode().iloc[0] if len(x) > 0 else ""),
    ).reset_index()

    # 逐年事故數（供 s07 趨勢欄使用）
    hot["年份"] = hot["發生時間"].dt.year
    yearly = (
        hot.groupby(["cluster", "年份"])
        .size()
        .unstack(fill_value=0)
        .rename(columns=lambda y: f"件數_{y}")
    )
    summary = summary.merge(yearly, on="cluster", how="left")

    logger.info(f"熱點摘要建立：{len(summary)} 個熱點")
    return summary


# ─────────────────────────────────────────────
# 3. 計算各熱點到照相設備的距離
# ─────────────────────────────────────────────

def add_camera_distances(hotspots: pd.DataFrame, cameras: pd.DataFrame) -> pd.DataFrame:
    """
    以向量化 haversine 計算各熱點到最近設備的距離（公尺）。
    utils.min_haversine 需要 DataFrame 含 lat, lon 欄位。
    """
    # min_haversine 需要 lat/lon 欄名，建立暫時別名 view
    hot_coords = hotspots.rename(columns={"latitude": "lat", "longitude": "lon"})
    cam_all    = cameras.rename(columns={"latitude": "lat", "longitude": "lon"})
    cam_speed  = cameras[cameras["is_speed"]].rename(columns={"latitude": "lat", "longitude": "lon"})

    hotspots = hotspots.copy()
    hotspots["dist_any"]   = min_haversine(hot_coords, cam_all).round(1)
    hotspots["dist_speed"] = min_haversine(hot_coords, cam_speed).round(1)

    logger.info(
        f"設備距離計算完成："
        f"中位距離(任意)={hotspots['dist_any'].median():.0f}m，"
        f"中位距離(測速)={hotspots['dist_speed'].median():.0f}m"
    )
    return hotspots


# ─────────────────────────────────────────────
# 主程式
# ─────────────────────────────────────────────

def main():
    # ── 載入
    start("載入 accidents.parquet 與 cameras.parquet")
    acc = pd.read_parquet(PROC_DIR / "accidents.parquet")
    cam = pd.read_parquet(PROC_DIR / "cameras.parquet")
    logger.info(f"  事故：{len(acc):,} 筆 / 設備：{len(cam)} 台")
    stop()

    # ── DBSCAN
    start("DBSCAN 熱點辨識")
    acc = run_dbscan(acc)
    stop()

    # 驗收 AC-02a
    n_clusters = acc["cluster"].nunique() - 1  # 去掉 -1
    logger.info(f"[驗收 AC-02a] 熱點數：{n_clusters}（預期 145）")
    if n_clusters != 145:
        logger.warning(f"  ！熱點數與預期不符")

    in_hot_ratio = (acc["cluster"] >= 0).mean() * 100
    logger.info(f"[驗收 AC-02a] 熱點內占比：{in_hot_ratio:.1f}%（預期 19.2%±0.1%）")

    # ── 儲存更新後的 accidents（加入 cluster 欄）
    start("儲存 accidents.parquet（含 cluster 欄）")
    acc.to_parquet(PROC_DIR / "accidents.parquet", index=False)
    logger.info(f"  輸出：{PROC_DIR / 'accidents.parquet'}")
    stop()

    # ── 熱點摘要
    start("建立熱點摘要")
    hotspots = build_hotspot_summary(acc)
    stop()

    # ── 加入設備距離
    start("計算照相設備距離")
    hotspots = add_camera_distances(hotspots, cam)
    stop()

    # cluster_type 留給 s05 填入，先設空字串
    hotspots["cluster_type"] = ""

    # ── 欄位排序輸出
    base_cols = ["cluster", "件數", "A1", "latitude", "longitude", "地點",
                 "dist_any", "dist_speed", "cluster_type"]
    trend_cols = sorted([c for c in hotspots.columns if c.startswith("件數_")])
    hotspots = hotspots[base_cols + trend_cols]

    out_path = PROC_DIR / "hotspots.csv"
    hotspots.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info(f"輸出：{out_path}")

    # ── 快速預覽前 5 名（依件數排序）
    top5 = hotspots.sort_values("件數", ascending=False).head(5)
    logger.info("Top 5 熱點（依件數）：")
    for _, row in top5.iterrows():
        logger.info(
            f"  Cluster {int(row['cluster']):>3d} | "
            f"{int(row['件數']):>4d} 件 | A1={int(row['A1']):>2d} | "
            f"dist_any={row['dist_any']:.0f}m | {row['地點']}"
        )

    logger.info("=== s02_hotspot 完成 ===")


if __name__ == "__main__":
    main()
