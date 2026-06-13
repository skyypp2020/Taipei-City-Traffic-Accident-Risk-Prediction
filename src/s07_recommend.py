"""
s07_recommend.py — 覆蓋缺口分析與建議清單（FR-08）

流程：
  1. 載入 hotspots.csv（已含 dist_any / dist_speed / cluster_type）
  2. 篩選覆蓋缺口：dist_any > GAP_RADIUS_M（300m）
  3. 計算風險評分：
       risk_score = 件數（五年總計，第一階段）
       pred_2025  = 熱點月 XGBoost 預測 2025 年總件數（供參考）
  4. 主排序：risk_score 降序；次排序：A1 降序
  5. 依 cluster_type 對映建議設備類型
  6. 輸出前 TOP_N 名建議清單

輸出：
  outputs/recommendations.csv      前 10 名建議清單（AC-08）
  outputs/gap_analysis.csv         全部覆蓋缺口清單（87 個）
  outputs/logs/s07_recommend.log
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from config import PROC_DIR, OUT_DIR, GAP_RADIUS_M, TOP_N
from utils import get_logger, timer

logger = get_logger("s07_recommend")
start, stop = timer(logger)

# 型態 → 建議設備類型對映（SDD 4.5）
TYPE_MAPPING = {
    "夜間型":    "測速照相（深夜超速情境）",
    "通勤尖峰型":"闖紅燈照相 / 路口科技執法",
    "全日型":    "綜合評估（以 A1 數加權）",
    "假日型":    "行人安全導向科技執法",
}


# ─────────────────────────────────────────────
# 1. 計算 2025 預測件數
# ─────────────────────────────────────────────

def load_pred_2025(pred_path: Path) -> pd.Series:
    """
    讀取 hotspot_monthly_pred.csv（145×12），加總 12 個月得到各熱點 2025 預測年總件數。
    """
    if not pred_path.exists():
        logger.warning(f"找不到熱點月預測檔：{pred_path}，pred_2025 設為 NaN")
        return pd.Series(dtype=float, name="pred_2025")

    pred = pd.read_csv(pred_path, index_col="cluster", encoding="utf-8-sig")
    pred_2025 = pred.sum(axis=1).rename("pred_2025").round(1)
    logger.info(f"  pred_2025 載入：{len(pred_2025)} 個熱點，平均 {pred_2025.mean():.1f} 件/年")
    return pred_2025


# ─────────────────────────────────────────────
# 2. 覆蓋缺口篩選
# ─────────────────────────────────────────────

def find_gaps(hot: pd.DataFrame, gap_radius_m: float) -> pd.DataFrame:
    """篩選 dist_any > gap_radius_m 的熱點為覆蓋缺口。"""
    gap = hot[hot["dist_any"] > gap_radius_m].copy()
    covered = len(hot) - len(gap)
    logger.info(f"覆蓋分析（GAP_RADIUS={gap_radius_m}m）：")
    logger.info(f"  已覆蓋（dist_any ≤ {gap_radius_m}m）：{covered} 個熱點")
    logger.info(f"  覆蓋缺口（dist_any > {gap_radius_m}m）：{len(gap)} 個熱點")
    return gap


# ─────────────────────────────────────────────
# 3. 風險評分與排序
# ─────────────────────────────────────────────

def compute_risk_score(gap: pd.DataFrame, pred_2025: pd.Series) -> pd.DataFrame:
    """
    風險評分（第一階段）：risk_score = 件數（五年總計）
    同時附上 pred_2025 供參考（不影響排序）。
    排序規則：risk_score 降序 → A1 降序（次要）
    """
    gap = gap.copy()

    # 加入 pred_2025
    gap = gap.join(pred_2025, on="cluster", how="left")

    # 第一階段：以五年事故數為風險評分
    gap["risk_score"] = gap["件數"].astype(float)

    # 排序
    gap = gap.sort_values(
        ["risk_score", "A1"],
        ascending=[False, False],
    ).reset_index(drop=True)

    gap["rank"] = range(1, len(gap) + 1)
    return gap


# ─────────────────────────────────────────────
# 4. 建議設備類型
# ─────────────────────────────────────────────

def assign_device_type(gap: pd.DataFrame) -> pd.DataFrame:
    gap = gap.copy()
    gap["建議設備類型"] = gap["cluster_type"].map(TYPE_MAPPING).fillna("待評估")
    unknown = gap["建議設備類型"].eq("待評估").sum()
    if unknown:
        logger.warning(f"  {unknown} 個熱點的 cluster_type 無對映，設為「待評估」")
    return gap


# ─────────────────────────────────────────────
# 5. 整理輸出欄位
# ─────────────────────────────────────────────

def format_output(gap: pd.DataFrame, top_n: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    整理欄位順序，產出：
      (1) recommendations.csv — 前 TOP_N 名
      (2) gap_analysis.csv    — 全部缺口清單
    """
    trend_cols = [c for c in gap.columns if c.startswith("件數_")]

    output_cols = (
        ["rank", "地點", "latitude", "longitude",
         "cluster", "件數", "A1",
         "dist_any", "dist_speed"]
        + trend_cols
        + ["pred_2025", "cluster_type", "建議設備類型", "risk_score"]
    )
    # 確保欄位存在
    output_cols = [c for c in output_cols if c in gap.columns]

    gap_out = gap[output_cols].copy()
    top_n_df = gap_out[gap_out["rank"] <= top_n].copy()

    return top_n_df, gap_out


# ─────────────────────────────────────────────
# 主程式
# ─────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 載入資料
    start("載入 hotspots.csv")
    hot = pd.read_csv(PROC_DIR / "hotspots.csv", encoding="utf-8-sig")
    logger.info(f"  熱點：{len(hot)} 個，欄位：{hot.columns.tolist()}")
    stop()

    start("載入 hotspot_monthly_pred.csv（2025 預測）")
    pred_2025 = load_pred_2025(OUT_DIR / "hotspot_monthly_pred.csv")
    stop()

    # ── 覆蓋缺口
    start("覆蓋缺口篩選")
    gap = find_gaps(hot, GAP_RADIUS_M)
    stop()

    # ── 風險評分
    start("計算風險評分與排序")
    gap = compute_risk_score(gap, pred_2025)
    stop()

    # ── 建議設備類型
    start("對映建議設備類型")
    gap = assign_device_type(gap)
    stop()

    # ── 格式化輸出
    top_n_df, gap_all_df = format_output(gap, TOP_N)

    # 輸出 recommendations.csv
    out_rec = OUT_DIR / "recommendations.csv"
    top_n_df.to_csv(out_rec, index=False, encoding="utf-8-sig")
    logger.info(f"輸出：{out_rec}（前 {TOP_N} 名）")

    # 輸出 gap_analysis.csv
    out_gap = OUT_DIR / "gap_analysis.csv"
    gap_all_df.to_csv(out_gap, index=False, encoding="utf-8-sig")
    logger.info(f"輸出：{out_gap}（全部 {len(gap_all_df)} 個缺口）")

    # ── 驗收 AC-08
    top1 = top_n_df.iloc[0]
    logger.info("=== 驗收 AC-08 ===")
    logger.info(f"  第 1 名地點  ：{top1['地點']}")
    logger.info(f"  件數         ：{int(top1['件數'])}（預期 539）")
    logger.info(f"  dist_any     ：{top1['dist_any']:.1f}m（預期 915m）")
    logger.info(f"  cluster_type ：{top1['cluster_type']}")
    logger.info(f"  建議設備類型 ：{top1['建議設備類型']}")

    ac08_pass = (int(top1["件數"]) == 539 and abs(top1["dist_any"] - 915) < 2)
    logger.info(f"  AC-08 驗收：{'✓ 通過' if ac08_pass else '！未通過'}")

    # ── 完整 Top 10 列印
    logger.info("=== 前 10 名建議清單 ===")
    for _, row in top_n_df.iterrows():
        pred = f"pred_2025={row['pred_2025']:.0f}件" if pd.notna(row.get("pred_2025")) else ""
        logger.info(
            f"  #{int(row['rank']):>2d} | {row['地點'][:20]:<20} | "
            f"件數={int(row['件數']):>3d} A1={int(row['A1'])} | "
            f"dist_any={row['dist_any']:.0f}m | {row['cluster_type']} | "
            f"{row['建議設備類型']}"
            + (f" | {pred}" if pred else "")
        )

    logger.info("=== s07_recommend 完成 ===")


if __name__ == "__main__":
    main()
