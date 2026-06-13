"""
s05_pca_kmeans.py — PCA 降維與 K-means 熱點型態分群（FR-05、FR-06）

流程：
  dist_vectors(145×43)
    → StandardScaler
    → PCA(n_components=10) — scree 選 2-3 維
    → KMeans(k=2..8, random_state=42, n_init=10) — elbow + silhouette 選 k
    → 各群平均 24hr 曲線 → 自動命名型態
    → 回填 hotspots.csv 的 cluster_type 欄位

輸出：
  data/processed/pca_coords.csv           主成分座標 + 群標籤 + 地理座標
  outputs/explained_variance.csv          各主成分解釋變異
  outputs/loadings.csv                    PCA 負荷量矩陣
  data/processed/hotspot_clusters.csv     熱點→群標籤對映表
  data/processed/cluster_profiles.csv     各群平均時間分布曲線
  data/processed/hotspots.csv             更新：回填 cluster_type
  outputs/logs/s05_pca_kmeans.log
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))

from config import PROC_DIR, OUT_DIR, RANDOM_STATE
from utils import get_logger, timer

logger = get_logger("s05_pca_kmeans")
start, stop = timer(logger)

# 小時欄、星期欄、月份欄
HOUR_COLS    = [f"hour_{h}"    for h in range(24)]
WEEKDAY_COLS = [f"weekday_{d}" for d in range(7)]
MONTH_COLS   = [f"month_{m}"   for m in range(1, 13)]


# ─────────────────────────────────────────────
# 1. PCA 降維（FR-05）
# ─────────────────────────────────────────────

def run_pca(vecs: pd.DataFrame, n_components: int = 10):
    """
    StandardScaler → PCA。
    回傳 (coords_df, explained_var_df, loadings_df, scaler, pca)。
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(vecs.values)

    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)

    # 累積解釋變異
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    var_df = pd.DataFrame({
        "component":           range(1, n_components + 1),
        "explained_variance":  pca.explained_variance_ratio_.round(4),
        "cumulative":          cum_var.round(4),
    })

    logger.info("PCA 解釋變異（前 10 成分）：")
    for _, row in var_df.iterrows():
        logger.info(f"  PC{int(row['component']):2d}: {row['explained_variance']:.3f}  累積: {row['cumulative']:.3f}")

    pc3_cum = var_df.loc[var_df["component"] == 3, "cumulative"].values[0]
    if pc3_cum < 0.60:
        logger.warning(f"  ！前 3 主成分累積解釋變異 {pc3_cum:.1%} < 60%，報告中需討論")
    else:
        logger.info(f"  前 3 主成分累積解釋變異 {pc3_cum:.1%} ✓")

    # 負荷量矩陣（feature × component）
    loadings_df = pd.DataFrame(
        pca.components_.T,
        index=vecs.columns,
        columns=[f"PC{i+1}" for i in range(n_components)],
    ).round(4)

    # 主成分座標（前 3 維供視覺化）
    coords_df = pd.DataFrame(
        X_pca[:, :3],
        index=vecs.index,
        columns=["PC1", "PC2", "PC3"],
    )

    return coords_df, var_df, loadings_df, scaler, pca


# ─────────────────────────────────────────────
# 2. 選最佳 k（elbow + silhouette）（FR-06）
# ─────────────────────────────────────────────

def select_k(X_pca: np.ndarray, k_range=range(2, 9)) -> int:
    """
    對前 3 個主成分執行 k=2..8 的 KMeans，
    以 silhouette 最大值選最佳 k（elbow 輔助判斷）。
    回傳最佳 k 與完整指標 DataFrame。
    """
    results = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X_pca)
        inertia = km.inertia_
        sil = silhouette_score(X_pca, labels)
        results.append({"k": k, "inertia": round(inertia, 2), "silhouette": round(sil, 4)})
        logger.info(f"  k={k}: inertia={inertia:.2f}, silhouette={sil:.4f}")

    metrics_df = pd.DataFrame(results)

    # 以 silhouette 最大為主要選擇依據
    best_k = int(metrics_df.loc[metrics_df["silhouette"].idxmax(), "k"])
    best_sil = metrics_df.loc[metrics_df["silhouette"].idxmax(), "silhouette"]
    logger.info(f"最佳 k = {best_k}（silhouette = {best_sil:.4f}）")

    return best_k, metrics_df


# ─────────────────────────────────────────────
# 3. 命名型態（跨群相對閾值，避免全部被歸為同一類）
# ─────────────────────────────────────────────

def compute_cluster_stats(cluster_profiles: pd.DataFrame) -> pd.DataFrame:
    """
    計算各群在小時/星期正規化後的關鍵指標：
      night   — 深夜（22:00–03:59）占所有小時比例
      am_peak — 早峰（07:00–09:59）占所有小時比例
      pm_peak — 晚峰（17:00–19:59）占所有小時比例
      commute — am_peak + pm_peak
      weekend — 週六+週日 占所有星期比例
    """
    profiles_num = cluster_profiles[HOUR_COLS + WEEKDAY_COLS].astype(float)
    stats = pd.DataFrame(index=profiles_num.index)

    for km_label, row in profiles_num.iterrows():
        h = row[HOUR_COLS]
        h_norm = h / h.sum() if h.sum() > 0 else h
        w = row[WEEKDAY_COLS]
        w_norm = w / w.sum() if w.sum() > 0 else w

        stats.loc[km_label, "night"]   = float(h_norm[["hour_22","hour_23","hour_0","hour_1","hour_2","hour_3"]].sum())
        stats.loc[km_label, "am_peak"] = float(h_norm[["hour_7","hour_8","hour_9"]].sum())
        stats.loc[km_label, "pm_peak"] = float(h_norm[["hour_17","hour_18","hour_19"]].sum())
        stats.loc[km_label, "commute"] = stats.loc[km_label, "am_peak"] + stats.loc[km_label, "pm_peak"]
        stats.loc[km_label, "weekend"] = float(w_norm[["weekday_5","weekday_6"]].sum())

    return stats


def name_clusters(cluster_profiles: pd.DataFrame) -> dict:
    """
    以跨群統計（mean ± 0.5×std）動態設定閾值，依優先順序命名：
      夜間型     — night 超過 (mean + 0.5×std)
      通勤尖峰型 — commute 超過 (mean + 0.3×std)（須非夜間型）
      假日型     — weekend 超過 (mean + 0.3×std)（須非上述兩者）
      全日型     — 不符合任何條件（分布最為平坦）

    使用相對閾值而非固定值，確保即使資料整體偏向某型態（如
    臺北市全面通勤高峰），仍能產生有意義的分群差異。
    """
    stats = compute_cluster_stats(cluster_profiles)

    # 動態閾值
    night_th   = stats["night"].mean()   + stats["night"].std()   * 0.5
    commute_th = stats["commute"].mean() + stats["commute"].std() * 0.3
    weekend_th = stats["weekend"].mean() + stats["weekend"].std() * 0.3

    logger.info(f"  命名閾值 — night≥{night_th:.3f}, commute≥{commute_th:.3f}, weekend≥{weekend_th:.3f}")

    type_map = {}
    for km_label, row in stats.iterrows():
        if row["night"] >= night_th:
            type_map[km_label] = "夜間型"
        elif row["commute"] >= commute_th:
            type_map[km_label] = "通勤尖峰型"
        elif row["weekend"] >= weekend_th:
            type_map[km_label] = "假日型"
        else:
            type_map[km_label] = "全日型"

    # 詳細 log
    for km_label, row in stats.iterrows():
        logger.info(
            f"  群 {km_label}: night={row['night']:.3f}, "
            f"commute={row['commute']:.3f}(am={row['am_peak']:.3f} pm={row['pm_peak']:.3f}), "
            f"weekend={row['weekend']:.3f} → {type_map[km_label]}"
        )

    return type_map


# ─────────────────────────────────────────────
# 4. 執行最終 KMeans 並產出所有結果
# ─────────────────────────────────────────────

def run_final_kmeans(vecs: pd.DataFrame, coords_df: pd.DataFrame,
                     best_k: int):
    """以最佳 k 執行 KMeans，產出群標籤、曲線剖面、型態命名。"""
    X_pca3 = coords_df[["PC1", "PC2", "PC3"]].values

    km = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(X_pca3)

    # 各熱點對映群標籤
    cluster_map = pd.Series(labels, index=vecs.index, name="kmeans_label")

    # 各群平均時間分布曲線（原始 43 維）
    profiles_raw = vecs.copy()
    profiles_raw["kmeans_label"] = labels
    cluster_profiles = profiles_raw.groupby("kmeans_label")[HOUR_COLS + WEEKDAY_COLS + MONTH_COLS].mean()

    # 跨群相對閾值命名
    type_map = name_clusters(cluster_profiles)

    cluster_profiles["cluster_type"] = cluster_profiles.index.map(type_map)

    return cluster_map, cluster_profiles, type_map


# ─────────────────────────────────────────────
# 主程式
# ─────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 載入資料
    start("載入 dist_vectors.csv 與 hotspots.csv")
    vecs = pd.read_csv(PROC_DIR / "dist_vectors.csv", index_col="cluster", encoding="utf-8-sig")
    hot  = pd.read_csv(PROC_DIR / "hotspots.csv", encoding="utf-8-sig")
    logger.info(f"  dist_vectors: {vecs.shape} / hotspots: {hot.shape}")
    stop()

    # ── PCA（FR-05）
    start("PCA 降維（StandardScaler → PCA(n=10)）")
    coords_df, var_df, loadings_df, scaler, pca = run_pca(vecs, n_components=10)
    stop()

    # 輸出解釋變異與負荷量
    var_df.to_csv(OUT_DIR / "explained_variance.csv", index=False, encoding="utf-8-sig")
    loadings_df.to_csv(OUT_DIR / "loadings.csv", encoding="utf-8-sig")
    logger.info(f"輸出：explained_variance.csv / loadings.csv")

    # ── 選最佳 k（FR-06）
    start("K-means k 選擇（elbow + silhouette, k=2..8）")
    X_pca3 = coords_df[["PC1", "PC2", "PC3"]].values
    best_k, metrics_df = select_k(X_pca3)
    stop()

    # 輸出 silhouette 指標表（AC-06）
    metrics_df.to_csv(OUT_DIR / "kmeans_metrics.csv", index=False, encoding="utf-8-sig")
    logger.info(f"輸出：outputs/kmeans_metrics.csv")

    # ── 最終 KMeans（FR-06）
    start(f"最終 KMeans（k={best_k}）")
    cluster_map, cluster_profiles, type_map = run_final_kmeans(vecs, coords_df, best_k)
    stop()

    # 驗收 AC-06：每個熱點皆有群標籤
    assert len(cluster_map) == 145, f"熱點數異常：{len(cluster_map)}"
    assert cluster_map.notna().all(), "有熱點缺少群標籤"
    logger.info(f"[驗收 AC-06] 全部 {len(cluster_map)} 個熱點皆有群標籤 ✓")

    # 型態分布
    type_counts = cluster_map.map(type_map).value_counts()
    logger.info("型態分布：")
    for t, cnt in type_counts.items():
        logger.info(f"  {t}：{cnt} 個熱點")

    # ── 合併地理座標至 pca_coords（kepler.gl 相容）
    hot_geo = hot[["cluster", "latitude", "longitude", "地點", "件數"]].set_index("cluster")
    coords_out = coords_df.copy()
    coords_out["kmeans_label"] = cluster_map
    coords_out["cluster_type"] = cluster_map.map(type_map)
    coords_out = coords_out.join(hot_geo)
    coords_out.to_csv(PROC_DIR / "pca_coords.csv", encoding="utf-8-sig")
    logger.info(f"輸出：pca_coords.csv（{coords_out.shape}）")

    # ── 熱點→群對映表
    hc = cluster_map.reset_index()
    hc.columns = ["cluster", "kmeans_label"]
    hc["cluster_type"] = hc["kmeans_label"].map(type_map)
    hc = hc.merge(hot_geo.reset_index(), on="cluster", how="left")
    hc.to_csv(PROC_DIR / "hotspot_clusters.csv", index=False, encoding="utf-8-sig")
    logger.info(f"輸出：hotspot_clusters.csv（{hc.shape}）")

    # ── 各群平均時間分布曲線
    cluster_profiles.to_csv(PROC_DIR / "cluster_profiles.csv", encoding="utf-8-sig")
    logger.info(f"輸出：cluster_profiles.csv（{cluster_profiles.shape}）")

    # ── 回填 hotspots.csv 的 cluster_type（原本全部為空）
    hot["cluster_type"] = hot["cluster"].map(
        hc.set_index("cluster")["cluster_type"]
    )
    hot.to_csv(PROC_DIR / "hotspots.csv", index=False, encoding="utf-8-sig")
    null_after = hot["cluster_type"].isna().sum()
    logger.info(f"hotspots.csv cluster_type 回填完成，剩餘空值：{null_after}（預期 0）")

    # ── 摘要
    logger.info("=== 驗收摘要 ===")
    logger.info(f"  AC-05 累積解釋變異（前3PC）：{var_df.loc[2,'cumulative']:.1%}")
    logger.info(f"  AC-06 最佳 k = {best_k}，全 145 熱點皆有標籤 ✓")
    logger.info(f"  型態種類：{sorted(type_map.values())}")
    logger.info("=== s05_pca_kmeans 完成 ===")


if __name__ == "__main__":
    main()
