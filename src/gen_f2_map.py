"""
gen_f2_map.py — 產生含街道地圖底圖的 F2.png（kepler.gl 風格）

使用 contextily 下載 CartoDB Dark Matter 磚作為背景，
pyproj 將 WGS84 經緯度轉換為 Web Mercator（EPSG:3857）。
"""

import sys
from pathlib import Path

import contextily as ctx
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from pyproj import Transformer

sys.path.insert(0, str(Path(__file__).parent))
from config import PROC_DIR, OUT_DIR, FIG_DIR

# ── 字型
plt.rcParams["font.family"]        = ["Microsoft JhengHei", "Microsoft YaHei", "SimHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

# ── WGS84 → Web Mercator 轉換器
TO_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

TYPE_COLORS = {
    "全日型":    "#4FC3F7",   # 淡藍
    "通勤尖峰型": "#FFB74D",  # 橙
    "假日型":    "#81C784",   # 綠
    "夜間型":    "#F48FB1",   # 粉紅
}

def to_mercator(lon, lat):
    """numpy array 的批次轉換。"""
    x, y = TO_3857.transform(lon, lat)
    return np.array(x), np.array(y)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # ── 載入資料
    hot = pd.read_csv(PROC_DIR / "hotspots.csv",       encoding="utf-8-sig")
    cam = pd.read_parquet(PROC_DIR / "cameras.parquet")
    rec = pd.read_csv(OUT_DIR  / "recommendations.csv", encoding="utf-8-sig")

    top10_clusters = set(rec["cluster"].astype(int).tolist())

    # ── 座標轉換（WGS84 → Web Mercator）
    hot_x, hot_y = to_mercator(hot["longitude"].values, hot["latitude"].values)
    cam_x, cam_y = to_mercator(cam["longitude"].values, cam["latitude"].values)

    # ── 台北市地圖範圍（Web Mercator，稍留邊距）
    x_min, y_min = TO_3857.transform(121.44, 24.95)
    x_max, y_max = TO_3857.transform(121.68, 25.21)

    # ──────────────────────────────────────────
    # 繪圖
    # ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 11))
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")

    # 1. 街道地圖底圖（CartoDB Dark Matter，kepler.gl 風格）
    ctx.add_basemap(
        ax,
        source=ctx.providers.CartoDB.DarkMatter,
        zoom=13,
        attribution=False,
    )

    # 2. 熱點圓圈（發光效果：先畫大白暈，再畫彩色實點）
    for idx, row in hot.iterrows():
        color = TYPE_COLORS.get(row["cluster_type"], "#FFFFFF")
        is_top = int(row["cluster"]) in top10_clusters
        x, y   = hot_x[idx], hot_y[idx]
        size   = row["件數"] / 2.5 + 15

        # 外暈（glow effect）
        glow_size  = size * 4 if is_top else size * 2.5
        glow_color = "#FF5252" if is_top else color
        ax.scatter(x, y, s=glow_size, c=glow_color, alpha=0.18, zorder=3, linewidths=0)
        ax.scatter(x, y, s=glow_size * 0.5, c=glow_color, alpha=0.25, zorder=3, linewidths=0)

        # 實心點
        ax.scatter(x, y, s=size,
                   c="#FF5252" if is_top else color,
                   alpha=0.92 if is_top else 0.78,
                   edgecolors="white" if is_top else "none",
                   linewidths=1.5 if is_top else 0,
                   zorder=5)

    # 3. Top10 排名標注
    for _, row in rec.iterrows():
        h_row = hot[hot["cluster"] == row["cluster"]]
        if h_row.empty:
            continue
        h = h_row.iloc[0]
        x, y = to_mercator([h["longitude"]], [h["latitude"]])
        ax.annotate(
            f"★#{int(row['rank'])}",
            (x[0], y[0]),
            fontsize=8.5, fontweight="bold",
            color="white",
            ha="center", va="bottom",
            xytext=(0, 9), textcoords="offset points",
            zorder=7,
        )

    # 4. 照相設備（三角形標記）
    cam_speed = cam[cam["is_speed"]]
    cam_other = cam[~cam["is_speed"]]
    cs_x, cs_y = to_mercator(cam_speed["longitude"].values, cam_speed["latitude"].values)
    co_x, co_y = to_mercator(cam_other["longitude"].values, cam_other["latitude"].values)

    ax.scatter(cs_x, cs_y, marker="^", s=55, c="#EF5350",
               alpha=0.95, edgecolors="#FFCDD2", linewidths=0.8, zorder=6, label="測速照相")
    ax.scatter(co_x, co_y, marker="^", s=55, c="#42A5F5",
               alpha=0.95, edgecolors="#BBDEFB", linewidths=0.8, zorder=6, label="闖紅燈照相")

    # ── 圖例
    type_patches = [
        mpatches.Patch(color=c, label=t, alpha=0.9)
        for t, c in TYPE_COLORS.items()
    ]
    top10_patch = mpatches.Patch(color="#FF5252", label="覆蓋缺口 Top10（★）", alpha=0.9)
    cam_leg = [
        Line2D([0],[0], marker="^", color="w", markerfacecolor="#EF5350",
               markersize=9, label="測速照相"),
        Line2D([0],[0], marker="^", color="w", markerfacecolor="#42A5F5",
               markersize=9, label="闖紅燈照相"),
    ]
    legend = ax.legend(
        handles=type_patches + [top10_patch] + cam_leg,
        loc="lower left", fontsize=10,
        facecolor="#1A1A2E", edgecolor="#444466",
        labelcolor="white", title="圖例",
        title_fontsize=10,
    )
    legend.get_title().set_color("white")

    # ── 標題與說明
    ax.set_title(
        "臺北市交通事故熱點與固定式照相設備覆蓋分析（2021–2025）\n"
        "圓圈大小 ∝ 五年事故件數，★ = 覆蓋缺口前10名建議優先設置",
        fontsize=13, color="white", pad=14,
        fontweight="bold",
    )
    ax.tick_params(colors="gray", labelsize=8)
    ax.set_xlabel("經度（Web Mercator）", fontsize=9, color="gray")
    ax.set_ylabel("緯度（Web Mercator）", fontsize=9, color="gray")
    fig.patch.set_facecolor("#0D1117")
    ax.set_facecolor("#0D1117")

    # ── 資料來源浮水印
    ax.text(0.99, 0.01,
            "底圖：© CartoDB  資料：臺北市政府開放資料",
            transform=ax.transAxes, fontsize=7,
            color="#888888", ha="right", va="bottom")

    # ── 儲存
    out = FIG_DIR / "F2.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"輸出完成：{out}  ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
