"""
s08_visualize.py — 視覺化圖表產出（FR-09）

圖表：
  F1  事故空間密度 + 熱點分布（hexbin + scatter）         PNG
  F2  熱點 vs 設備疊圖（folium 互動地圖）                HTML + PNG 靜態版
  F3  各年 A1/A2 事故件數與比例                          PNG
  F4  全市日事故數時間序列（COVID 灰底標注）              PNG
  F5  熱點時間指紋 PCA 散佈 + 各群 24hr 曲線             PNG
  F6  2025 驗證期事故數預測比較                          PNG

AC-09：PNG ≥ 150 dpi，含標題與軸標籤
中文字型：Microsoft JhengHei（Windows 系統字型）

輸出：figures/F1.png ~ F6.png、figures/F2.html
"""

import sys
from pathlib import Path

import folium
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from folium.plugins import HeatMap
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    PROC_DIR, OUT_DIR, FIG_DIR,
    COVID_LV3_START, COVID_LV3_END,
    TRAIN_END, VALID_START,
)
from utils import get_logger, timer

logger = get_logger("s08_visualize")
start, stop = timer(logger)

# ── 中文字型設定
plt.rcParams["font.family"]     = ["Microsoft JhengHei", "Microsoft YaHei", "SimHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
DPI = 150

# ── 型態顏色對映
TYPE_COLORS = {
    "全日型":    "#4C72B0",
    "通勤尖峰型": "#DD8452",
    "假日型":    "#55A868",
    "夜間型":    "#C44E52",
}
HOUR_COLS    = [f"hour_{h}"    for h in range(24)]
WEEKDAY_COLS = [f"weekday_{d}" for d in range(7)]
MONTH_COLS   = [f"month_{m}"   for m in range(1, 13)]


def save_fig(fig, name: str):
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"  輸出：{path}")


# ─────────────────────────────────────────────
# F1：事故空間密度 + 熱點分布
# ─────────────────────────────────────────────

def plot_f1(acc: pd.DataFrame, hot: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(11, 10))

    # 底層：hexbin 密度
    hb = ax.hexbin(
        acc["longitude"], acc["latitude"],
        gridsize=60, cmap="YlOrRd", bins="log",
        linewidths=0.1, alpha=0.85,
    )
    fig.colorbar(hb, ax=ax, label="事故件數（對數）", shrink=0.7)

    # 上層：熱點中心（大小 ∝ 件數）
    for _, row in hot.iterrows():
        color = TYPE_COLORS.get(row["cluster_type"], "#888888")
        ax.scatter(row["longitude"], row["latitude"],
                   s=row["件數"] / 3, c=color, alpha=0.7,
                   edgecolors="white", linewidths=0.5, zorder=5)

    # 圖例
    legend_handles = [
        mpatches.Patch(color=c, label=t) for t, c in TYPE_COLORS.items()
    ]
    ax.legend(handles=legend_handles, title="熱點型態", loc="upper left",
              framealpha=0.9, fontsize=10)

    ax.set_title("臺北市交通事故空間密度與熱點分布（2021–2025）", fontsize=14, pad=12)
    ax.set_xlabel("經度（Longitude）", fontsize=11)
    ax.set_ylabel("緯度（Latitude）", fontsize=11)
    ax.set_xlim(121.45, 121.67)
    ax.set_ylim(24.96, 25.20)
    fig.tight_layout()
    save_fig(fig, "F1")


# ─────────────────────────────────────────────
# F2：熱點 vs 設備疊圖（folium 互動地圖）+ 靜態截圖
# ─────────────────────────────────────────────

def plot_f2(hot: pd.DataFrame, cam: pd.DataFrame, rec: pd.DataFrame):
    center = [25.05, 121.53]
    m = folium.Map(location=center, zoom_start=12, tiles="CartoDB positron")

    top10_clusters = set(rec["cluster"].astype(int).tolist())

    # 熱點圓圈（半徑 ∝ 件數，缺口 Top10 高亮）
    for _, row in hot.iterrows():
        is_top10 = int(row["cluster"]) in top10_clusters
        color    = "#E74C3C" if is_top10 else TYPE_COLORS.get(row["cluster_type"], "#4C72B0")
        weight   = 3 if is_top10 else 1

        popup_text = (
            f"<b>{'★ Top10 缺口 ' if is_top10 else ''}Cluster {int(row['cluster'])}</b><br>"
            f"地點：{row['地點']}<br>"
            f"件數：{int(row['件數'])} | A1：{int(row['A1'])}<br>"
            f"型態：{row['cluster_type']}<br>"
            f"dist_any：{row['dist_any']:.0f}m"
        )
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=max(5, row["件數"] / 30),
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.5,
            weight=weight,
            popup=folium.Popup(popup_text, max_width=250),
            tooltip=f"{'★ ' if is_top10 else ''}{row['地點'][:15]} ({int(row['件數'])}件)",
        ).add_to(m)

    # 照相設備標記
    for _, row in cam.iterrows():
        icon_color = "red" if row["is_speed"] else "blue"
        icon_name  = "camera" if row["is_speed"] else "eye"
        label = "測速照相" if row["is_speed"] else "闖紅燈照相"
        folium.Marker(
            location=[row["latitude"], row["longitude"]],
            popup=folium.Popup(
                f"<b>{label}</b><br>{row.get('設置路段','')}<br>{row.get('速限','?')} km/h",
                max_width=200,
            ),
            tooltip=label,
            icon=folium.Icon(color=icon_color, icon=icon_name, prefix="fa"),
        ).add_to(m)

    # 圖例
    legend_html = """
    <div style='position:fixed;bottom:30px;left:30px;background:white;padding:10px;
                border:1px solid grey;border-radius:5px;font-size:13px;z-index:9999'>
    <b>圖例</b><br>
    <span style='color:#E74C3C'>●</span> 覆蓋缺口 Top10（優先建議）<br>
    <span style='color:#4C72B0'>●</span> 全日型熱點<br>
    <span style='color:#DD8452'>●</span> 通勤尖峰型熱點<br>
    <span style='color:#55A868'>●</span> 假日型熱點<br>
    <span style='color:#C44E52'>●</span> 夜間型熱點<br>
    <span style='color:red'>📷</span> 測速照相設備<br>
    <span style='color:blue'>📷</span> 闖紅燈照相設備
    </div>"""
    m.get_root().html.add_child(folium.Element(legend_html))

    html_path = FIG_DIR / "F2.html"
    m.save(str(html_path))
    logger.info(f"  輸出（HTML）：{html_path}")

    # 靜態版本（matplotlib）
    fig, ax = plt.subplots(figsize=(11, 10))
    colors_hot = [TYPE_COLORS.get(t, "#888888") for t in hot["cluster_type"]]
    ax.scatter(hot["longitude"], hot["latitude"],
               s=hot["件數"] / 3, c=colors_hot, alpha=0.6,
               edgecolors="white", linewidths=0.5, label="_nolegend_")

    # Top10 缺口高亮
    top10_df = hot[hot["cluster"].isin(top10_clusters)]
    ax.scatter(top10_df["longitude"], top10_df["latitude"],
               s=top10_df["件數"] / 2, c="#E74C3C", alpha=0.9,
               edgecolors="black", linewidths=1.5, zorder=6)
    for _, row in top10_df.head(10).iterrows():
        rank = rec.loc[rec["cluster"] == row["cluster"], "rank"].values
        label_txt = f"#{int(rank[0])}" if len(rank) > 0 else ""
        ax.annotate(label_txt, (row["longitude"], row["latitude"]),
                    fontsize=8, ha="center", va="bottom", color="black",
                    xytext=(0, 6), textcoords="offset points")

    # 設備標記
    cam_speed = cam[cam["is_speed"]]
    cam_other = cam[~cam["is_speed"]]
    ax.scatter(cam_speed["longitude"], cam_speed["latitude"],
               marker="^", s=40, c="#E74C3C", alpha=0.9, label="測速照相", zorder=7)
    ax.scatter(cam_other["longitude"], cam_other["latitude"],
               marker="^", s=40, c="#3498DB", alpha=0.9, label="闖紅燈照相", zorder=7)

    legend_handles = (
        [mpatches.Patch(color=c, label=t) for t, c in TYPE_COLORS.items()] +
        [mpatches.Patch(color="#E74C3C", label="覆蓋缺口 Top10（標★）")] +
        [Line2D([0],[0], marker="^", color="w", markerfacecolor="#E74C3C", markersize=8, label="測速照相"),
         Line2D([0],[0], marker="^", color="w", markerfacecolor="#3498DB", markersize=8, label="闖紅燈照相")]
    )
    ax.legend(handles=legend_handles, loc="upper left", fontsize=9, framealpha=0.9)
    ax.set_title("臺北市熱點與固定式照相設備分布（覆蓋缺口前10名標示）", fontsize=13, pad=12)
    ax.set_xlabel("經度", fontsize=11)
    ax.set_ylabel("緯度", fontsize=11)
    ax.set_xlim(121.45, 121.67)
    ax.set_ylim(24.96, 25.20)
    fig.tight_layout()
    save_fig(fig, "F2")


# ─────────────────────────────────────────────
# F3：各年 A1/A2 事故件數與比例
# ─────────────────────────────────────────────

def plot_f3(acc: pd.DataFrame):
    acc = acc.copy()
    acc["年份"] = acc["發生時間"].dt.year
    yearly = acc.groupby(["年份", "處理別"]).size().unstack(fill_value=0)
    yearly.columns = ["A1（死亡）", "A2（受傷）"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))

    # 左：件數堆疊長條
    x = np.arange(len(yearly))
    w = 0.6
    bars_a2 = ax1.bar(x, yearly["A2（受傷）"], width=w, color="#4C72B0", label="A2（受傷）")
    bars_a1 = ax1.bar(x, yearly["A1（死亡）"], width=w, bottom=yearly["A2（受傷）"],
                      color="#C44E52", label="A1（死亡）")
    ax1.set_xticks(x)
    ax1.set_xticklabels(yearly.index.astype(str))
    ax1.set_xlabel("年份", fontsize=11)
    ax1.set_ylabel("事故件數", fontsize=11)
    ax1.set_title("各年 A1/A2 事故件數", fontsize=13)
    ax1.legend(fontsize=10)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    for bar, val in zip(bars_a2, yearly["A2（受傷）"]):
        ax1.text(bar.get_x() + bar.get_width()/2, val/2, f"{int(val):,}",
                 ha="center", va="center", fontsize=9, color="white")

    # 右：A1 比例折線
    a1_pct = (yearly["A1（死亡）"] / yearly.sum(axis=1) * 100).round(2)
    ax2.plot(yearly.index, a1_pct, marker="o", color="#C44E52", linewidth=2.5, markersize=8)
    for yr, val in zip(yearly.index, a1_pct):
        ax2.annotate(f"{val:.2f}%", (yr, val), textcoords="offset points",
                     xytext=(0, 8), ha="center", fontsize=10)
    ax2.set_xlabel("年份", fontsize=11)
    ax2.set_ylabel("A1 占比（%）", fontsize=11)
    ax2.set_title("各年 A1（死亡）事故占比", fontsize=13)
    ax2.set_ylim(0, max(a1_pct) * 1.5)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}%"))

    fig.suptitle("臺北市交通事故嚴重程度分析（2021–2025）", fontsize=14, y=1.01)
    fig.tight_layout()
    save_fig(fig, "F3")


# ─────────────────────────────────────────────
# F4：全市日事故數時間序列（COVID 灰底標注）
# ─────────────────────────────────────────────

def plot_f4(daily: pd.Series):
    # 30 日移動平均（僅供視覺）
    ma30 = daily.rolling(30, center=True).mean()

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(daily.index, daily.values, color="#AAAAAA", linewidth=0.6, alpha=0.7, label="日事故數")
    ax.plot(ma30.index, ma30.values, color="#4C72B0", linewidth=2, label="30日移動平均")

    # COVID 灰底
    covid_s = pd.Timestamp(COVID_LV3_START)
    covid_e = pd.Timestamp(COVID_LV3_END)
    ax.axvspan(covid_s, covid_e, color="gray", alpha=0.3, label="COVID-19 三級警戒")
    ax.annotate("COVID-19\n三級警戒", xy=(covid_s + (covid_e - covid_s)/2, ax.get_ylim()[1]),
                xytext=(0, -5), textcoords="offset points",
                ha="center", va="top", fontsize=9, color="dimgray")

    # 訓練/驗證切分線
    split_date = pd.Timestamp(VALID_START)
    ax.axvline(split_date, color="#E74C3C", linewidth=1.5, linestyle="--", label="訓練/驗證切分（2025-01-01）")

    ax.set_title("臺北市全市日事故數時間序列（2021–2025）", fontsize=14, pad=10)
    ax.set_xlabel("日期", fontsize=11)
    ax.set_ylabel("事故件數", fontsize=11)
    ax.legend(fontsize=10, loc="upper right")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x)}"))
    fig.tight_layout()
    save_fig(fig, "F4")


# ─────────────────────────────────────────────
# F5：PCA 散佈 + 各群 24hr 平均曲線
# ─────────────────────────────────────────────

def plot_f5(pca_coords: pd.DataFrame, cluster_profiles: pd.DataFrame):
    cluster_types = cluster_profiles["cluster_type"].dropna().unique()

    fig = plt.figure(figsize=(16, 7))
    gs  = fig.add_gridspec(2, 2 + len(cluster_types), hspace=0.45, wspace=0.4)

    # 左：PCA 散佈圖（PC1 vs PC2）
    ax_pca = fig.add_subplot(gs[:, :2])
    for ctype, color in TYPE_COLORS.items():
        mask = pca_coords["cluster_type"] == ctype
        ax_pca.scatter(pca_coords.loc[mask, "PC1"], pca_coords.loc[mask, "PC2"],
                       s=pca_coords.loc[mask, "件數"] / 5 + 20,
                       c=color, alpha=0.75, label=ctype,
                       edgecolors="white", linewidths=0.5)
    ax_pca.set_xlabel("PC1", fontsize=11)
    ax_pca.set_ylabel("PC2", fontsize=11)
    ax_pca.set_title("熱點時間指紋 PCA 散佈（PC1 vs PC2）\n點大小 ∝ 五年事故件數", fontsize=12)
    ax_pca.legend(title="型態", fontsize=9)

    # 右：各型態平均 24hr 曲線
    hours = list(range(24))
    for col_idx, ctype in enumerate(sorted(cluster_types, key=lambda x: list(TYPE_COLORS.keys()).index(x) if x in TYPE_COLORS else 99)):
        color = TYPE_COLORS.get(ctype, "gray")
        subset = cluster_profiles[cluster_profiles["cluster_type"] == ctype][HOUR_COLS].astype(float)
        mean_h = subset.mean()
        h_norm = mean_h / mean_h.sum()   # 小時內正規化

        row_idx = 0 if col_idx < 2 else 1
        c_idx   = 2 + (col_idx % 2) if len(cluster_types) == 4 else 2 + col_idx
        if len(cluster_types) == 4:
            ax = fig.add_subplot(gs[col_idx // 2, 2 + col_idx % 2])
        else:
            ax = fig.add_subplot(gs[:, 2 + col_idx])

        ax.fill_between(hours, h_norm.values, color=color, alpha=0.35)
        ax.plot(hours, h_norm.values, color=color, linewidth=2)
        ax.set_title(ctype, fontsize=11, color=color)
        ax.set_xlabel("小時", fontsize=9)
        ax.set_ylabel("事故比例", fontsize=9)
        ax.set_xticks([0, 6, 12, 18, 23])
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x)}時"))

        # 尖峰時段標注
        peak_h = int(h_norm.idxmax().replace("hour_", ""))
        ax.axvline(peak_h, color=color, linewidth=1.2, linestyle="--", alpha=0.7)
        ax.annotate(f"{peak_h}時", xy=(peak_h, h_norm.max()),
                    fontsize=8, ha="left", va="top", color=color, xytext=(3, -3),
                    textcoords="offset points")

    fig.suptitle("臺北市熱點時間指紋 PCA 降維與型態分群（k=8 合併為 4 種型態）",
                 fontsize=13, y=1.02)
    save_fig(fig, "F5")


# ─────────────────────────────────────────────
# F6：2025 驗證期事故數預測比較
# ─────────────────────────────────────────────

def plot_f6(pred: pd.DataFrame, metrics: pd.DataFrame):
    model_styles = {
        "y_true": ("實際值",     "#222222", 2.0, "-"),
        "Naive":  ("Naive(lag-7)","#999999", 1.2, "--"),
        "SARIMA": ("SARIMA",     "#4C72B0", 1.8, "-."),
        "XGBoost":("XGBoost",   "#DD8452", 1.8, "-"),
    }

    fig, ax = plt.subplots(figsize=(16, 6))
    for col, (label, color, lw, ls) in model_styles.items():
        if col in pred.columns:
            ax.plot(pred.index, pred[col], label=label,
                    color=color, linewidth=lw, linestyle=ls, alpha=0.9)

    # 指標文字框
    metric_rows = metrics[metrics["model"].isin(["Naive(lag-7)", "SARIMA", "XGBoost"])]
    txt_lines = ["模型指標（驗證期 2025）"]
    for _, row in metric_rows.iterrows():
        name = row["model"].split("(")[0].replace("SARIMA(1,1,2)(1,0,1,7) AIC=11351.1", "SARIMA")
        improve = f"  改善{row['improve_vs_naive_pct']:.1f}%" if pd.notna(row.get("improve_vs_naive_pct")) else ""
        txt_lines.append(f"{name}：MAE={row['mae']:.2f}  RMSE={row['rmse']:.2f}{improve}")
    textstr = "\n".join(txt_lines)
    ax.text(0.01, 0.97, textstr, transform=ax.transAxes, fontsize=9.5,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.85, edgecolor="#BBBBBB"))

    ax.set_title("臺北市日事故數預測比較（2025年驗證期）", fontsize=14, pad=10)
    ax.set_xlabel("日期", fontsize=11)
    ax.set_ylabel("事故件數", fontsize=11)
    ax.legend(fontsize=10, loc="upper right")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x)}"))
    fig.tight_layout()
    save_fig(fig, "F6")


# ─────────────────────────────────────────────
# 主程式
# ─────────────────────────────────────────────

def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # ── 載入資料
    start("載入所有資料")
    acc      = pd.read_parquet(PROC_DIR / "accidents.parquet")
    hot      = pd.read_csv(PROC_DIR / "hotspots.csv",         encoding="utf-8-sig")
    cam      = pd.read_parquet(PROC_DIR / "cameras.parquet")
    daily    = pd.read_csv(PROC_DIR / "daily_series.csv",     index_col="date",
                           parse_dates=True, encoding="utf-8-sig")["count"]
    pca_c    = pd.read_csv(PROC_DIR / "pca_coords.csv",       index_col="cluster", encoding="utf-8-sig")
    profiles = pd.read_csv(PROC_DIR / "cluster_profiles.csv", index_col="kmeans_label", encoding="utf-8-sig")
    pred     = pd.read_csv(OUT_DIR / "pred.csv",              index_col="date",
                           parse_dates=True, encoding="utf-8-sig")
    metrics  = pd.read_csv(OUT_DIR / "metrics.csv",           encoding="utf-8-sig")
    rec      = pd.read_csv(OUT_DIR / "recommendations.csv",   encoding="utf-8-sig")
    logger.info(f"  資料載入完成：acc={len(acc):,}, hot={len(hot)}, cam={len(cam)}, pred={len(pred)}")
    stop()

    # ── F1
    start("F1：空間密度 + 熱點分布")
    plot_f1(acc, hot)
    stop()

    # ── F2
    start("F2：熱點 vs 設備疊圖（folium + 靜態）")
    plot_f2(hot, cam, rec)
    stop()

    # ── F3
    start("F3：A1/A2 比例")
    plot_f3(acc)
    stop()

    # ── F4
    start("F4：日序列（COVID 灰底）")
    plot_f4(daily)
    stop()

    # ── F5
    start("F5：PCA 散佈 + 群曲線")
    plot_f5(pca_c, profiles)
    stop()

    # ── F6
    start("F6：驗證期預測 vs 實際")
    plot_f6(pred, metrics)
    stop()

    # ── 驗收摘要
    figs = list(FIG_DIR.glob("*.png")) + list(FIG_DIR.glob("*.html"))
    logger.info("=== 驗收 AC-09 ===")
    for f in sorted(figs):
        size_kb = f.stat().st_size / 1024
        logger.info(f"  {f.name}：{size_kb:.0f} KB ✓")
    logger.info("=== s08_visualize 完成 ===")


if __name__ == "__main__":
    main()
