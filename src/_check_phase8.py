from pathlib import Path
import matplotlib.font_manager as fm

files = {
    "accidents.parquet": "data/processed/accidents.parquet",
    "hotspots.csv":      "data/processed/hotspots.csv",
    "cameras.parquet":   "data/processed/cameras.parquet",
    "daily_series.csv":  "data/processed/daily_series.csv",
    "pca_coords.csv":    "data/processed/pca_coords.csv",
    "cluster_profiles":  "data/processed/cluster_profiles.csv",
    "pred.csv":          "outputs/pred.csv",
    "metrics.csv":       "outputs/metrics.csv",
    "recommendations":   "outputs/recommendations.csv",
}
for name, path in files.items():
    status = "OK" if Path(path).exists() else "MISSING"
    print(f"{name}: {status}")

print()
fonts = {f.name for f in fm.fontManager.ttflist}
for fn in ["Microsoft JhengHei", "Microsoft YaHei", "SimHei", "Arial Unicode MS", "Noto Sans CJK TC"]:
    print(f"font '{fn}': {fn in fonts}")
