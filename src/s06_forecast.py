"""
s06_forecast.py — 預測建模與 80/20 驗證（FR-07）

模型：
  Naive(lag-7)         ← 下限 baseline，ŷ(t) = y(t-7)
  SARIMA(s=7)          ← 統計 baseline，訓練期 AIC 網格搜尋
  XGBoost              ← 主要挑戰者，全特徵
  全域熱點月 XGBoost   ← 熱點月預測，供 s07 排序使用

切分（嚴格時序，AC-07c）：
  訓練 2021-01-29 ~ 2024-12-31（feature_matrix 截斷後）
  驗證 2025-01-01 ~ 2025-12-31

輸出：
  outputs/metrics.csv    各模型 MAE / RMSE / 相對 Naive 改善率
  outputs/pred.csv       驗證期每日預測值（各模型）
  outputs/logs/s06_forecast.log
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    PROC_DIR, OUT_DIR,
    TRAIN_END, VALID_START,
    RANDOM_STATE,
)
from utils import get_logger, timer

logger = get_logger("s06_forecast")
start, stop = timer(logger)

FEAT_COLS = [
    "lag_1","lag_7","lag_14","lag_28",
    "ma_7","std_7","ma_28","std_28",
    "weekday_0","weekday_1","weekday_2","weekday_3",
    "weekday_4","weekday_5","weekday_6",
    "month","covid_lv3",
]


# ─────────────────────────────────────────────
# 工具：評估指標
# ─────────────────────────────────────────────

def evaluate(y_true: pd.Series, y_pred: np.ndarray,
             model_name: str, naive_mae: float = None) -> dict:
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    row  = {"model": model_name, "mae": round(mae, 3), "rmse": round(rmse, 3)}
    if naive_mae is not None:
        row["improve_vs_naive_pct"] = round((naive_mae - mae) / naive_mae * 100, 1)
    logger.info(f"  [{model_name}] MAE={mae:.3f}, RMSE={rmse:.3f}" +
                (f", 改善率={row['improve_vs_naive_pct']:.1f}%" if naive_mae else ""))
    return row


# ─────────────────────────────────────────────
# 1. Naive baseline（lag-7）
# ─────────────────────────────────────────────

def naive_forecast(valid_X: pd.DataFrame, valid_y: pd.Series) -> np.ndarray:
    """ŷ(t) = y(t-7)，直接使用已計算的 lag_7 特徵，零訓練成本。"""
    return valid_X["lag_7"].values


# ─────────────────────────────────────────────
# 2. SARIMA（statsmodels，AIC 網格搜尋）
# ─────────────────────────────────────────────

def sarima_forecast(train_y: pd.Series, n_forecast: int,
                    s: int = 7) -> tuple[np.ndarray, str]:
    """
    對 (p,d,q)(P,D,Q,s) 做網格搜尋，以 AIC 選最佳模型並預測未來 n_forecast 步。
    p,q ≤ 2，D 固定 = 0，s = 7。
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    best_aic    = np.inf
    best_order  = None
    best_params = None

    grid = [(p, 1, q, P, 0, Q)
            for p in range(3) for q in range(3)
            for P in range(2) for Q in range(2)]

    logger.info(f"  SARIMA 網格搜尋：{len(grid)} 組參數...")
    for p, d, q, P, D, Q in grid:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = SARIMAX(
                    train_y,
                    order=(p, d, q),
                    seasonal_order=(P, D, Q, s),
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit(disp=False)
            if m.aic < best_aic:
                best_aic   = m.aic
                best_order = (p, d, q, P, D, Q)
                best_model = m
        except Exception:
            continue

    if best_order is None:
        logger.warning("  SARIMA 所有參數皆收斂失敗，以 Naive 替代")
        return None, "SARIMA(failed)"

    p, d, q, P, D, Q = best_order
    label = f"SARIMA({p},{d},{q})({P},{D},{Q},{s}) AIC={best_aic:.1f}"
    logger.info(f"  最佳 {label}")

    pred = best_model.forecast(steps=n_forecast)
    pred = np.maximum(pred.values, 0)   # 事故數不可為負
    return pred, label


# ─────────────────────────────────────────────
# 3. XGBoost（主要挑戰者）
# ─────────────────────────────────────────────

def xgboost_forecast(train_X: pd.DataFrame, train_y: pd.Series,
                     valid_X: pd.DataFrame) -> np.ndarray:
    """
    n_estimators=500, lr=0.05, max_depth=5。
    以訓練集尾端 10% 作為 early_stopping 的內部驗證集（AC-07c）。
    """
    import xgboost as xgb

    n_eval = max(int(len(train_X) * 0.10), 30)
    X_tr = train_X.iloc[:-n_eval]
    y_tr = train_y.iloc[:-n_eval]
    X_ev = train_X.iloc[-n_eval:]
    y_ev = train_y.iloc[-n_eval:]

    model = xgb.XGBRegressor(
        n_estimators    = 500,
        learning_rate   = 0.05,
        max_depth       = 5,
        subsample       = 0.8,
        colsample_bytree= 0.8,
        random_state    = RANDOM_STATE,
        early_stopping_rounds = 30,
        eval_metric     = "mae",
        verbosity       = 0,
    )
    model.fit(
        X_tr, y_tr,
        eval_set        = [(X_ev, y_ev)],
        verbose         = False,
    )

    best_iter = model.best_iteration
    logger.info(f"  XGBoost best_iteration={best_iter}")
    pred = np.maximum(model.predict(valid_X), 0)
    return pred, model


# ─────────────────────────────────────────────
# 4. 全域熱點月 XGBoost（輔助模型，供 s07 排序）
# ─────────────────────────────────────────────

def build_hotspot_monthly_dataset(panel: pd.DataFrame,
                                  hot: pd.DataFrame) -> pd.DataFrame:
    """
    對 145 個熱點、60 個月份建立特徵：
      lag_1m ~ lag_12m（過去 12 個月事故數）
      + 靜態特徵：件數、A1、dist_any、dist_speed、kmeans_label（編碼）
    從月份索引 12 開始（確保有 12 個 lag 可用）。
    """
    # 靜態特徵（件數=五年總件數, A1, dist_any, dist_speed, kmeans_label）
    static = hot.set_index("cluster")[["件數","A1","dist_any","dist_speed","kmeans_label"]].copy()
    static["kmeans_label"] = pd.to_numeric(static["kmeans_label"], errors="coerce").fillna(0).astype(int)

    records = []
    months  = panel.columns.tolist()   # 60 個月份字串

    for cluster in panel.index:
        for i in range(12, len(months)):
            month = months[i]
            y_val = panel.loc[cluster, month]
            lags  = {f"lag_{k}m": panel.loc[cluster, months[i - k]] for k in range(1, 13)}
            stat  = static.loc[cluster].to_dict() if cluster in static.index else {}
            records.append({"cluster": cluster, "month": month, "y": y_val,
                            **lags, **stat})

    df = pd.DataFrame(records)
    df["month_dt"] = pd.to_datetime(df["month"])
    return df


def hotspot_monthly_xgb(panel: pd.DataFrame, hc: pd.DataFrame) -> pd.DataFrame:
    """
    全域 XGBoost：樣本 = (熱點, 月)，預測 2025 各月各熱點事故數。
    回傳驗證期 (145×12) 的預測 DataFrame。
    hc = hotspot_clusters.csv（含靜態特徵）
    """
    import xgboost as xgb

    # 補入靜態特徵：從 hotspots.csv 載入 A1、dist_any、dist_speed
    hot_full = pd.read_csv(PROC_DIR / "hotspots.csv", encoding="utf-8-sig")
    hot_full = hot_full[["cluster","A1","dist_any","dist_speed"]].set_index("cluster")
    hc = hc.set_index("cluster").join(hot_full).reset_index()

    df = build_hotspot_monthly_dataset(panel, hc)
    feat_cols = [c for c in df.columns if c not in ("cluster","month","y","month_dt")]

    train_df = df[df["month_dt"] <  pd.Timestamp(VALID_START)]
    valid_df = df[df["month_dt"] >= pd.Timestamp(VALID_START)]

    X_tr, y_tr = train_df[feat_cols], train_df["y"]
    X_va, y_va = valid_df[feat_cols], valid_df["y"]

    model = xgb.XGBRegressor(
        n_estimators = 300,
        learning_rate= 0.05,
        max_depth    = 4,
        subsample    = 0.8,
        random_state = RANDOM_STATE,
        verbosity    = 0,
    )
    model.fit(X_tr, y_tr)

    pred = np.maximum(model.predict(X_va), 0)
    mae  = mean_absolute_error(y_va, pred)
    rmse = np.sqrt(mean_squared_error(y_va, pred))
    logger.info(f"  [熱點月 XGBoost] Train={len(X_tr)}, Valid={len(X_va)}, MAE={mae:.3f}, RMSE={rmse:.3f}")

    # 整理成 (cluster × month) 形式供 s07 使用
    valid_df = valid_df.copy()
    valid_df["pred"] = pred
    pivot = valid_df.pivot(index="cluster", columns="month", values="pred")
    return pivot, {"model":"熱點月_XGBoost", "mae": round(mae,3), "rmse": round(rmse,3)}


# ─────────────────────────────────────────────
# 主程式
# ─────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 載入特徵矩陣
    start("載入 feature_matrix.csv")
    feat = pd.read_csv(
        PROC_DIR / "feature_matrix.csv",
        index_col="date", parse_dates=True, encoding="utf-8-sig",
    )
    train = feat[feat.index <= TRAIN_END]
    valid = feat[feat.index >= VALID_START]
    train_X, train_y = train[FEAT_COLS], train["y"]
    valid_X, valid_y = valid[FEAT_COLS], valid["y"]
    logger.info(f"  Train={len(train)}, Valid={len(valid)}")
    stop()

    all_metrics = []
    pred_df     = pd.DataFrame(index=valid.index)
    pred_df["y_true"] = valid_y.values

    # ── Model 1：Naive
    start("Naive(lag-7)")
    naive_pred = naive_forecast(valid_X, valid_y)
    m = evaluate(valid_y, naive_pred, "Naive(lag-7)")
    all_metrics.append(m)
    pred_df["Naive"] = naive_pred
    stop()
    naive_mae = m["mae"]

    # ── Model 2：SARIMA
    start("SARIMA 網格搜尋 + 預測")
    daily = pd.read_csv(
        PROC_DIR / "daily_series.csv",
        index_col="date", parse_dates=True, encoding="utf-8-sig",
    )["count"]
    train_daily = daily[daily.index <= TRAIN_END]
    sarima_pred, sarima_label = sarima_forecast(train_daily, n_forecast=len(valid_y))
    if sarima_pred is not None:
        # SARIMA 的預測期從訓練集最後一天的次日開始
        sarima_series = pd.Series(sarima_pred, index=valid_y.index)
        m = evaluate(valid_y, sarima_series.values, sarima_label, naive_mae)
    else:
        sarima_series = pd.Series(naive_pred, index=valid_y.index)
        m = {"model": sarima_label, "mae": None, "rmse": None}
    all_metrics.append(m)
    pred_df["SARIMA"] = sarima_series.values
    stop()

    # ── Model 3：XGBoost
    start("XGBoost 訓練 + 預測")
    xgb_pred, xgb_model = xgboost_forecast(train_X, train_y, valid_X)
    m = evaluate(valid_y, xgb_pred, "XGBoost", naive_mae)
    all_metrics.append(m)
    pred_df["XGBoost"] = xgb_pred

    # 驗收 AC-07b
    if m.get("improve_vs_naive_pct", -999) > 0:
        logger.info(f"[驗收 AC-07b] XGBoost 優於 Naive ✓（改善 {m['improve_vs_naive_pct']:.1f}%）")
    else:
        logger.warning(f"[驗收 AC-07b] XGBoost 未優於 Naive，請於報告說明原因")
    stop()

    # ── Model 4：全域熱點月 XGBoost
    start("全域熱點月 XGBoost")
    panel = pd.read_csv(PROC_DIR / "hotspot_monthly.csv", index_col="cluster", encoding="utf-8-sig")
    hc    = pd.read_csv(PROC_DIR / "hotspot_clusters.csv", encoding="utf-8-sig")
    monthly_pivot, m_hot = hotspot_monthly_xgb(panel, hc)
    all_metrics.append(m_hot)
    monthly_out = OUT_DIR / "hotspot_monthly_pred.csv"
    monthly_pivot.to_csv(monthly_out, encoding="utf-8-sig")
    logger.info(f"  熱點月預測輸出：{monthly_out}")
    stop()

    # ── 輸出 metrics.csv
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(OUT_DIR / "metrics.csv", index=False, encoding="utf-8-sig")
    logger.info(f"輸出：outputs/metrics.csv")

    # ── 輸出 pred.csv（驗證期逐日預測）
    pred_df.to_csv(OUT_DIR / "pred.csv", encoding="utf-8-sig")
    logger.info(f"輸出：outputs/pred.csv（{pred_df.shape}）")

    # ── 驗收摘要
    logger.info("=== 驗收摘要（AC-07） ===")
    logger.info(f"  AC-07a 各模型指標：")
    for row in all_metrics:
        logger.info(f"    {row}")
    logger.info("=== s06_forecast 完成 ===")


if __name__ == "__main__":
    main()
