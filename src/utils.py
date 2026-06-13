import time
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from config import LOG_DIR


def get_logger(name: str) -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{name}.log"

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)

    return logger


def min_haversine(points: pd.DataFrame, targets: pd.DataFrame) -> np.ndarray:
    """
    各 point 到 targets 集合中最近點的 haversine 距離（公尺）。
    points / targets 需含 lat, lon 欄位（度）。
    回傳 shape=(len(points),) 的 ndarray。
    """
    p_lat = np.radians(points["lat"].values)[:, None]   # (N, 1)
    p_lon = np.radians(points["lon"].values)[:, None]
    t_lat = np.radians(targets["lat"].values)[None, :]  # (1, M)
    t_lon = np.radians(targets["lon"].values)[None, :]

    dlat = t_lat - p_lat
    dlon = t_lon - p_lon
    a = np.sin(dlat / 2) ** 2 + np.cos(p_lat) * np.cos(t_lat) * np.sin(dlon / 2) ** 2
    dist = 2 * 6_371_000 * np.arcsin(np.sqrt(a))       # (N, M)，公尺
    return dist.min(axis=1)


def timer(logger: logging.Logger):
    """回傳 (start_fn, stop_fn)，stop_fn 會 log 耗時。"""
    state = {}

    def start(label: str):
        state["label"] = label
        state["t0"] = time.perf_counter()
        logger.info(f"[開始] {label}")

    def stop():
        elapsed = time.perf_counter() - state["t0"]
        logger.info(f"[完成] {state['label']} — 耗時 {elapsed:.1f}s")

    return start, stop
