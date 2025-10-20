# utils.py
# Common utils for: logging, seeding, pactus->DataFrame conversion, CSV I/O, and repo path handling.

from __future__ import annotations
import os
import sys
import time
import json
import math
import random
import shutil
import argparse
import ast
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, List, Tuple, Optional, Dict

import numpy as np
import pandas as pd

from yupi import Trajectory
from pactus import Dataset

# ---------- Reproducibility ----------
def set_seed(seed: int = 3112) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch  # optional
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True  # type: ignore
        torch.backends.cudnn.benchmark = False     # type: ignore
    except Exception:
        pass

# ---------- Simple logger ----------
class Logger:
    def __init__(self, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        self.path = os.path.join(out_dir, f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

    def log(self, msg: str) -> None:
        print(msg, flush=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

# ---------- Repo path utils ----------
def add_repo_to_syspath(repo_dir: str) -> None:
    """
    Append a cloned repo directory (e.g., ./external/Geolet) to sys.path so we can import it.
    """
    repo_dir = os.path.abspath(repo_dir)
    if not os.path.isdir(repo_dir):
        raise FileNotFoundError(f"Geolet repository not found at {repo_dir}")
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)

# ---------- pactus dataset helpers ----------
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DATASET_ROOT = os.path.join(REPO_ROOT, "dataset")


def _ensure_file_exists(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required data file not found: {path}")


def _ensure_dir_exists(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required data directory not found: {path}")


def _load_custom_animals_csv_dataset() -> Dataset:
    csv_path = os.path.join(DATASET_ROOT, "animals", "train_csv", "all_data.csv")
    _ensure_file_exists(csv_path)

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"Animals CSV dataset at {csv_path} is empty")

    trajs = []
    labels = []

    required_cols = {"index", "trajectory_name", "x", "y"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"Animals CSV missing columns: {missing}")

    for (_, _), group in df.groupby(["index", "trajectory_name"], sort=True):
        x = group["x"].to_numpy(dtype=float)
        y = group["y"].to_numpy(dtype=float)
        t = group["t"].to_numpy(dtype=float) if "t" in group.columns else np.arange(len(group), dtype=float)

        traj = Trajectory(x=x, y=y, t=t if t is not None else None)
        trajs.append(traj)
        label_col = "label" if "label" in group.columns else None
        labels.append(group[label_col].iloc[0] if label_col else str(group["index"].iloc[0]))

    return Dataset("animals_local", trajs, labels)


def _load_custom_seabird_dataset() -> Dataset:
    csv_path = os.path.join(DATASET_ROOT, "seabird", "anon_gps_tracks_with_dive.csv")
    _ensure_file_exists(csv_path)

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"Seabird CSV dataset at {csv_path} is empty")

    required_cols = {"bird", "lat", "lon"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"Seabird CSV missing columns: {missing}")

    trajs = []
    labels = []

    for bird_id, group in df.groupby("bird", sort=True):
        lat = group["lat"].to_numpy(dtype=float)
        lon = group["lon"].to_numpy(dtype=float)
        alt = group["alt"].to_numpy(dtype=float) if "alt" in group.columns else None

        traj = Trajectory(x=lat, y=lon, z=alt if alt is not None else None)
        trajs.append(traj)
        label_col = "species" if "species" in group.columns else None
        labels.append(group[label_col].iloc[0] if label_col else str(bird_id))

    return Dataset("seabird_local", trajs, labels)


def _load_custom_taxi_dataset() -> Dataset:
    csv_path = os.path.join(DATASET_ROOT, "taxi", "train.csv")
    _ensure_file_exists(csv_path)

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"Taxi CSV dataset at {csv_path} is empty")

    required_cols = {"POLYLINE"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"Taxi CSV missing columns: {missing}")

    trajs = []
    labels = []

    for _, row in df.iterrows():
        try:
            polyline = ast.literal_eval(row["POLYLINE"]) if isinstance(row["POLYLINE"], str) else row["POLYLINE"]
        except (ValueError, SyntaxError) as exc:
            raise ValueError(f"Invalid POLYLINE entry in taxi dataset: {exc}") from exc

        if not polyline or len(polyline) < 2:
            continue

        arr = np.asarray(polyline, dtype=float)
        if arr.ndim != 2 or arr.shape[1] < 2:
            continue

        traj = Trajectory(x=arr[:, 0], y=arr[:, 1])
        trajs.append(traj)
        labels.append(row.get("CALL_TYPE", "unknown"))

    if not trajs:
        raise ValueError("No valid trajectories found in taxi dataset")

    return Dataset("taxi_local", trajs, labels)


def _load_custom_vehicle_dataset() -> Dataset:
    train_dir = os.path.join(DATASET_ROOT, "vehicle", "train")
    test_dir = os.path.join(DATASET_ROOT, "vehicle", "test")
    _ensure_dir_exists(os.path.dirname(train_dir))

    dirs = [d for d in (train_dir, test_dir) if os.path.exists(d)]
    if not dirs:
        raise FileNotFoundError("Vehicle dataset directories not found. Expected train/ or test/ under dataset/vehicle/")

    filename_re = re.compile(
        r"""^\s*(?P<index>\d+)\s+(?P<traj>s\d+)\s+(?P<label>[A-Za-z]+)\.r\d+\s*$""",
        re.VERBOSE,
    )

    def parse_filename(name: str):
        match = filename_re.match(name)
        return match.groupdict() if match else None

    def parse_line(line: str):
        line = line.strip()
        if not line:
            return None
        parts = line.split()
        if len(parts) != 2:
            return None
        tx, y_str = parts
        if "," not in tx:
            return None
        t_str, x_str = tx.split(",", 1)
        try:
            return float(t_str), float(x_str), float(y_str)
        except ValueError:
            return None

    trajs = []
    labels = []

    for base_dir in dirs:
        for fname in os.listdir(base_dir):
            file_path = os.path.join(base_dir, fname)
            if not os.path.isfile(file_path):
                continue
            parsed = parse_filename(fname)
            if not parsed:
                continue

            points = []
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    parsed_line = parse_line(line)
                    if parsed_line:
                        points.append(parsed_line)

            if not points:
                continue

            arr = np.asarray(points, dtype=float)
            t = arr[:, 0]
            x = arr[:, 1]
            y = arr[:, 2]

            traj = Trajectory(x=x, y=y, t=t)
            trajs.append(traj)
            labels.append(parsed["label"])

    if not trajs:
        raise ValueError("No valid trajectories found in vehicle dataset")

    return Dataset("vehicle_local", trajs, labels)


_BUILTIN_DATASET_FACTORIES = {
    # "geolife": Dataset.geolife,
    # "animals": Dataset.animals,
    # "hurdat2": Dataset.hurdat2,
    # "cma_bst": Dataset.cma_bst,
    # "mnist_stroke": Dataset.mnist_stroke,
    # "uci_pen_digits": Dataset.uci_pen_digits,
    # "uci_gotrack": Dataset.uci_gotrack,
    # "uci_characters": Dataset.uci_characters,
    # "uci_movement_libras": Dataset.uci_movement_libras,
    # "diffusive_particles": Dataset.diffusive_particles,
    "traffic": Dataset.traffic,
}

_CUSTOM_DATASET_FACTORIES = {
    # "animals_local": _load_custom_animals_csv_dataset,
    # "seabird_local": _load_custom_seabird_dataset,
    # "taxi_local": _load_custom_taxi_dataset,
    "vehicle_local": _load_custom_vehicle_dataset,
}


def list_available_dataset_names(include_custom: bool = True) -> List[str]:
    names = list(_BUILTIN_DATASET_FACTORIES.keys())
    if include_custom:
        names.extend(_CUSTOM_DATASET_FACTORIES.keys())
    return sorted(names)


def get_pactus_dataset_by_name(name: str):
    """Returns a pactus Dataset for known names (built-in + custom)."""

    key = name.strip().lower()

    if key in _BUILTIN_DATASET_FACTORIES:
        return _BUILTIN_DATASET_FACTORIES[key]()

    if key in _CUSTOM_DATASET_FACTORIES:
        return _CUSTOM_DATASET_FACTORIES[key]()

    raise ValueError(
        "Unknown dataset '{name}'. Supported: {supported}".format(
            name=name,
            supported=list_available_dataset_names(include_custom=True),
        )
    )

def _trajectory_to_rows(traj) -> List[Dict[str, float]]:
    """
    Supports:
      - yupi.Trajectory (iterable of TrajectoryPoint(.r -> [c1,c2], .t -> time))
      - np.ndarray of shape (N,2) or (N,3)
      - list of [c1,c2,(t)]
    Returns list of dicts: {"c1":..., "c2":..., "t":...}
    """
    rows: List[Dict[str, float]] = []

    # Case 1: yupi.Trajectory (duck-typing: iterable, has .copy attr, but not ndarray/list)
    if hasattr(traj, "__iter__") and hasattr(traj, "copy") and not isinstance(traj, (np.ndarray, list, tuple)):
        for p in traj:  # TrajectoryPoint
            c1 = float(p.r[0]) if hasattr(p, "r") else None
            c2 = float(p.r[1]) if hasattr(p, "r") else None
            t  = float(p.t) if hasattr(p, "t") and p.t is not None else None
            rows.append({"c1": c1, "c2": c2, "t": t})
        return rows

    # Case 2: numpy array
    arr = np.asarray(traj)
    if arr.ndim == 2 and arr.shape[1] >= 2:
        for i in range(arr.shape[0]):
            c1 = float(arr[i, 0])
            c2 = float(arr[i, 1])
            t  = float(arr[i, 2]) if arr.shape[1] >= 3 else None
            rows.append({"c1": c1, "c2": c2, "t": t})
        return rows

    # Case 3: generic list of points
    for pt in traj:
        if isinstance(pt, (list, tuple)) and len(pt) >= 2:
            c1 = float(pt[0]); c2 = float(pt[1])
            t  = float(pt[2]) if len(pt) >= 3 else None
            rows.append({"c1": c1, "c2": c2, "t": t})
    return rows

def pactus_to_df(ds) -> pd.DataFrame:
    """
    Convert pactus Dataset -> tidy DataFrame with columns: tid, class, c1, c2, t
    """
    parts: List[pd.DataFrame] = []
    for tid, (traj, label) in enumerate(zip(ds.trajs, ds.labels)):
        rows = _trajectory_to_rows(traj)
        if not rows:
            continue
        part = pd.DataFrame(rows)
        if "t" not in part or part["t"].isna().all():
            part["t"] = np.arange(len(part), dtype=float)
        part["tid"] = tid
        part["class"] = label
        part = part[["tid", "class", "c1", "c2", "t"]].sort_values(["tid", "t"])
        parts.append(part)
    if not parts:
        return pd.DataFrame(columns=["tid","class","c1","c2","t"])
    return pd.concat(parts, ignore_index=True)

def save_df_to_csv(df: pd.DataFrame, out_dir: str, dataset_name: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    fp = os.path.join(out_dir, f"{dataset_name}.csv")
    df.to_csv(fp, index=False)
    return fp
