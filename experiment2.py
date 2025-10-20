# =============================================================================
# IMPORTS (ESSENTIAL)
# =============================================================================
import os

# =============================================================================
# CONFIGURATION
# =============================================================================
SEED = 3112
NO_OF_EPOCHS = 20
BATCH_SIZE = 32

# Output directory for results
OUTPUT_DIR = "/Users/phuc.buidang/Documents/UIT/XAI/results2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# IMPORTS
# =============================================================================
import ast
import io
import re
import time
import contextlib
import traceback
from datetime import datetime

import pandas as pd
import numpy as np

# TensorFlow (nếu còn dùng LSTM pactus)
import tensorflow as tf
try:
    tf.config.optimizer.set_jit(False)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            gpu = gpus[0]
            tf.config.experimental.set_visible_devices([gpu], 'GPU')
            tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU configuration successful. Using GPU: {gpu.name}")
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
            tf.config.set_visible_devices([], 'GPU')
            print("Falling back to CPU")
    else:
        print("No GPUs found, using CPU")
except Exception as e:
    print(f"TensorFlow configuration error: {e}")

# pactus / yupi
from yupi import Trajectory
from pactus import Dataset

# Optional: LSTM pactus (nếu vẫn muốn so sánh)
from pactus.models import LSTMModel

# Geolet
from Geolet.classifier.geoletclassifier import GeoletClassifier, prepare_y
from Geolet import distancers

# TCIF (TIF)
from TCIF.algorithms.utils import prepare as tif_prepare
from TCIF.classes.T_CIF_observation import T_CIF_observations
from TCIF.classes.T_CIF_space import T_CIF_space
from TCIF.classes.T_CIF_time import T_CIF_time

# =============================================================================
# FILE OUTPUT UTILITIES
# =============================================================================
class FileLogger:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.log_file = os.path.join(output_dir, f"experiment_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    def log(self, message):
        print(message)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(message + '\n')

logger = FileLogger(OUTPUT_DIR)

# =============================================================================
# DATA LOADING (CUSTOM EXAMPLES YOU ALREADY HAD) - optional
# =============================================================================
BASE_DIR = "/home/phucbuidang/Work/XAI4Class/dataset"

def load_animals_dataset_custom_csv():
    try:
        animals_csv_path = os.path.join(BASE_DIR, "animals/train_csv/all_data.csv")
        if not os.path.exists(animals_csv_path):
            logger.log(f"[animals custom] CSV not found: {animals_csv_path}")
            return None
        df = pd.read_csv(animals_csv_path)
        trajs, labels = [], []
        for (idx, traj_name), g in df.groupby(["index", "trajectory_name"]):
            traj = Trajectory(x=g["x"].to_numpy(),
                              y=g["y"].to_numpy(),
                              t=g["t"].to_numpy() if "t" in g.columns else None)
            trajs.append(traj)
            labels.append(g["label"].iloc[0])
        ds = Dataset("animals_custom", trajs, labels)
        logger.log(f"[animals custom] loaded: {len(trajs)} trajectories, labels={set(labels)}")
        return ds
    except Exception as e:
        logger.log(f"[animals custom] error: {e}")
        return None

def load_seabird_dataset_custom_csv():
    try:
        seabird_csv_path = os.path.join(BASE_DIR, "seabird/anon_gps_tracks_with_dive.csv")
        if not os.path.exists(seabird_csv_path):
            logger.log(f"[seabird custom] CSV not found: {seabird_csv_path}")
            return None
        df = pd.read_csv(seabird_csv_path)
        trajs, labels = [], []
        for bird_id, g in df.groupby("bird"):
            traj = Trajectory(x=g["lat"].to_numpy(),
                              y=g["lon"].to_numpy(),
                              t=g["time"].to_numpy() if "time" in g.columns else None)
            trajs.append(traj)
            labels.append(g["species"].iloc[0] if "species" in g.columns else str(bird_id))
        ds = Dataset("seabird_custom", trajs, labels)
        logger.log(f"[seabird custom] loaded: {len(trajs)} trajectories, labels={set(labels)}")
        return ds
    except Exception as e:
        logger.log(f"[seabird custom] error: {e}")
        return None

# =============================================================================
# DATASET COLLECTION (pactus built-ins + custom)
# =============================================================================
def get_all_pactus_datasets():
    datasets = []
    # built-in
    for getter in [
        Dataset.geolife, Dataset.animals, Dataset.hurdat2, Dataset.cma_bst,
        Dataset.mnist_stroke, Dataset.uci_pen_digits, Dataset.uci_gotrack,
        Dataset.uci_characters, Dataset.uci_movement_libras, Dataset.diffusive_particles,
        Dataset.traffic,
    ]:
        try:
            ds = getter()
            if ds is not None:
                datasets.append(ds)
        except Exception as e:
            logger.log(f"[pactus] skip {getter.__name__} due to error: {e}")

    # custom (optional)
    for load_fn in [load_animals_dataset_custom_csv, load_seabird_dataset_custom_csv]:
        try:
            ds = load_fn()
            if ds is not None:
                datasets.append(ds)
        except Exception as e:
            logger.log(f"[custom] loader error: {e}")

    return datasets

# =============================================================================
# CONVERSION: pactus Dataset -> DataFrame -> CSV
# =============================================================================
def trajectory_to_rows(traj):
    """
    Hỗ trợ cả 3 dạng:
    - yupi.Trajectory (iterable ra TrajectoryPoint có .r và .t)
    - np.ndarray shape (N,2) hoặc (N,3) (x/y/(t) hoặc lon/lat/(t))
    - list of tuples
    Return: list[dict{c1,c2,t}]
    """
    rows = []
    # yupi.Trajectory: có thuộc tính .__iter__ ra TrajectoryPoint
    if hasattr(traj, "__iter__") and hasattr(traj, "copy") and not isinstance(traj, (np.ndarray, list, tuple)):
        # assume yupi.Trajectory
        for p in traj:
            # p.r: Vector([c1, c2]), p.t: time (có thể None)
            c1 = float(p.r[0]) if hasattr(p, "r") else None
            c2 = float(p.r[1]) if hasattr(p, "r") else None
            t  = float(p.t) if hasattr(p, "t") and p.t is not None else None
            rows.append({"c1": c1, "c2": c2, "t": t})
        return rows

    # numpy array
    arr = np.asarray(traj)
    if arr.ndim == 2 and arr.shape[1] >= 2:
        for i in range(arr.shape[0]):
            c1 = float(arr[i, 0])
            c2 = float(arr[i, 1])
            t  = float(arr[i, 2]) if arr.shape[1] >= 3 else None
            rows.append({"c1": c1, "c2": c2, "t": t})
        return rows

    # list of points
    for pt in traj:
        if isinstance(pt, (list, tuple)) and len(pt) >= 2:
            c1 = float(pt[0]); c2 = float(pt[1])
            t  = float(pt[2]) if len(pt) >= 3 else None
            rows.append({"c1": c1, "c2": c2, "t": t})
        else:
            # không rõ định dạng → bỏ qua
            continue
    return rows

def pactus_to_df(ds: Dataset) -> pd.DataFrame:
    parts = []
    for tid, (traj, label) in enumerate(zip(ds.trajs, ds.labels)):
        rows = trajectory_to_rows(traj)
        if not rows:
            continue
        part = pd.DataFrame(rows)
        # nếu thiếu t → dùng chỉ số
        if "t" not in part or part["t"].isna().all():
            part["t"] = np.arange(len(part), dtype=float)
        part["tid"] = tid
        part["class"] = label
        # sắp theo thời gian
        part = part[["tid", "class", "c1", "c2", "t"]].sort_values(["tid", "t"])
        parts.append(part)
    if not parts:
        return pd.DataFrame(columns=["tid","class","c1","c2","t"])
    df = pd.concat(parts, ignore_index=True)
    return df

def save_df_to_csv(df: pd.DataFrame, dataset_name: str) -> str:
    out_dir = os.path.join(OUTPUT_DIR, "csv")
    os.makedirs(out_dir, exist_ok=True)
    fp = os.path.join(out_dir, f"{dataset_name}.csv")
    df.to_csv(fp, index=False)
    logger.log(f"[CSV] Saved: {fp} (rows={len(df)})")
    return fp

# =============================================================================
# GEOLET RUN
# =============================================================================
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

def run_geolet_on_df(df: pd.DataFrame, random_state: int = 3):
    """
    Input df: columns tid,class,c1,c2,t
    Return: dict(metrics)
    """
    if df.empty:
        return {"model":"Geolet","accuracy":None,"f1":None,"eval_seconds":None}

    # split theo trajectory-level
    by_tid = df.groupby("tid").agg({"class":"max"}).reset_index()
    tid_all = by_tid["tid"].to_numpy()
    y_all  = by_tid["class"].to_numpy()

    tid_tr, tid_te, _, _ = train_test_split(
        tid_all, y_all, test_size=0.3, stratify=y_all, random_state=random_state
    )

    geo = GeoletClassifier(
        precision=3,
        geolet_per_class=10,
        selector="MutualInformation",
        top_k=5,
        trajectory_for_stats=100,
        bestFittingMeasure=distancers.InterpolatedRouteDistance.interpolatedRootDistanceBestFitting,
        distancer="IRD",
        verbose=False,
        n_jobs=4
    )

    # Train trên records thuộc tid_tr
    df_tr = df[df.tid.isin(tid_tr)]
    X_tr = df_tr[["tid","t","c1","c2"]].values  # Geolet đọc theo cột vị trí/tid/t
    y_tr = df_tr["class"].values
    # y chuẩn hoá theo tid:
    y_tid = prepare_y(classes=df["class"].values, tids=df["tid"].values)

    t0 = time.perf_counter()
    geo.fit(X_tr, y_tr)
    # transform toàn bộ để có ma trận khoảng cách
    _, X_dist = geo.transform(df[["tid","t","c1","c2"]].values)
    elapsed = time.perf_counter() - t0

    # chia train/test trên ma trận khoảng cách bằng y_tid
    Xd_tr, Xd_te, y_tr2, y_te2 = train_test_split(
        X_dist, y_tid, test_size=0.3, stratify=y_tid, random_state=random_state
    )

    clf = RandomForestClassifier(random_state=random_state)
    clf.fit(Xd_tr, y_tr2)
    y_pred = clf.predict(Xd_te)

    acc = accuracy_score(y_te2, y_pred)
    f1  = f1_score(y_te2, y_pred, average="weighted")
    return {"model":"Geolet","accuracy":acc,"f1":f1,"eval_seconds":elapsed}

# =============================================================================
# TIF (TCIF) RUN
# =============================================================================
def run_tif_on_df(df: pd.DataFrame, random_state: int = 3, variant: str = "observations"):
    """
    variant: one of {"observations","time","space"}
    """
    if df.empty:
        return {"model":f"TIF-{variant}","accuracy":None,"f1":None,"eval_seconds":None}

    # split theo trajectory-level
    by_tid = df.groupby("tid").agg({"class":"max"}).reset_index()
    tid_all = by_tid["tid"]; y_all = by_tid["class"]

    tid_tr, tid_te, _, _ = train_test_split(
        tid_all, y_all, test_size=0.3, stratify=y_all, random_state=random_state
    )

    # TCIF prepare
    id_tr, y_tr, lat_tr, lon_tr, time_tr = tif_prepare(df, tid_tr)
    id_te, y_te, lat_te, lon_te, time_te = tif_prepare(df, tid_te)
    X_tr = [(_lat, _lon, _time) for _lat, _lon, _time in zip(lat_tr, lon_tr, time_tr)]
    X_te = [(_lat, _lon, _time) for _lat, _lon, _time in zip(lat_te, lon_te, time_te)]

    if variant == "observations":
        model = T_CIF_observations(n_trees=500, n_interval=50, min_length=10, interval_type=None)
    elif variant == "time":
        model = T_CIF_time(n_trees=500, n_interval=50, min_length=10)
    else:
        model = T_CIF_space(n_trees=500, n_interval=50, min_length=10)

    t0 = time.perf_counter()
    model.fit(X_tr, y=y_tr)
    y_pred = model.predict(X_te)
    elapsed = time.perf_counter() - t0

    # đánh giá
    from sklearn.metrics import accuracy_score, f1_score
    acc = accuracy_score(y_te, y_pred)
    f1  = f1_score(y_te, y_pred, average="weighted")
    return {"model":f"TIF-{variant}","accuracy":acc,"f1":f1,"eval_seconds":elapsed}

# =============================================================================
# (OPTIONAL) PACTUS LSTM RUN – nếu vẫn muốn giữ
# =============================================================================
def run_lstm_pactus(train_data: Dataset, test_data: Dataset, dataset_name: str):
    try:
        if tf.config.list_physical_devices('GPU'):
            tf.keras.backend.clear_session()
        model = LSTMModel(
            optimizer="rmsprop",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        model.train(train_data, train_data, epochs=NO_OF_EPOCHS, batch_size=BATCH_SIZE)
        t0 = time.perf_counter()
        evaluation = model.evaluate(test_data)
        elapsed = time.perf_counter() - t0
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            evaluation.show()
        out = buf.getvalue()
        acc = re.search(r"Accuracy:\s*([0-9.]+)", out)
        f1m = re.search(r"F1-score:\s*([0-9.]+)", out)
        acc = float(acc.group(1)) if acc else None
        f1m = float(f1m.group(1)) if f1m else None
        return {"model":"LSTM(pactus)","dataset":dataset_name,"accuracy":acc,"f1":f1m,"eval_seconds":elapsed}
    except Exception as e:
        logger.log(f"[LSTM] error on {dataset_name}: {e}")
        return {"model":"LSTM(pactus)","dataset":dataset_name,"error":str(e),"accuracy":None,"f1":None,"eval_seconds":None}

# =============================================================================
# MAIN
# =============================================================================
def main():
    datasets = get_all_pactus_datasets()
    logger.log(f"\nCollected {len(datasets)} datasets.")
    results = []
    csv_index = []

    for ds in datasets:
        try:
            logger.log("\n" + "="*60)
            logger.log(f"Processing dataset: {ds.name}")
            df = pactus_to_df(ds)

            if df.empty:
                logger.log(f"[{ds.name}] empty after conversion – skip.")
                continue

            # Lưu CSV
            csv_path = save_df_to_csv(df, ds.name)
            csv_index.append({"dataset": ds.name, "csv_path": csv_path, "rows": len(df)})

            # RUN GEOLET
            try:
                geo_res = run_geolet_on_df(df)
                geo_res["dataset"] = ds.name
                results.append(geo_res)
                logger.log(f"[Geolet][{ds.name}] acc={geo_res['accuracy']:.4f}, f1={geo_res['f1']:.4f}, time={geo_res['eval_seconds']:.3f}s")
            except Exception as e:
                logger.log(f"[Geolet][{ds.name}] error: {e}")
                results.append({"model":"Geolet","dataset":ds.name,"error":str(e),"accuracy":None,"f1":None,"eval_seconds":None})

            # RUN TIF (chọn 1–2 biến thể cho nhanh)
            for variant in ["observations", "time"]:
                try:
                    tif_res = run_tif_on_df(df, variant=variant)
                    tif_res["dataset"] = ds.name
                    results.append(tif_res)
                    logger.log(f"[TIF-{variant}][{ds.name}] acc={tif_res['accuracy']:.4f}, f1={tif_res['f1']:.4f}, time={tif_res['eval_seconds']:.3f}s")
                except Exception as e:
                    logger.log(f"[TIF-{variant}][{ds.name}] error: {e}")
                    results.append({"model":f"TIF-{variant}","dataset":ds.name,"error":str(e),"accuracy":None,"f1":None,"eval_seconds":None})

            # (Optional) LSTM pactus baseline
            # tr, te = ds.split(train_size=0.7, random_state=SEED)
            # lstm_res = run_lstm_pactus(tr, te, ds.name)
            # results.append(lstm_res)

        except Exception as e:
            logger.log(f"[{ds.name}] fatal error: {e}")
            results.append({"dataset":ds.name,"error":str(e)})

    # Lưu results + index CSV
    res_df = pd.DataFrame(results)
    idx_df = pd.DataFrame(csv_index)

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    res_path = os.path.join(OUTPUT_DIR, f"geolet_tif_results_{ts}.csv")
    idx_path = os.path.join(OUTPUT_DIR, f"converted_csv_index_{ts}.csv")
    res_df.to_csv(res_path, index=False)
    idx_df.to_csv(idx_path, index=False)

    logger.log("\n=== FINISHED ===")
    logger.log(f"Results saved: {res_path}")
    logger.log(f"CSV index saved: {idx_path}")

if __name__ == "__main__":
    main()
