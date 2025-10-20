# run_tif.py
# Run TIF (TCIF) on all CSVs in a directory. Import TCIF from a cloned repo path.

import os
import glob
import time
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from utils import Logger, set_seed, add_repo_to_syspath

def run_tif_variant(df: pd.DataFrame, variant: str = "observations", random_state: int = 3):
    # Lazy import after sys.path is patched
    from TCIF.algorithms.utils import prepare as tif_prepare
    from TCIF.classes.T_CIF_observation import T_CIF_observations
    from TCIF.classes.T_CIF_time import T_CIF_time
    from TCIF.classes.T_CIF_space import T_CIF_space

    by_tid = df.groupby("tid")["class"].max().reset_index()
    tid_all = by_tid["tid"]; y_all = by_tid["class"]

    tid_tr, tid_te, _, _ = train_test_split(
        tid_all, y_all, test_size=0.3, stratify=y_all, random_state=random_state
    )

    # prepare for TCIF
    id_tr, y_tr, lat_tr, lon_tr, time_tr = tif_prepare(df, tid_tr)
    id_te, y_te, lat_te, lon_te, time_te = tif_prepare(df, tid_te)
    X_tr = [(_lat, _lon, _time) for _lat, _lon, _time in zip(lat_tr, lon_tr, time_tr)]
    X_te = [(_lat, _lon, _time) for _lat, _lon, _time in zip(lat_te, lon_te, time_te)]

    if variant == "observations":
        model = T_CIF_observations(n_trees=10, n_interval=5, min_length=5, interval_type=None)
    elif variant == "time":
        model = T_CIF_time(n_trees=500, n_interval=50, min_length=10)
    elif variant == "space":
        model = T_CIF_space(n_trees=500, n_interval=50, min_length=10)
    else:
        raise ValueError("variant must be one of: observations, time, space")

    t0 = time.perf_counter()
    model.fit(X_tr, y=y_tr)
    y_pred = model.predict(X_te)
    elapsed = time.perf_counter() - t0

    acc = accuracy_score(y_te, y_pred)
    f1m = f1_score(y_te, y_pred, average="weighted")
    return acc, f1m, elapsed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-dir", default="./data_csv", help="Folder contains *.csv")
    parser.add_argument("--out", default="./results/tif_results.csv")
    parser.add_argument("--tif-dir", default="./external/TCIF", help="Cloned TCIF repo path")
    parser.add_argument("--variants", nargs="+", default=["observations","time"],
                        help="Which TIF variants to run: observations time space")
    parser.add_argument("--seed", type=int, default=3112)
    args = parser.parse_args()

    set_seed(args.seed)
    logger = Logger(out_dir="./results")
    add_repo_to_syspath(args.tif_dir)

    csv_files = sorted(glob.glob(os.path.join(args.csv_dir, "*.csv")))
    if not csv_files:
        logger.log(f"[tif] No CSVs found in {args.csv_dir}")
        return

    rows = []
    for fp in csv_files:
        dataset = os.path.splitext(os.path.basename(fp))[0]
        try:
            df = pd.read_csv(fp).sort_values(["tid","t"])
        except Exception as e:
            logger.log(f"[tif][{dataset}] read error: {e}")
            rows.append({"dataset": dataset, "model": "TIF", "error": str(e)})
            continue

        for variant in args.variants:
            try:
                logger.log(f"[tif-{variant}][{dataset}] rows={len(df)} start...")
                acc, f1m, sec = run_tif_variant(df, variant=variant, random_state=3)
                logger.log(f"[tif-{variant}][{dataset}] ACC={acc:.4f} F1={f1m:.4f} time={sec:.3f}s")
                rows.append({"dataset": dataset, "model": f"TIF-{variant}", "accuracy": acc, "f1": f1m, "eval_seconds": sec})
            except Exception as e:
                logger.log(f"[tif-{variant}][{dataset}] ERROR: {e}")
                rows.append({"dataset": dataset, "model": f"TIF-{variant}", "error": str(e)})

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    logger.log(f"[tif] Saved -> {args.out}")

if __name__ == "__main__":
    main()
