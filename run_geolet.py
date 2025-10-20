# run_geolet.py
# Run Geolet on all CSVs in a directory. Import Geolet from a cloned repo path.

import os
import glob
import time
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from utils import Logger, set_seed, add_repo_to_syspath

def run_geolet_on_df(df: pd.DataFrame, random_state: int = 3):
    # lazy import after sys.path is patched
    from Geolet.classifier.geoletclassifier import GeoletClassifier, prepare_y
    from Geolet import distancers

    by_tid = df.groupby("tid")["class"].max().reset_index()
    tid_all = by_tid["tid"].to_numpy()
    y_all = by_tid["class"].to_numpy()

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
        n_jobs=4,
    )

    df_tr = df[df.tid.isin(tid_tr)]
    X_tr = df_tr[["tid","t","c1","c2"]].values
    y_tr = df_tr["class"].values

    t0 = time.perf_counter()
    geo.fit(X_tr, y_tr)
    _, X_dist = geo.transform(df[["tid","t","c1","c2"]].values)
    elapsed = time.perf_counter() - t0

    y_tid = prepare_y(classes=df["class"].values, tids=df["tid"].values)
    Xd_tr, Xd_te, y_tr2, y_te2 = train_test_split(
        X_dist, y_tid, test_size=0.3, stratify=y_tid, random_state=random_state
    )

    clf = RandomForestClassifier(random_state=random_state)
    clf.fit(Xd_tr, y_tr2)
    y_pred = clf.predict(Xd_te)

    acc = accuracy_score(y_te2, y_pred)
    f1m = f1_score(y_te2, y_pred, average="weighted")
    return acc, f1m, elapsed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-dir", default="./data_csv", help="Folder contains *.csv")
    parser.add_argument("--out", default="./results1/geolet_results.csv")
    parser.add_argument(
        "--geolet-dir",
        default="./external/Geolet",
        help="Path to the cloned Geolet repository (package root containing Geolet/)",
    )
    parser.add_argument("--seed", type=int, default=3112)
    args = parser.parse_args()

    set_seed(args.seed)
    logger = Logger(out_dir="./results1")
    add_repo_to_syspath(args.geolet_dir)

    records = []
    csv_files = sorted(glob.glob(os.path.join(args.csv_dir, "*.csv")))
    if not csv_files:
        logger.log(f"[geolet] No CSVs found in {args.csv_dir}")
        return

    for fp in csv_files:
        try:
            dataset = os.path.splitext(os.path.basename(fp))[0]
            df = pd.read_csv(fp).sort_values(["tid","t"])
            logger.log(f"[geolet][{dataset}] rows={len(df)} start...")
            acc, f1m, sec = run_geolet_on_df(df, random_state=3)
            logger.log(f"[geolet][{dataset}] ACC={acc:.4f} F1={f1m:.4f} time={sec:.3f}s")
            records.append({"dataset": dataset, "model": "Geolet", "accuracy": acc, "f1": f1m, "eval_seconds": sec})
        except Exception as e:
            logger.log(f"[geolet][{fp}] ERROR: {e}")
            records.append({"dataset": dataset, "model": "Geolet", "error": str(e)})

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    pd.DataFrame(records).to_csv(args.out, index=False)
    logger.log(f"[geolet] Saved -> {args.out}")

if __name__ == "__main__":
    main()
