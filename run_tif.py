# run_tif.py
# Run TIF (TCIF) on all CSVs in a directory. Import TCIF from a cloned repo path.

import os
import glob
import time
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from itertools import product

from utils import Logger, set_seed, add_repo_to_syspath

def run_tif_variant(df: pd.DataFrame, variant: str = "observations", random_state: int = 3):
    # Lazy import after sys.path is patched
    from TCIF.algorithms.utils import prepare as tif_prepare
    from TCIF.classes.T_CIF_observation import T_CIF_observations
    # from TCIF.classes.T_CIF_time import T_CIF_time
    # from TCIF.classes.T_CIF_space import T_CIF_space

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
    # elif variant == "time":
    #     model = T_CIF_time(n_trees=500, n_interval=50, min_length=10)
    # elif variant == "space":
    #     model = T_CIF_space(n_trees=500, n_interval=50, min_length=10)
    # else:
    #     raise ValueError("variant must be one of: observations, time, space")

    t0 = time.perf_counter()
    model.fit(X_tr, y=y_tr)
    y_pred = model.predict(X_te)
    elapsed = time.perf_counter() - t0

    acc = accuracy_score(y_te, y_pred)
    f1m = f1_score(y_te, y_pred, average="weighted")
    return acc, f1m, elapsed

def run_tif_variant_with_params(df: pd.DataFrame, variant: str = "observations", 
                               n_trees: int = 10, n_interval: int = 5, min_length: int = 5,
                               random_state: int = 3):
    # Lazy import after sys.path is patched
    from TCIF.algorithms.utils import prepare as tif_prepare
    from TCIF.classes.T_CIF_observation import T_CIF_observations

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
        model = T_CIF_observations(n_trees=n_trees, n_interval=n_interval, min_length=min_length, interval_type=None)
    # elif variant == "time":
    #     model = T_CIF_time(n_trees=n_trees, n_interval=n_interval, min_length=min_length)
    # elif variant == "space":
    #     model = T_CIF_space(n_trees=n_trees, n_interval=n_interval, min_length=min_length)
    # else:
    #     raise ValueError("variant must be one of: observations, time, space")

    model.fit(X_tr, y=y_tr)
    t0 = time.perf_counter()
    y_pred = model.predict(X_te)
    elapsed = time.perf_counter() - t0

    acc = accuracy_score(y_te, y_pred)
    f1m = f1_score(y_te, y_pred, average="weighted")
    return acc, f1m, elapsed

def grid_search_tif(df: pd.DataFrame, variant: str = "observations", random_state: int = 3112):
    """Perform grid search to find parameters with lowest accuracy"""
    # Define parameter grid
    param_grid = {
        'n_trees': [3, 5, 10, 15, 20, 25],
        'n_interval': [1, 2, 3, 5, 7, 10],
        'min_length': [2, 3, 5, 7, 8]
    }
    
    # Generate all combinations
    param_combinations = list(product(
        param_grid['n_trees'],
        param_grid['n_interval'], 
        param_grid['min_length']
    ))
    
    results = []
    best_acc = float('inf')
    worst_params = None
    
    for n_trees, n_interval, min_length in param_combinations:
        try:
            acc, f1m, elapsed = run_tif_variant_with_params(
                df, variant=variant, 
                n_trees=n_trees, n_interval=n_interval, min_length=min_length,
                random_state=random_state
            )
            
            result = {
                'n_trees': n_trees,
                'n_interval': n_interval,
                'min_length': min_length,
                'accuracy': acc,
                'f1': f1m,
                'eval_seconds': elapsed
            }
            results.append(result)
            
            # Track lowest accuracy
            if acc < best_acc:
                best_acc = acc
                worst_params = result
                
        except Exception as e:
            results.append({
                'n_trees': n_trees,
                'n_interval': n_interval,
                'min_length': min_length,
                'error': str(e)
            })
    
    return results, worst_params

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-dir", default="./data_csv", help="Folder contains *.csv")
    parser.add_argument("--out", default="./results/tif_results.csv")
    parser.add_argument("--tif-dir", default="./external/TIF", help="Cloned TIF repo path")
    parser.add_argument("--variants", nargs="+", default=["observations"],
                        help="Which TIF variants to run: observations time space")
    parser.add_argument("--grid-search", action="store_true", 
                        help="Perform grid search to find worst parameters")
    parser.add_argument("--seed", type=int, default=3112)
    args = parser.parse_args()

    set_seed(args.seed)
    logger = Logger(out_dir="./resultss")
    add_repo_to_syspath(args.tif_dir)

    csv_files = sorted(glob.glob(os.path.join(args.csv_dir, "*.csv")))
    if not csv_files:
        logger.log(f"[tif] No CSVs found in {args.csv_dir}")
        return

    all_results = []
    worst_results = []
    
    for fp in csv_files:
        dataset = os.path.splitext(os.path.basename(fp))[0]
        try:
            df = pd.read_csv(fp).sort_values(["tid","t"])
        except Exception as e:
            logger.log(f"[tif][{dataset}] read error: {e}")
            all_results.append({"dataset": dataset, "model": "TIF", "error": str(e)})
            continue

        for variant in args.variants:
            if args.grid_search:
                try:
                    logger.log(f"[tif-{variant}][{dataset}] Grid search starting...")
                    results, worst_params = grid_search_tif(df, variant=variant, random_state=3)
                    
                    # Add dataset info to all results
                    for result in results:
                        result['dataset'] = dataset
                        result['model'] = f"TIF-{variant}"
                        all_results.append(result)
                    
                    if worst_params:
                        worst_params['dataset'] = dataset
                        worst_params['model'] = f"TIF-{variant}"
                        worst_results.append(worst_params)
                        logger.log(f"[tif-{variant}][{dataset}] Worst ACC={worst_params['accuracy']:.4f} "
                                 f"with params: n_trees={worst_params['n_trees']}, "
                                 f"n_interval={worst_params['n_interval']}, "
                                 f"min_length={worst_params['min_length']}")
                        
                except Exception as e:
                    logger.log(f"[tif-{variant}][{dataset}] Grid search ERROR: {e}")
                    all_results.append({"dataset": dataset, "model": f"TIF-{variant}", "error": str(e)})
            else:
                try:
                    logger.log(f"[tif-{variant}][{dataset}] rows={len(df)} start...")
                    acc, f1m, sec = run_tif_variant_with_params(df, variant=variant, random_state=3)
                    logger.log(f"[tif-{variant}][{dataset}] ACC={acc:.4f} F1={f1m:.4f} time={sec:.3f}s")
                    all_results.append({"dataset": dataset, "model": f"TIF-{variant}", "accuracy": acc, "f1": f1m, "eval_seconds": sec})
                except Exception as e:
                    logger.log(f"[tif-{variant}][{dataset}] ERROR: {e}")
                    all_results.append({"dataset": dataset, "model": f"TIF-{variant}", "error": str(e)})

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    pd.DataFrame(all_results).to_csv(args.out, index=False)
    logger.log(f"[tif] Saved all results -> {args.out}")
    
    if args.grid_search and worst_results:
        worst_out = args.out.replace('.csv', '_worst_params.csv')
        pd.DataFrame(worst_results).to_csv(worst_out, index=False)
        logger.log(f"[tif] Saved worst parameters -> {worst_out}")

if __name__ == "__main__":
    main()
