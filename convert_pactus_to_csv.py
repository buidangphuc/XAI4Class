# convert_pactus_to_csv.py
# Read one or many pactus datasets, convert to df(tid,class,c1,c2,t) and save CSVs.

import os
import argparse
from typing import List

from utils import (
    Logger,
    set_seed,
    get_pactus_dataset_by_name,
    pactus_to_df,
    save_df_to_csv,
    list_available_dataset_names,
)

def main():
    parser = argparse.ArgumentParser(description="Convert pactus datasets to CSV format")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["all"],
        help=(
            "List of dataset names to convert. Use 'all' (default) to export every "
            "available pactus built-in and custom dataset."
        ),
    )
    parser.add_argument(
        "--csv-dir",
        default="./data_csv",
        help="Directory where CSV files will be stored",
    )
    parser.add_argument("--seed", type=int, default=3112)
    parser.add_argument(
        "--exclude-custom",
        action="store_true",
        help="When used with 'all', skip custom datasets (animals_local, seabird_local, ...)",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    logger = Logger(out_dir="./results")
    os.makedirs(args.csv_dir, exist_ok=True)

    dataset_names: List[str]
    if len(args.datasets) == 1 and args.datasets[0].lower() == "all":
        dataset_names = list_available_dataset_names(include_custom=not args.exclude_custom)
        logger.log(
            "[convert] 'all' selected -> exporting {count} datasets ({names})".format(
                count=len(dataset_names),
                names=", ".join(dataset_names),
            )
        )
    else:
        dataset_names = args.datasets
        logger.log(f"[convert] datasets={dataset_names}")

    dataset_names = [n.strip() for n in dataset_names if n.strip()]
    if not dataset_names:
        logger.log("[convert] No datasets specified after filtering; nothing to do.")
        return

    for name in dataset_names:
        try:
            ds = get_pactus_dataset_by_name(name)
            df = pactus_to_df(ds)
            if df.empty:
                logger.log(f"[convert][{name}] WARN: empty dataframe, skipped.")
                continue
            csv_path = save_df_to_csv(df, args.csv_dir, name)
            logger.log(f"[convert][{name}] saved -> {csv_path} (rows={len(df)})")
        except Exception as e:
            logger.log(f"[convert][{name}] ERROR: {e}")

if __name__ == "__main__":
    main()
