# build_yupi_dataset.py
# -*- coding: utf-8 -*-

"""
Convert raw trajectory files into canonical CSVs and (optionally) a yupi dataset.

Raw file line format:
    0,50.1066 3.79665
    4.39,50.1045 3.79455
… i.e. "t,x y".

Filename format:
    "01 s1 cD.r2"
where:
    01 -> index
    s1 -> trajectory name
    cD -> label
    r2 -> run/rep (ignored)

Output:
- CSVs with header: t,x,y
- index.json + index.csv listing metadata
- dataset.pkl (list of yupi.Trajectory) if --save_yupi is set
"""

import argparse, os, re, json, pickle
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional

import pandas as pd


def try_import_yupi():
    try:
        import yupi
        return yupi
    except Exception:
        return None


# Regex to parse filenames like "01 s1 cD.r2"
FILENAME_RE = re.compile(
    r"""^\s*
    (?P<index>\d+)
    \s+
    (?P<traj>s\d+)
    \s+
    (?P<label>[A-Za-z]+)
    \.r\d+
    \s*$""",
    re.VERBOSE,
)


@dataclass
class ItemMeta:
    index: str
    trajectory_name: str
    label: str
    src_filename: str
    csv_path: str


def parse_line(line: str) -> Optional[Tuple[float, float, float]]:
    """
    Parse one line "t,x y" -> (t, x, y).
    """
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


def parse_filename(name: str) -> Optional[Tuple[str, str, str]]:
    """
    Parse a filename like "01 s1 cD.r2" -> (index, trajectory_name, label).
    """
    m = FILENAME_RE.match(name)
    if not m:
        return None
    return m.group("index"), m.group("traj"), m.group("label")


def convert_file(src_path: str, out_dir: str) -> Optional[ItemMeta]:
    """
    Convert a raw file into CSV with t,x,y.
    """
    base = os.path.basename(src_path)
    parsed = parse_filename(base) or parse_filename(os.path.splitext(base)[0])
    if not parsed:
        return None
    idx, traj, label = parsed

    rows: List[Tuple[float, float, float]] = []
    with open(src_path, "r", encoding="utf-8") as f:
        for line in f:
            triple = parse_line(line)
            if triple is not None:
                rows.append(triple)

    if not rows:
        return None

    df = pd.DataFrame(rows, columns=["t", "x", "y"])

    subdir = os.path.join(out_dir, label, traj)
    os.makedirs(subdir, exist_ok=True)
    out_csv = os.path.join(subdir, f"{idx}_{traj}_{label}.csv")
    df.to_csv(out_csv, index=False)

    return ItemMeta(
        index=idx,
        trajectory_name=traj,
        label=label,
        src_filename=base,
        csv_path=os.path.relpath(out_csv, out_dir),
    )


def build_yupi_objects(index_items: List[ItemMeta], out_dir: str) -> str:
    yupi = try_import_yupi()
    if yupi is None:
        raise RuntimeError("yupi not installed. Run `pip install yupi`.")

    trajs = []
    for it in index_items:
        csv_abs = os.path.join(out_dir, it.csv_path)
        df = pd.read_csv(csv_abs)
        T = df["t"].to_numpy()
        X = df["x"].to_numpy()
        Y = df["y"].to_numpy()

        tr = yupi.Trajectory(x=X, y=Y, t=T)
        # Set name attribute if available
        if hasattr(tr, 'name'):
            tr.name = it.trajectory_name
        try:
            tr.info["index"] = it.index
            tr.info["label"] = it.label
            tr.info["src_filename"] = it.src_filename
        except Exception:
            pass
        trajs.append(tr)

    out_pkl = os.path.join(out_dir, "dataset.pkl")
    with open(out_pkl, "wb") as f:
        pickle.dump(trajs, f)
    return out_pkl


def main():
    # Đường dẫn dữ liệu
    train_dir = "/Users/phuc.buidang/Documents/UIT/XAI/dataset/animals/train"
    test_dir = "/Users/phuc.buidang/Documents/UIT/XAI/dataset/animals/test"
    out_dir = "/Users/phuc.buidang/Documents/UIT/XAI/dataset/animals/train_csv"
    save_yupi = True

    os.makedirs(out_dir, exist_ok=True)

    # Lấy danh sách file từ cả train và test
    candidates = []
    for data_dir in [train_dir, test_dir]:
        for entry in os.listdir(data_dir):
            full = os.path.join(data_dir, entry)
            if os.path.isfile(full) and (parse_filename(entry) or parse_filename(os.path.splitext(entry)[0])):
                candidates.append(full)

    index_items: List[ItemMeta] = []
    all_rows = []
    for path in sorted(candidates):
        meta = convert_file(path, out_dir)
        if meta:
            index_items.append(meta)
            # Đọc lại dữ liệu vừa convert để gộp
            df = pd.read_csv(os.path.join(out_dir, meta.csv_path))
            df["index"] = meta.index
            df["trajectory_name"] = meta.trajectory_name
            df["label"] = meta.label
            df["src_filename"] = meta.src_filename
            all_rows.append(df)

    # Gộp toàn bộ dữ liệu vào một file CSV duy nhất
    if all_rows:
        merged_df = pd.concat(all_rows, ignore_index=True)
        merged_df.to_csv(os.path.join(out_dir, "all_data.csv"), index=False)

    # Save index.json and index.csv
    with open(os.path.join(out_dir, "index.json"), "w", encoding="utf-8") as f:
        json.dump([asdict(it) for it in index_items], f, ensure_ascii=False, indent=2)
    pd.DataFrame([asdict(it) for it in index_items]).to_csv(os.path.join(out_dir, "index.csv"), index=False)


    if save_yupi:
        try:
            out_pkl = build_yupi_objects(index_items, out_dir)
            print("Saved yupi dataset:", out_pkl)
        except Exception as e:
            print("Could not save yupi dataset:", e)

    # Tạo dataset kiểu pactus
    try:
        from yupi import Trajectory
        from pactus import Dataset
    except ImportError:
        print("Bạn cần cài đặt yupi và pactus để tạo dataset kiểu pactus.")
        return

    trajs = []
    labels = []
    for it in index_items:
        csv_abs = os.path.join(out_dir, it.csv_path)
        df = pd.read_csv(csv_abs)
        T = df["t"].to_numpy()
        X = df["x"].to_numpy()
        Y = df["y"].to_numpy()
        tr = Trajectory(x=X, y=Y, t=T)
        trajs.append(tr)
        labels.append(it.label)

    ds = Dataset("animals", trajs, labels)
    print("Đã tạo dataset kiểu pactus:", ds)


if __name__ == "__main__":
    main()
