# create_dataset.py
from __future__ import annotations

import gzip
from datetime import datetime, timezone
from collections import defaultdict
from typing import Iterable, Iterator, Tuple, Dict, List, Any, Optional
import inspect

import numpy as np
from yupi import Trajectory
from pactus import Dataset


# ---------- 1) Đọc & parse Gowalla ----------
def parse_gowalla_checkins(path_gz: str) -> Iterator[Tuple[int, datetime, float, float, int]]:
    with gzip.open(path_gz, "rt", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            user = int(parts[0])
            ts = datetime.fromisoformat(parts[1].replace("Z", "+00:00"))
            lat = float(parts[2])
            lon = float(parts[3])
            loc = int(parts[4])
            yield user, ts, lat, lon, loc


# ---------- 2) Gom theo user ----------
def group_by_user(
    records: Iterable[Tuple[int, datetime, float, float, int]],
) -> Dict[int, List[Tuple[datetime, float, float, int]]]:
    buckets: Dict[int, List[Tuple[datetime, float, float, int]]] = defaultdict(list)
    for user, ts, lat, lon, loc in records:
        buckets[user].append((ts, lat, lon, loc))
    for user in buckets:
        buckets[user].sort(key=lambda x: x[0])
    return buckets


# ---------- 3) (tuỳ chọn) session hóa ----------
def split_sessions(
    items: List[Tuple[datetime, float, float, int]],
    gap_hours: float = 6.0,
) -> List[List[Tuple[datetime, float, float, int]]]:
    if not items:
        return []
    sessions: List[List[Tuple[datetime, float, float, int]]] = []
    cur: List[Tuple[datetime, float, float, int]] = [items[0]]
    gap = gap_hours * 3600.0
    for i in range(1, len(items)):
        prev_ts = items[i - 1][0].replace(tzinfo=timezone.utc).timestamp()
        cur_ts = items[i][0].replace(tzinfo=timezone.utc).timestamp()
        if (cur_ts - prev_ts) > gap:
            sessions.append(cur)
            cur = []
        cur.append(items[i])
    if cur:
        sessions.append(cur)
    return sessions


# ---------- 4) Build Trajectory (tương thích bản yupi cũ) ----------
def build_trajectory_from_items(
    items: List[Tuple[datetime, float, float, int]],
    traj_id: str,
) -> Optional[Trajectory]:
    if len(items) < 2:
        return None

    t = np.array([i[0].replace(tzinfo=timezone.utc).timestamp() for i in items], dtype=float)
    lat = np.array([i[1] for i in items], dtype=float)
    lon = np.array([i[2] for i in items], dtype=float)
    loc = np.array([i[3] for i in items], dtype=np.int64)

    has_extra = 'extra' in inspect.signature(Trajectory.__init__).parameters

    if has_extra:
        # Bản yupi có hỗ trợ 'extra'
        traj = Trajectory(x=lon, y=lat, t=t, traj_id=traj_id, extra={"loc_id": loc})
    else:
        # Bản yupi cũ: chỉ truyền các tham số chuẩn
        traj = Trajectory(x=lon, y=lat, t=t, traj_id=traj_id)
        # Nếu có thuộc tính metadata thì nhét loc_id vào metadata (không bắt buộc)
        if hasattr(traj, "metadata") and isinstance(traj.metadata, dict):
            traj.metadata["loc_id"] = loc
    return traj


# ---------- 5) Downsample đều ----------
def downsample_evenly(items: List[Any], k: int) -> List[Any]:
    if len(items) <= k:
        return items
    idx = np.linspace(0, len(items) - 1, k).astype(int)
    return [items[i] for i in idx]


# ---------- 6) Build Dataset ----------
class BuildOptions:
    def __init__(
        self,
        min_points: int = 10,
        max_points: Optional[int] = None,
        sessionize: bool = False,
        gap_hours: float = 6.0,
        max_users: Optional[int] = None,
        name: str = "gowalla",
    ):
        self.min_points = min_points
        self.max_points = max_points
        self.sessionize = sessionize
        self.gap_hours = gap_hours
        self.max_users = max_users
        self.name = name


def build_pactus_dataset_from_gowalla(path_gz: str, opt: BuildOptions = BuildOptions()) -> Dataset:
    buckets = group_by_user(parse_gowalla_checkins(path_gz))

    users = list(buckets.keys())
    if opt.max_users is not None:
        users = users[: opt.max_users]

    trajs: List[Trajectory] = []
    labels: List[str] = []

    for user in users:
        items = buckets[user]
        sequences = split_sessions(items, opt.gap_hours) if opt.sessionize else [items]

        for si, seq in enumerate(sequences):
            if len(seq) < max(2, opt.min_points):
                continue
            if opt.max_points is not None and len(seq) > opt.max_points:
                seq = downsample_evenly(seq, opt.max_points)

            traj_id = f"{user}" if not opt.sessionize else f"{user}_s{si}"
            traj = build_trajectory_from_items(seq, traj_id=traj_id)
            if traj is None:
                continue

            trajs.append(traj)
            labels.append(str(user))  # <-- dùng user_id làm label

    ds = Dataset(opt.name, trajs, labels)
    return ds


# ---------- 7) Main ----------
if __name__ == "__main__":
    # Ví dụ cấu hình
    opts = BuildOptions(
        min_points=20,
        max_points=200,
        sessionize=True,   # tách theo phiên dựa trên gap giờ
        gap_hours=6.0,
        max_users=500,     # chọn bớt user để chạy nhanh
        name="gowalla_session_user_label",
    )
    ds = build_pactus_dataset_from_gowalla("loc-gowalla_totalCheckins.txt.gz", opt=opts)