import os
import logging
from typing import Union, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from pactus.dataset import Data
from pactus.models import Model
from pactus.models.evaluation import Evaluation
from sklearn.preprocessing import LabelEncoder
from .trajformer import TrajFormer

NAME = "trajformer_model"


# ====================== DDP helpers =======================
def _is_distributed() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1

def _setup_ddp():
    if _is_distributed():
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def _cleanup_ddp():
    if _is_distributed():
        dist.destroy_process_group()

def _is_main_process() -> bool:
    return (not _is_distributed()) or dist.get_rank() == 0


# ==================== Simple TensorDataset ================
class _TensorTrajDataset(Dataset):
    def __init__(self, x, m, d, y):
        self.x, self.m, self.d, self.y = x, m, d, y
    def __len__(self): return self.x.shape[0]
    def __getitem__(self, i): return self.x[i], self.m[i], self.d[i], self.y[i]


class TrajFormerWrapper(nn.Module):
    """Wrapper cho TrajFormer để đảm bảo vào/ra đúng device (hữu ích cho DP/DDP)."""

    def __init__(self, model, device):
        super().__init__()
        self.model = model.to(device)
        self.device = device

    def forward(self, x_trajs, masks, distances):
        # Move inputs to the correct device
        x_trajs = x_trajs.to(self.device, non_blocking=True)
        masks = masks.to(self.device, non_blocking=True)
        distances = distances.to(self.device, non_blocking=True)
        # Forward pass through the base model
        return self.model(x_trajs, masks, distances)

    def train(self, mode=True):
        self.model.train(mode)
        return self

    def eval(self):
        self.model.eval()
        return self


class TrajFormerModel(Model):
    """Implementation of TrajFormer model for trajectory classification (DDP-ready)."""

    def __init__(
        self,
        c_in=6,
        c_out=4,
        trans_layers=3,
        n_heads=4,
        token_dim=64,
        kv_pool=1,
        mlp_dim=256,
        max_points=100,
        cpe_layers=1,
        metrics=None,
        random_state: Union[int, None] = None,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__(NAME)
        self.c_in = c_in
        self.c_out = c_out
        self.trans_layers = trans_layers
        self.n_heads = n_heads
        self.token_dim = token_dim
        self.kv_pool = kv_pool
        self.mlp_dim = mlp_dim
        self.max_points = max_points
        self.cpe_layers = cpe_layers
        self.metrics = ["accuracy"] if metrics is None else metrics
        self.random_state = random_state
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.encoder = None
        self.labels = None
        self.model = None

        # Set summary for evaluation reporting
        self.set_summary(
            c_in=self.c_in,
            c_out=self.c_out,
            trans_layers=self.trans_layers,
            n_heads=self.n_heads,
            token_dim=self.token_dim,
            kv_pool=self.kv_pool,
            mlp_dim=self.mlp_dim,
            max_points=self.max_points,
            cpe_layers=self.cpe_layers,
            metrics=self.metrics,
        )

    # ---------------------- TRAIN (DDP) ----------------------
    def train(self, data: Data, original_data: Data, training=True, **kwargs):
        _setup_ddp()
        try:
            if self.random_state is not None:
                torch.manual_seed(self.random_state)
                np.random.seed(self.random_state)
                logging.warning(
                    f"Custom seed provided for {self.name} model. This "
                    "sets random seeds for python, numpy, and PyTorch."
                )

            # Init encoder/labels
            self.encoder = LabelEncoder()
            self.labels = data.labels
            y_encoded = self.encoder.fit_transform(self.labels)

            # Device theo LOCAL_RANK nếu DDP
            if _is_distributed():
                local_rank = int(os.environ["LOCAL_RANK"])
                self.device = torch.device(f"cuda:{local_rank}")
            else:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Create base TrajFormer model
            base_model = TrajFormer(
                name=self.name,
                c_in=self.c_in,
                c_out=self.c_out,
                trans_layers=self.trans_layers,
                n_heads=self.n_heads,
                token_dim=self.token_dim,
                kv_pool=self.kv_pool,
                mlp_dim=self.mlp_dim,
                max_points=self.max_points,
                cpe_layers=self.cpe_layers,
                device=str(self.device),
            ).to(self.device)

            # Wrap model (wrapper giữ trách nhiệm move input sang device)
            self.model = TrajFormerWrapper(base_model, self.device)

            # DDP wrap **inner module** nếu nhiều GPU
            if _is_distributed():
                self.model.model = DDP(
                    self.model.model,
                    device_ids=[self.device.index],
                    output_device=self.device.index,
                    find_unused_parameters=False,
                )

            # Chuẩn bị dữ liệu (tensors trên CPU; wrapper sẽ .to(device) khi forward)
            x_trajs, masks, distances = self._prepare_data(data)
            y_tensor = torch.tensor(y_encoded, dtype=torch.long)

            # DataLoader + DistributedSampler
            ds = _TensorTrajDataset(x_trajs, masks, distances, y_tensor)
            if _is_distributed():
                sampler = DistributedSampler(ds, shuffle=True, drop_last=False)
                shuffle = False
            else:
                sampler = None
                shuffle = True

            batch_size = kwargs.get("batch_size", 64)  # batch MỖI GPU
            loader = DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=shuffle,
                sampler=sampler,
                num_workers=4,
                pin_memory=True,
                persistent_workers=False,
            )

            # Optim/loss
            optimizer = torch.optim.Adam(self.model.parameters(), lr=kwargs.get("learning_rate", 0.001))
            criterion = nn.CrossEntropyLoss().to(self.device)

            n_epochs = kwargs.get("n_epochs", 20)
            self.model.train()

            for epoch in range(n_epochs):
                if _is_distributed():
                    sampler.set_epoch(epoch)  # đảm bảo shuffling khác nhau giữa epoch

                total_loss = 0.0
                for bx, bm, bd, by in loader:
                    by = by.to(self.device, non_blocking=True)

                    optimizer.zero_grad(set_to_none=True)
                    outputs = self.model(bx, bm, bd)
                    loss = criterion(outputs, by)
                    loss.backward()
                    optimizer.step()

                    total_loss += float(loss.detach().cpu())

                if _is_main_process() and ((epoch + 1) % 2 == 0 or epoch == n_epochs - 1):
                    logging.info(f"Epoch {epoch+1}/{n_epochs}, Loss: {total_loss:.4f}")

            if _is_main_process():
                logging.info(f"Trained TrajFormer model (DDP) with {len(self.labels)} samples")

        finally:
            _cleanup_ddp()

    # ---------------------- PREDICT ----------------------
    def predict(self, data: Data) -> np.ndarray:
        """
        Predict class probabilities for each trajectory

        Returns:
            Array of shape (n_samples, n_classes) with class probabilities
        """
        assert self.model is not None, "Model has not been trained yet"
        assert self.encoder is not None, "Encoder is not initialized"

        # Prepare data
        x_trajs, masks, distances = self._prepare_data(data)

        # Get predictions
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(x_trajs, masks, distances)
            probabilities = F.softmax(outputs, dim=1).cpu().numpy()

        return probabilities

    def evaluate(self, data: Data) -> Evaluation:
        """Evaluate the model on test data"""
        assert self.encoder is not None, "Encoder is not set."

        # Get predicted probabilities
        probabilities = self.predict(data)

        # Get class with highest probability for each sample
        pred_indices = np.argmax(probabilities, axis=1)

        # Convert back to original class labels
        predictions = self.encoder.inverse_transform(pred_indices)

        return Evaluation.from_data(data, predictions, self.summary)

    # ---------------------- DATA PIPE ----------------------
    def _prepare_data(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert data from pactus format to TrajFormer input format

        Returns:
            x_trajs: [B, max_points, c_in] float32
            masks:   [B, max_points] bool (True = PAD)
            distances: [B, max_points, 9, 2] float32
        """
        all_features: List[np.ndarray] = []
        all_masks: List[torch.Tensor] = []
        all_distances: List[np.ndarray] = []

        kernel_size = 9

        for traj in data.trajs:
            # Extract coordinates, time, etc.
            coords = np.asarray(traj.r, dtype=np.float32)   # [N, 2]
            times = np.asarray(traj.t, dtype=np.float32) if hasattr(traj, "t") else None

            # Calculate features (speeds, accelerations, etc.)
            feats = self._extract_features(coords, times)   # [N, c_in]
            n = feats.shape[0]
            n_clip = min(n, self.max_points)

            # Pad/truncate features to max_points
            x = np.zeros((self.max_points, self.c_in), dtype=np.float32)
            if n_clip > 0:
                x[:n_clip] = feats[:n_clip]

            # Create mask (False for data, True for padding)
            mask = torch.ones(self.max_points, dtype=torch.bool)
            mask[:n_clip] = False

            # Calculate distance matrix for CPE (truncate to n_clip, then pad to max_points)
            dist_matrix = self._calculate_distances(coords[:n_clip])
            if dist_matrix.shape[0] < self.max_points:
                pad = np.zeros((self.max_points - dist_matrix.shape[0], kernel_size, 2), dtype=np.float32)
                dist_matrix = np.concatenate([dist_matrix, pad], axis=0)

            all_features.append(x)
            all_masks.append(mask)
            all_distances.append(dist_matrix)

        # Convert to tensors
        return (
            torch.from_numpy(np.stack(all_features, axis=0)),        # [B, T, C]
            torch.stack(all_masks, dim=0),                           # [B, T]
            torch.from_numpy(np.stack(all_distances, axis=0)),       # [B, T, 9, 2]
        )

    def _extract_features(self, coords: np.ndarray, times: Union[np.ndarray, None]):
        """Extract features from trajectory coordinates and times: [lat, lng, dt, dd, speed, accel]."""
        n_points = coords.shape[0]
        features = np.zeros((n_points, self.c_in), dtype=np.float32)

        if n_points == 0:
            return features

        # lat, lng
        features[:, 0] = coords[:, 0]
        features[:, 1] = coords[:, 1]

        # Calculate time differences
        if n_points > 1 and times is not None and times.shape[0] == n_points:
            dt = np.diff(times, prepend=times[0]).astype(np.float32)
        else:
            dt = np.ones(n_points, dtype=np.float32)
        features[:, 2] = dt  # delta time

        # Calculate distances between consecutive points
        if n_points > 1:
            dx = np.diff(coords[:, 0], prepend=coords[0, 0])
            dy = np.diff(coords[:, 1], prepend=coords[0, 1])
            dd = np.sqrt(dx**2 + dy**2).astype(np.float32)
        else:
            dd = np.zeros(n_points, dtype=np.float32)
        features[:, 3] = dd  # delta distance

        # Speeds
        with np.errstate(divide="ignore", invalid="ignore"):
            speeds = np.zeros_like(dt, dtype=np.float32)
            valid_dt = dt > 0
            speeds[valid_dt] = dd[valid_dt] / dt[valid_dt]
        features[:, 4] = speeds  # speed

        # Accelerations
        accels = np.diff(speeds, prepend=speeds[0]).astype(np.float32)
        features[:, 5] = accels  # acceleration

        return features

    def _calculate_distances(self, coords: np.ndarray):
        """Calculate distance matrix for CPE module. Returns [T, 9, 2] padded later to [max_points, 9, 2]."""
        n_points = min(coords.shape[0], self.max_points)
        kernel_size = 9
        distances = np.zeros((n_points, kernel_size, 2), dtype=np.float32)

        half_k = kernel_size // 2
        for i in range(n_points):
            for j in range(kernel_size):
                idx = i - half_k + j
                if 0 <= idx < n_points:
                    # delta lat/lng
                    distances[i, j, 0] = coords[idx, 0] - coords[i, 0]
                    distances[i, j, 1] = coords[idx, 1] - coords[i, 1]

        return distances
