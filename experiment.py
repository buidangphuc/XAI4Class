# -*- coding: utf-8 -*-
# Auto-generated from experiment.ipynb
# Tip: For multi-GPU training with torchrun, run e.g.:
#   CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 your_script.py

#!/usr/bin/env python
# coding: utf-8

# In[1]:


SEED = 3112
NO_OF_EPOCHS = 20
BATCH_SIZE = 32


# In[2]:


# Import necessary libraries
import os
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from yupi import Trajectory
from pactus import Dataset

# Set display options for better visualization
pd.set_option("display.max_columns", None)
pd.set_option("display.expand_frame_repr", False)

# Define the base directory for all datasets
BASE_DIR = "/home/phucbuidang/Work/XAI/dataset"


# In[3]:


# 1. Import Animals Dataset
# Based on animals.py script


def load_animals_dataset():
    try:
        # Path to the all_data.csv file
        animals_csv_path = os.path.join(BASE_DIR, "animals/train_csv/all_data.csv")

        if os.path.exists(animals_csv_path):
            # Read the CSV file
            df = pd.read_csv(animals_csv_path)

            # Group data by trajectory (index and trajectory_name combination)
            trajs = []
            labels = []

            # Group by unique trajectories
            for (idx, traj_name), group in df.groupby(["index", "trajectory_name"]):
                # Create Trajectory from t, x, y
                traj = Trajectory(
                    x=group["x"].to_numpy(),
                    y=group["y"].to_numpy(),
                    t=group["t"].to_numpy() if "t" in group.columns else None,
                )

                # Add metadata if possible
                if hasattr(traj, "info"):
                    traj.info["index"] = idx
                    traj.info["trajectory_name"] = traj_name
                    traj.info["label"] = group["label"].iloc[0]
                    traj.info["src_filename"] = group["src_filename"].iloc[0]

                trajs.append(traj)
                labels.append(group["label"].iloc[0])

            # Create a pactus Dataset
            animals_ds = Dataset("animals", trajs, labels)
            print(
                f"Animals dataset loaded from CSV: {len(trajs)} trajectories with labels {set(labels)}"
            )
            return animals_ds
        else:
            print(f"Animals CSV file not found at {animals_csv_path}")
            return None

    except Exception as e:
        print(f"Error loading animals dataset: {e}")
        return None


# # Load the animals dataset
# animals_dataset = load_animals_dataset()

# # Display a summary if loaded successfully
# if animals_dataset is not None:
#     print(f"Dataset name: {animals_dataset.name}")
#     print(f"Labels: {set(animals_dataset.labels)}")


# In[4]:


# 2. Import Seabird Dataset
# Based on seabird.py script


def load_seabird_dataset():
    try:
        # Path to the seabird CSV file
        seabird_csv_path = os.path.join(
            BASE_DIR, "seabird/anon_gps_tracks_with_dive.csv"
        )

        if os.path.exists(seabird_csv_path):
            # Read the CSV file
            df = pd.read_csv(seabird_csv_path)

            # Group data by bird
            grouped = df.groupby("bird")

            trajs = []
            labels = []

            for bird_id, group in grouped:
                # Create Trajectory from lat, lon, alt
                traj = Trajectory(
                    x=group["lat"].tolist(),
                    y=group["lon"].tolist(),
                    z=group["alt"].tolist() if "alt" in group.columns else None,
                )
                trajs.append(traj)
                # Use species as label
                labels.append(
                    group["species"].iloc[0]
                    if "species" in group.columns
                    else str(bird_id)
                )

            # Create a pactus Dataset
            seabird_ds = Dataset("seabird", trajs, labels)
            print(
                f"Seabird dataset loaded: {len(trajs)} trajectories with labels {set(labels)}"
            )
            return seabird_ds
        else:
            print(f"Seabird CSV file not found at {seabird_csv_path}")
            return None

    except Exception as e:
        print(f"Error loading seabird dataset: {e}")
        return None


# # Load the seabird dataset
# seabird_dataset = load_seabird_dataset()

# # Display a summary if loaded successfully
# if seabird_dataset is not None:
#     print(f"Dataset name: {seabird_dataset.name}")
#     print(f"Labels: {set(seabird_dataset.labels)}")


# In[5]:


import ast

# 3. Import Taxi Dataset
# Based on the taxi data format which contains polyline coordinates


def load_taxi_dataset():
    try:
        # Path to the taxi data CSV file
        taxi_csv_path = os.path.join(BASE_DIR, "taxi/train.csv")

        if os.path.exists(taxi_csv_path):
            # Read the CSV file with appropriate handling for the POLYLINE column
            df = pd.read_csv(taxi_csv_path)

            trajs = []
            labels = []

            # Process each row to extract trajectory data
            for idx, row in df.iterrows():
                try:
                    # Parse the POLYLINE column which contains trajectory coordinates
                    # The POLYLINE is stored as a string representation of a list of [lon, lat] points
                    polyline = ast.literal_eval(row["POLYLINE"])

                    # Skip if there aren't enough points to form a trajectory
                    if len(polyline) < 2:
                        print(f"Skipping row {idx}: Not enough points in trajectory")
                        continue

                    # Extract x (longitude) and y (latitude) coordinates
                    x = [point[0] for point in polyline]
                    y = [point[1] for point in polyline]

                    # Create trajectory object
                    traj = Trajectory(x=np.array(x), y=np.array(y))

                    # Add metadata if supported
                    if hasattr(traj, "info"):
                        traj.info["trip_id"] = row["TRIP_ID"]
                        traj.info["taxi_id"] = row["TAXI_ID"]
                        traj.info["call_type"] = row["CALL_TYPE"]

                    trajs.append(traj)
                    # Use CALL_TYPE as the label (A, B, C categories)
                    labels.append(row["CALL_TYPE"])

                except Exception as e:
                    print(f"Error processing taxi row {idx}: {e}")
                    continue

            # Create dataset if we have valid trajectories
            if trajs:
                taxi_ds = Dataset("taxi", trajs, labels)
                print(
                    f"Taxi dataset loaded: {len(trajs)} trajectories with labels {set(labels)}"
                )
                return taxi_ds
            else:
                print("No taxi trajectories were loaded.")
                return None
        else:
            print(f"Taxi CSV file not found at {taxi_csv_path}")
            return None

    except Exception as e:
        print(f"Error loading taxi dataset: {e}")
        return None


# # Load the taxi dataset
# taxi_dataset = load_taxi_dataset()

# # Display a summary if loaded successfully
# if taxi_dataset is not None:
#     print(f"Dataset name: {taxi_dataset.name}")
#     print(f"Labels: {set(taxi_dataset.labels)}")


# In[6]:


# 4. Import Vehicle Dataset
# Using similar approach as animals.py for processing vehicle data files


def load_vehicle_dataset():
    try:
        # Directories for vehicle data
        train_dir = os.path.join(BASE_DIR, "vehicle/train")
        test_dir = os.path.join(BASE_DIR, "vehicle/test")

        # Regex pattern to parse filenames like "001 s30902 cB.r2"
        import re

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

        def parse_line(line):
            """Parse one line "t,x y" -> (t, x, y)."""
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

        def parse_filename(name):
            """Parse a filename like "001 s30902 cB.r2" -> (index, trajectory_name, label)."""
            m = FILENAME_RE.match(name)
            if not m:
                return None
            return m.group("index"), m.group("traj"), m.group("label")

        # Collect all trajectory files
        trajs = []
        labels = []

        # Process both train and test directories if they exist
        for data_dir in [train_dir, test_dir]:
            if not os.path.exists(data_dir):
                continue

            for filename in os.listdir(data_dir):
                file_path = os.path.join(data_dir, filename)
                if not os.path.isfile(file_path):
                    continue

                # Parse the filename to extract metadata
                parsed = parse_filename(filename)
                if not parsed:
                    continue

                idx, traj_name, label = parsed

                # Read and parse the trajectory file
                points = []
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        for line in f:
                            point = parse_line(line)
                            if point:
                                points.append(point)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
                    continue

                if not points:
                    continue

                # Extract t, x, y arrays
                t = np.array([p[0] for p in points])
                x = np.array([p[1] for p in points])
                y = np.array([p[2] for p in points])

                # Create trajectory object
                try:
                    traj = Trajectory(x=x, y=y, t=t)
                    # Add metadata if possible
                    if hasattr(traj, "info"):
                        traj.info["index"] = idx
                        traj.info["traj_name"] = traj_name
                    trajs.append(traj)
                    labels.append(label)
                except Exception as e:
                    print(f"Error creating trajectory for {file_path}: {e}")
                    continue

        # Create a pactus Dataset if we have trajectories
        if trajs:
            vehicle_ds = Dataset("vehicle", trajs, labels)
            print(
                f"Vehicle dataset loaded: {len(trajs)} trajectories with labels {set(labels)}"
            )
            return vehicle_ds
        else:
            print("No vehicle trajectories were loaded.")
            return None

    except Exception as e:
        print(f"Error loading vehicle dataset: {e}")
        return None


# Load the vehicle dataset
# vehicle_dataset = load_vehicle_dataset()

# Display a summary if loaded successfully
# if vehicle_dataset is not None:
#     print(f"Dataset name: {vehicle_dataset.name}")
#     print(f"Labels: {set(vehicle_dataset.labels)}")


# In[7]:


from pactus import Dataset

# geolife_dataset = Dataset.geolife()
# animals_dataset = Dataset.animals()
# hurdat2_dataset = Dataset.hurdat2()
# cma_bst_dataset = Dataset.cma_bst()
# mnist_stroke_dataset = Dataset.mnist_stroke()
# uci_pen_digits_dataset = Dataset.uci_pen_digits()
uci_gotrack_dataset = Dataset.uci_gotrack()
# uci_characters_dataset = Dataset.uci_characters()
# uci_movement_libras_dataset = Dataset.uci_movement_libras()
# traffic = Dataset.traffic()


# In[8]:


from pactus.models import LSTMModel


# In[9]:


from models.gru.simple_gru_model import SimpleGRUModel
from models.trajformer.trajformer_model import TrajFormerModel


# In[10]:


all_dataset = [
    # geolife_dataset,
    # animals_dataset,
    # hurdat2_dataset,
    # cma_bst_dataset,
    # mnist_stroke_dataset,
    # uci_pen_digits_dataset,
    uci_gotrack_dataset,
    # uci_characters_dataset,
    # uci_movement_libras_dataset,
    # seabird_dataset,
    # taxi_dataset,
    # vehicle_dataset,
    # traffic,
]


# In[11]:


def create_data(dataset: Dataset):
    # Count samples per class to check if we can use stratified split
    # from collections import Counter

    # label_counts = Counter(dataset.labels)
    # min_samples = min(label_counts.values()) if label_counts else 0

    # # Check if any class has fewer than 2 samples
    # use_stratify = min_samples >= 2

    # if not use_stratify:
    #     print(f"WARNING: Dataset {dataset.name} has classes with fewer than 2 samples.")
    #     print(f"Class distribution: {dict(label_counts)}")
    #     print("Using non-stratified split instead.")

    # # Remove short and poorly time sampled trajectories
    # filtered_dataset = dataset.filter(lambda traj, _: len(traj) > 10 and traj.dt < 8)

    # Split with or without stratification based on our check
    train, test = dataset.split(
        train_size=0.7,
        random_state=SEED,
        # stratify=(min_samples >= 2),  # Only use stratify=True if we have enough samples
    )

    return train, test


# In[ ]:


import io
import re
import time
import contextlib
import traceback


def run_model(train_data: Dataset, test_data: Dataset, dataset: Dataset):
    results = []

    models = [
        ("TrajFormer", TrajFormerModel(c_out=len(dataset.classes))),
        # ("GRU", SimpleGRUModel()),
        # ("LSTM", LSTMModel()),
    ]

    for name, model in models:
        print(f"\n===== Running {name} Experiment =====")

        try:
            # --- Train ---
            model.train(train_data, dataset, epochs=NO_OF_EPOCHS, batch_size=BATCH_SIZE)

            # --- Evaluate (đo thời gian) ---
            t0 = time.perf_counter()
            evaluation = model.evaluate(test_data)
            elapsed = time.perf_counter() - t0

            # --- Capture output từ evaluation.show() ---
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                evaluation.show()
            out = buf.getvalue()

            # --- Parse Accuracy / F1 ---
            acc = re.search(r"Accuracy:\s*([0-9.]+)", out)
            f1 = re.search(r"F1-score:\s*([0-9.]+)", out)
            acc = float(acc.group(1)) if acc else None
            f1 = float(f1.group(1)) if f1 else None

            try:
                n_samples = len(test_data)
            except Exception:
                n_samples = len(getattr(evaluation, "y_true", [])) or None
            throughput = (n_samples / elapsed) if (n_samples and elapsed > 0) else None

            # --- Lưu kết quả ---
            result = {
                "model": name,
                "dataset": dataset.name,
                "accuracy": acc,
                "f1_score": f1,
                "eval_seconds": elapsed,
                "throughput_samples_per_s": throughput,
                "n_samples": n_samples,
            }
            results.append(result)
            print(f"Dataset = {dataset.name}")
            print(f"✅ {name} done:")
            print(
                f"   Accuracy = {acc:.3f}, F1 = {f1:.3f}, Time = {elapsed:.3f}s, Throughput = {throughput:.2f} samples/s"
            )
        except Exception as e:
            print(f"❌ Error running {name} on dataset {dataset.name}:")
            print(f"   {str(e)}")
            traceback.print_exc()
            # Add error record
            results.append(
                {
                    "model": name,
                    "dataset": dataset.name,
                    "error": str(e),
                    "accuracy": None,
                    "f1_score": None,
                }
            )

    return results


# In[ ]:


# run all experiments with all datasets and all models
all_results = []
for dataset in all_dataset:
    if dataset is None:
        print(f"Skipping None dataset")
        continue

    try:
        print(f"\n\n==================================================")
        print(f"Starting experiments for dataset: {dataset.name}")
        print(f"==================================================")

        train_data, test_data = create_data(dataset)
        results = run_model(train_data, test_data, dataset)
        all_results.extend(results)

    except Exception as e:
        print(f"❌ Error processing dataset {dataset.name}:")
        print(f"   {str(e)}")
        import traceback

        traceback.print_exc()

        # Add error record to results
        all_results.append(
            {
                "dataset": dataset.name,
                "error": str(e),
                "accuracy": None,
                "f1_score": None,
            }
        )

print(f"\n\nAll experiments completed. Total results: {len(all_results)}")


# In[ ]:


# Convert results to DataFrame for easy analysis
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Create DataFrame from results
results_df = pd.DataFrame(all_results)

# Filter out error records
valid_results = results_df.dropna(subset=["accuracy"]).copy()

if not valid_results.empty:
    # Display summary table
    print("Summary of experiment results:")
    summary = valid_results.pivot_table(
        index="dataset",
        columns="model",
        values=["accuracy", "f1_score", "eval_seconds"],
        aggfunc="mean",
    )
    display(summary)

    # Plot accuracy comparison
    plt.figure(figsize=(14, 8))
    sns.barplot(x="dataset", y="accuracy", hue="model", data=valid_results)
    plt.title("Model Accuracy by Dataset")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

    # Plot F1-score comparison
    plt.figure(figsize=(14, 8))
    sns.barplot(x="dataset", y="f1_score", hue="model", data=valid_results)
    plt.title("Model F1-Score by Dataset")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

    # Plot evaluation time
    plt.figure(figsize=(14, 8))
    sns.barplot(x="dataset", y="eval_seconds", hue="model", data=valid_results)
    plt.title("Model Evaluation Time by Dataset")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()
else:
    print("No valid results to visualize.")

# Print error summary if there were any errors
errors = results_df[results_df["error"].notna()]
if not errors.empty:
    print("\nErrors encountered:")
    for _, row in errors.iterrows():
        print(
            f"Dataset: {row.get('dataset', 'unknown')}, Model: {row.get('model', 'unknown')}"
        )
        print(f"Error: {row['error']}")
        print("-" * 50)

