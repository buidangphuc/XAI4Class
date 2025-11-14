# -*- coding: utf-8 -*-
# Experiment with RandomForestModel and UniversalFeaturizer

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
import time
from .models import TrajFormerModel

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


# Load the animals dataset
# animals_local_dataset = load_animals_dataset()

# # Display a summary if loaded successfully
# if animals_local_dataset is not None:
#     print(f"Dataset name: {animals_local_dataset.name}")
#     print(f"Labels: {set(animals_local_dataset.labels)}")


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


# Load the seabird dataset
seabird_dataset = load_seabird_dataset()

# Display a summary if loaded successfully
if seabird_dataset is not None:
    print(f"Dataset name: {seabird_dataset.name}")
    print(f"Labels: {set(seabird_dataset.labels)}")


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
vehicle_dataset = load_vehicle_dataset()

# Display a summary if loaded successfully
if vehicle_dataset is not None:
    print(f"Dataset name: {vehicle_dataset.name}")
    print(f"Labels: {set(vehicle_dataset.labels)}")


# In[7]:


from pactus import Dataset

geolife_dataset = Dataset.geolife()
animals_dataset = Dataset.animals()
# hurdat2_dataset = Dataset.hurdat2()
# cma_bst_dataset = Dataset.cma_bst()
# mnist_stroke_dataset = Dataset.mnist_stroke()
# uci_pen_digits_dataset = Dataset.uci_pen_digits()
uci_gotrack_dataset = Dataset.uci_gotrack()
# uci_characters_dataset = Dataset.uci_characters()
uci_movement_libras_dataset = Dataset.uci_movement_libras()
traffic_dataset = Dataset.traffic()


# In[8]:


# Import RandomForestModel and featurizers
from pactus.models import RandomForestModel
from pactus import featurizers


# In[9]:


all_dataset = [
    animals_dataset,
    # hurdat2_dataset,
    # animals_local_dataset,
    # cma_bst_dataset,
    # mnist_stroke_dataset,
    # uci_pen_digits_dataset,
    uci_gotrack_dataset,
    # uci_characters_dataset,
    uci_movement_libras_dataset,
    seabird_dataset,
    # taxi_dataset,
    vehicle_dataset,
    traffic_dataset,
    geolife_dataset,
]


# In[10]:


def create_data(dataset: Dataset):
    # Define the classes to use (you can modify this list based on your needs)
    use_classes = set(dataset.labels)  # Use all available classes by default
    
    # Create a processing pipeline with filters
    train, test = (
        dataset
        # Remove short and poorly time sampled trajectories
        .filter(lambda traj, _: len(traj) > 10 and traj.dt < 8)
        # Join "taxi" and "bus" into "taxi-bus"
        # .map(lambda _, label: (_, "taxi-bus" if label in ("bus", "taxi") else label))
        # Only use the classes defined in use_classes
        # .filter(lambda _, label: label in use_classes)
        # Split the dataset into train and test
        .split(train_size=0.7, random_state=SEED)
    )

    return train, test


def run_model(train_data: Dataset, test_data: Dataset, dataset: Dataset):
    """
    Run experiments with different models on the given dataset.
    
    Args:
        train_data (Dataset): Training dataset
        test_data (Dataset): Test dataset  
        dataset (Dataset): Original dataset for metadata
        
    Returns:
        list: List of experiment results
    """
    results = []

    models = [
        ("TrajFormer", TrajFormerModel(c_out=len(dataset.classes))),
        # ("GRU", SimpleGRUModel()),
        # ("LSTM", LSTMModel(
        #     optimizer="rmsprop",
        #     loss="sparse_categorical_crossentropy",
        #     metrics=["accuracy"],
        # )),
    ]

    for name, model in models:
        # logger.log(f"\n===== Running {name} Experiment =====")

        try:
            # Additional GPU memory clearing for TensorFlow models
            if name in ["LSTM"] and tf.config.list_physical_devices('GPU'):
                tf.keras.backend.clear_session()
            
            # Train the model
            model.train(train_data, dataset, epochs=NO_OF_EPOCHS, batch_size=BATCH_SIZE)

            # Evaluate the model (measure time)
            t0 = time.perf_counter()
            evaluation = model.evaluate(test_data)
            elapsed = time.perf_counter() - t0

            # Capture output from evaluation.show()
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                evaluation.show()
            out = buf.getvalue()

            # Parse Accuracy / F1
            acc = re.search(r"Accuracy:\s*([0-9.]+)", out)
            f1 = re.search(r"F1-score:\s*([0-9.]+)", out)
            acc = float(acc.group(1)) if acc else None
            f1 = float(f1.group(1)) if f1 else None

            try:
                n_samples = len(test_data)
            except Exception:
                n_samples = len(getattr(evaluation, "y_true", [])) or None
            
            throughput = (n_samples / elapsed) if (n_samples and elapsed > 0) else None

            # Save results
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
            
            logger.log(f"Dataset = {dataset.name}")
            logger.log(f"✅ {name} done:")
            logger.log(f"   Accuracy = {acc:.3f}, F1 = {f1:.3f}, Time = {elapsed:.3f}s, Throughput = {throughput:.2f} samples/s")
            
        except Exception as e:
            error_msg = str(e)
            logger.log(f"❌ Error running {name} on dataset {dataset.name}:")
            logger.log(f"   {error_msg}")
            
            # Check if it's a GPU-related error and suggest fallback
            if any(keyword in error_msg.lower() for keyword in ['gpu', 'cuda', 'jit', 'device']):
                logger.log(f"   GPU-related error detected. Consider running with CPU only.")
                logger.log(f"   You can set: os.environ['CUDA_VISIBLE_DEVICES'] = ''")
            
            traceback.print_exc()
            
            # Clear GPU memory after error for TensorFlow models
            if name in ["LSTM"] and tf.config.list_physical_devices('GPU'):
                try:
                    tf.keras.backend.clear_session()
                except:
                    pass
            
            # Add error record
            results.append({
                "model": name,
                "dataset": dataset.name,
                "error": error_msg,
                "accuracy": None,
                "f1_score": None,
                "eval_seconds": None,
                "throughput_samples_per_s": None,
                "n_samples": None,
            })

    return results


# =============================================================================
# MAIN EXPERIMENT EXECUTION
# =============================================================================

# Run all experiments with all datasets and all models
all_results = []

for dataset in all_dataset:
    if dataset is None:
        logger.log(f"Skipping None dataset")
        continue

    try:
        logger.log(f"\n\n" + "=" * 50)
        logger.log(f"Starting experiments for dataset: {dataset.name}")
        logger.log("=" * 50)

        train_data, test_data = create_data(dataset)
        results = run_model(train_data, test_data, dataset)
        all_results.extend(results)

    except Exception as e:
        logger.log(f"❌ Error processing dataset {dataset.name}:")
        logger.log(f"   {str(e)}")
        
        # Add error record to results
        all_results.append({
            "dataset": dataset.name,
            "error": str(e),
            "accuracy": None,
            "f1_score": None,
        })

logger.log(f"\n\nAll experiments completed. Total results: {len(all_results)}")

# =============================================================================
# RESULTS ANALYSIS AND VISUALIZATION
# =============================================================================

# Convert results to DataFrame for easy analysis
results_df = pd.DataFrame(all_results)

# Save raw results to CSV
results_csv_path = os.path.join(OUTPUT_DIR, f"experiment_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
results_df.to_csv(results_csv_path, index=False)
logger.log(f"Raw results saved to: {results_csv_path}")

# Filter out error records
valid_results = results_df.dropna(subset=["accuracy"]).copy()

if not valid_results.empty:
    # Display summary table
    logger.log("Summary of experiment results:")
    summary = valid_results.pivot_table(
        index="dataset",
        columns="model",
        values=["accuracy", "f1_score", "eval_seconds"],
        aggfunc="mean",
    )
    
    # Save summary to CSV
    summary_csv_path = os.path.join(OUTPUT_DIR, f"experiment_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    summary.to_csv(summary_csv_path)
    logger.log(f"Summary table saved to: {summary_csv_path}")
    
    # Log summary to text file
    with open(os.path.join(OUTPUT_DIR, f"summary_table_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"), 'w') as f:
        f.write("EXPERIMENT SUMMARY\n")
        f.write("==================\n\n")
        f.write(str(summary))

    # Plot accuracy comparison
    plt.figure(figsize=(14, 8))
    sns.barplot(x="dataset", y="accuracy", hue="model", data=valid_results)
    plt.title("Model Accuracy by Dataset")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    accuracy_plot_path = os.path.join(OUTPUT_DIR, f"accuracy_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(accuracy_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.log(f"Accuracy plot saved to: {accuracy_plot_path}")

    # Plot F1-score comparison
    plt.figure(figsize=(14, 8))
    sns.barplot(x="dataset", y="f1_score", hue="model", data=valid_results)
    plt.title("Model F1-Score by Dataset")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    f1_plot_path = os.path.join(OUTPUT_DIR, f"f1_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(f1_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.log(f"F1-score plot saved to: {f1_plot_path}")

    # Plot evaluation time
    plt.figure(figsize=(14, 8))
    sns.barplot(x="dataset", y="eval_seconds", hue="model", data=valid_results)
    plt.title("Model Evaluation Time by Dataset")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    time_plot_path = os.path.join(OUTPUT_DIR, f"evaluation_time_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(time_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.log(f"Evaluation time plot saved to: {time_plot_path}")

else:
    logger.log("No valid results to visualize.")

# Print error summary if there were any errors
errors = results_df[results_df["error"].notna()]
if not errors.empty:
    logger.log("\nErrors encountered:")
    error_summary_path = os.path.join(OUTPUT_DIR, f"error_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    with open(error_summary_path, 'w') as f:
        f.write("ERROR SUMMARY\n")
        f.write("=============\n\n")
        for _, row in errors.iterrows():
            error_msg = f"Dataset: {row.get('dataset', 'unknown')}, Model: {row.get('model', 'unknown')}\nError: {row['error']}\n" + "-" * 50 + "\n"
            logger.log(error_msg.strip())
            f.write(error_msg)
    
    logger.log(f"Error summary saved to: {error_summary_path}")

logger.log(f"\nAll results and visualizations saved to: {OUTPUT_DIR}")

