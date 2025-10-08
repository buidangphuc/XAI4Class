import pandas as pd
from yupi import Trajectory
from pactus import Dataset
import ast

# Đường dẫn tới file CSV taxi
csv_path = 'create_data/taxi/train.csv'

df = pd.read_csv(csv_path)

trajs = []
labels = []

for _, row in df.iterrows():
    polyline = ast.literal_eval(row['POLYLINE'])
    x = [p[0] for p in polyline]
    y = [p[1] for p in polyline]
    trajs.append(Trajectory(x=x, y=y))
    labels.append(row['CALL_TYPE'])

# Tạo dataset pactus
taxi_ds = Dataset("taxi", trajs, labels)

print(taxi_ds)
