# import các thư viện cần thiết
import pandas as pd
from yupi import Trajectory
from pactus import Dataset

# Đọc dữ liệu seabird từ file CSV
df = pd.read_csv('create_data/seabird/anon_gps_tracks_with_dive.csv')

# Nhóm dữ liệu theo từng bird
grouped = df.groupby('bird')

trajs = []
labels = []

for bird_id, group in grouped:
	# Tạo Trajectory từ lat, lon, alt
	traj = Trajectory(
		x=group['lat'].tolist(),
		y=group['lon'].tolist(),
		z=group['alt'].tolist()
	)
	trajs.append(traj)
	# Gán nhãn, ví dụ dùng species
	labels.append(group['species'].iloc[0])

# Tạo Dataset cho pactus
seabird_ds = Dataset("seabird", trajs, labels)
