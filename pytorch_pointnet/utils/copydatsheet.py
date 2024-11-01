import os
import shutil

source_dir = 'shapenet_partanno_v0_final'
destination_dir = 'testpoints'

if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)

for folder_name in os.listdir(source_dir):
    folder_path = os.path.join(source_dir, folder_name)
    if os.path.isdir(folder_path) and folder_name.isdigit():
        points_dir = os.path.join(folder_path, 'points')
        if os.path.exists(points_dir):
            pts_files = [f for f in os.listdir(points_dir) if f.endswith('.pts')]
            if pts_files:
                source_file = os.path.join(points_dir, pts_files[0])
                destination_file = os.path.join(destination_dir, f'{folder_name}.pts')
                shutil.copyfile(source_file, destination_file)