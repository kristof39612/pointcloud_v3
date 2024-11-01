import os
import sys
import shutil
import subprocess

testpoints_dir = 'testpoints'
renamed_dir = 'renamed_testpoints'
infer_script = 'infer.py'
model_path = 'output_cls/shapenet_classification_model.pth'
model_type = 'classification'

if not os.path.exists(renamed_dir):
    os.makedirs(renamed_dir)

pts_files = [f for f in os.listdir(testpoints_dir) if f.endswith('.pts')]

for pts_file in pts_files:
    pts_path = os.path.join(testpoints_dir, pts_file)
    
    subprocess.call([sys.executable, infer_script, 'shapenet', model_path, pts_path, model_type])

    new_name = input(f'Enter new filename for {pts_file} (without extension): ')
    new_name += '.pts'
    new_path = os.path.join(renamed_dir, new_name)
    shutil.move(pts_path, new_path)