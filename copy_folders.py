import shutil
import os

# List of subfolders to copy
subfolders = ['10504_0010504_t107_i002_0010504_t120_i002',
'10504_0010504_t107_i004_0010504_t120_i002',
'105504_0105504_t107_i002_0105504_t107_i001',
'105504_0105504_t141_i001_0105504_t107_i001',
'105504_0105504_t141_i005_0105504_t107_i005',
'105504_0105504_t141_i006_0105504_t107_i005',
'105504_0105504_t145_i001_0105504_t107_i001',
'105504_0105504_t145_i005_0105504_t107_i005',
'1071_0001071_t126_i003_0001071_t114_i003',
'10726_0010726_t127_i002_0010726_t107_i002',
'10726_0010726_t127_i003_0010726_t107_i003',
'10726_0010726_t127_i006_0010726_t107_i006',
'10726_0010726_t140_i002_0010726_t107_i002',
'10726_0010726_t140_i003_0010726_t107_i003',
'10726_0010726_t140_i006_0010726_t107_i006',
'10726_0010726_t140_i007_0010726_t107_i007',
'10751_0010751_t139_i002_0010751_t151_i002',
'10751_0010751_t139_i006_0010751_t151_i006',
'10751_0010751_t139_i007_0010751_t151_i007',
'10791_0010791_t108_i002_0010791_t128_i002',
'10791_0010791_t108_i003_0010791_t128_i003',
'108063_0108063_t159_i002_0108063_t153_i001',
'10831_0010831_t130_i006_0010831_t108_i006',
'109856_0109856_t138_i004_0109856_t109_i002',
'109856_0109856_t138_i006_0109856_t109_i006',
'109856_0109856_t149_i008_0109856_t109_i006',
'109_0000109_t110_i007_0000109_t139_i006',
'11055_0011055_t133_i002_0011055_t137_i001',
'11418_0011418_t130_i006_0011418_t115_i006',
'11630_0011630_t128_i002_0011630_t115_i002']  

# Source and destination directories
source_dir = '/home/borji/TechTeamDrive/tmp_borji/annotation_data'
destination_dir = '/home/borji/TechTeamDrive/tmp_borji/annotation_data_good_subset'

# Create the destination directory if it doesn't exist
os.makedirs(destination_dir, exist_ok=True)

# Copy each subfolder
for subfolder in subfolders:
    src_path = os.path.join(source_dir, subfolder)
    dest_path = os.path.join(destination_dir, subfolder)
    
    if os.path.exists(src_path):
        shutil.copytree(src_path, dest_path)
        print(f"Copied {src_path} to {dest_path}")
    else:
        print(f"{src_path} does not exist and was skipped.")
