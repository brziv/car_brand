import os
import shutil

# Read color labels
with open('../annot/cl_list.txt', 'r') as f:
    colors = [line.strip() for line in f.readlines()]

# Create folders for each color
os.makedirs('../colors_ps', exist_ok=True)
for color in colors:
    os.makedirs(f'../colors_ps/{color}', exist_ok=True)

# Read full.txt and organize images
with open('../annot/result.txt', 'r') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 2:
            img_name = parts[0]  # image name
            color_label = int(parts[1])  # color label
            color_name = colors[color_label]
            
            src_path = f'../images/{img_name}'
            dst_path = f'../colors_ps/{color_name}/{img_name}'
            
            if os.path.exists(src_path):
                shutil.copy(src_path, dst_path)
                print(f"Copied {img_name} to {color_name}")
            else:
                print(f"Image {src_path} not found")