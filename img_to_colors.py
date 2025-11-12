import os
import shutil

# Read color labels
with open('annot/color.txt', 'r') as f:
    colors = [line.strip() for line in f.readlines()]

# Create folders for each color
os.makedirs('images_by_color', exist_ok=True)
for color in colors:
    os.makedirs(f'images_by_color/{color}', exist_ok=True)

# Read full.txt and organize images
with open('annot/full.txt', 'r') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 3:
            img_name = ' '.join(parts[:-2])  # image name
            color_label = int(parts[-1])  # color label
            color_name = colors[color_label]
            
            src_path = f'images/{img_name}'
            dst_path = f'images_by_color/{color_name}/{img_name}'
            
            if os.path.exists(src_path):
                shutil.copy(src_path, dst_path)
                print(f"Copied {img_name} to {color_name}")
            else:
                print(f"Image {src_path} not found")