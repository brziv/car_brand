import os
from collections import Counter

# List of files to check
files = ['train.txt', 'val.txt', 'test.txt']

for file in files:
    if not os.path.exists(file):
        print(f"File {file} does not exist.")
        continue
    
    with open(file, 'r') as f:
        lines = f.readlines()
    
    brands = []
    colors = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 3:
            try:
                brands.append(int(parts[-2]))
                colors.append(int(parts[-1]))
            except ValueError:
                continue
    
    brand_count = Counter(brands)
    color_count = Counter(colors)
    
    print(f"For {file}:")
    print("Brand distribution:")
    for idx, count in sorted(brand_count.items()):
        print(f"{idx}: {count}")
    print("Color distribution:")
    for idx, count in sorted(color_count.items()):
        print(f"{idx}: {count}")
    print()