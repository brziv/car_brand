from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
from collections import defaultdict

with open("clean_label.txt", "r") as f:
    lines = [line.strip() for line in f]

# Group by prefix
prefix_to_lines = defaultdict(list)
prefix_to_label = {}
for line in lines:
    parts = line.split()
    if len(parts) >= 3:
        try:
            filename = parts[0]
            prefix = filename.split('_')[0]
            num1 = int(parts[-2])
            num2 = int(parts[-1])
            prefix_to_lines[prefix].append(line)
            if prefix not in prefix_to_label:
                prefix_to_label[prefix] = [num1, num2]
        except (ValueError, IndexError):
            continue

# Prepare data for splitting prefixes
all_prefixes = list(prefix_to_label.keys())
prefix_brands = [prefix_to_label[p][0] for p in all_prefixes]

# Group prefixes by type
group1_prefixes = [p for p in all_prefixes if p.isdigit()]
group2_prefixes = [p for p in all_prefixes if not p.isdigit()]

# Group2 all to train
train_prefixes = []
train_prefixes = group2_prefixes.copy()

# Split group1 prefixes stratified on brand
if group1_prefixes:
    group1_brands = [prefix_to_label[p][0] for p in group1_prefixes]
    # Split group1 into train and val
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=0)
    for train_idx, val_idx in sss.split(group1_prefixes, group1_brands):
        group1_train_prefixes = [group1_prefixes[i] for i in train_idx]
        val_prefixes = [group1_prefixes[i] for i in val_idx]
    
    # Add group1 train prefixes to train_prefixes
    train_prefixes.extend(group1_train_prefixes)
else:
    val_prefixes = []

# Now assign samples
train_samples = []
for p in train_prefixes:
    train_samples.extend(prefix_to_lines[p])
val_samples = []
for p in val_prefixes:
    val_samples.extend(prefix_to_lines[p])

# Save splits
with open("train.txt", "w") as f:
    f.write("\n".join(train_samples))
with open("val.txt", "w") as f:
    f.write("\n".join(val_samples))

print("Done.")
