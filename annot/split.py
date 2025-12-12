from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
import numpy as np

with open("fresh_label.txt", "r") as f:
    lines = [line.strip() for line in f]

# Parse labels: assuming each line is "filename num1 num2"
y = []
valid_lines = []
for line in lines:
    parts = line.split()
    if len(parts) >= 3:
        try:
            num1 = int(parts[-2])
            num2 = int(parts[-1])
            y.append([num1, num2])
            valid_lines.append(line)
        except ValueError:
            # Skip lines where num1 or num2 are not integers
            continue
    else:
        continue

y = np.array(y)
lines = valid_lines

# Use MultilabelStratifiedShuffleSplit for train/val/test
# First, split train (80%) and temp (20%)
msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, temp_idx in msss.split(lines, y):
    train_samples = [lines[i] for i in train_idx]
    temp_samples = [lines[i] for i in temp_idx]
    temp_y = y[temp_idx]

# Then split temp into val (50%) and test (50%)
msss_val = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
for val_idx, test_idx in msss_val.split(temp_samples, temp_y):
    val_samples = [temp_samples[i] for i in val_idx]
    test_samples = [temp_samples[i] for i in test_idx]

# Save splits
with open("train.txt", "w") as f:
    f.write("\n".join(train_samples))
with open("val.txt", "w") as f:
    f.write("\n".join(val_samples))
with open("test.txt", "w") as f:
    f.write("\n".join(test_samples))

print("Done.")
