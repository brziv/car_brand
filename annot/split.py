from sklearn.model_selection import train_test_split

with open("full.txt", "r") as f:
    lines = [line.strip() for line in f]

labels = [line.strip().split()[-1] for line in lines]

# Split train (80%) / temp (20%)
train_samples, temp_samples, train_labels, temp_labels = train_test_split(
    lines, labels, test_size=0.2, stratify=labels, random_state=42
)

# Split val (10%) / test (10%)
val_samples, test_samples = train_test_split(
    temp_samples, test_size=0.5, stratify=temp_labels, random_state=42
)

# Save splits
with open("train.txt", "w") as f:
    f.write("\n".join(train_samples))
with open("val.txt", "w") as f:
    f.write("\n".join(val_samples))
with open("test.txt", "w") as f:
    f.write("\n".join(test_samples))

print("Done.")
