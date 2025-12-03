from collections import Counter
import matplotlib.pyplot as plt

with open("train.txt", "r", encoding="utf-8") as merged:
    lines = [line.split() for line in merged]

labels = [int(parts[-2]) for parts in lines]

count = Counter(labels)

with open("br_list.txt", "r", encoding="utf-8") as synset:
    brands = [brand.strip() for brand in synset]

for i, brand in enumerate(brands):
    print(f"Class {i} ({brand}): {count.get(i)}")

x = range(len(brands))
y = [count.get(i, 0) for i in x]

plt.figure(figsize=(10,6))
plt.bar(x, y, color='skyblue')
plt.xticks(x, brands, rotation=45, ha='right')
plt.tight_layout()
plt.savefig("brands.png", dpi=300)
plt.close()