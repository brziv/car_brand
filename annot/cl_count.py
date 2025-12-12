from collections import Counter
import matplotlib.pyplot as plt

with open("full_label.txt", "r", encoding="utf-8") as merged:
    lines = [line.split() for line in merged]

labels = [int(parts[-1]) for parts in lines]

count = Counter(labels)

with open("cl_list.txt", "r", encoding="utf-8") as synset:
    brands = [brand.strip() for brand in synset]

for i, brand in enumerate(brands):
    print(f"Class {i} ({brand}): {count.get(i)}")

x = range(len(brands))
y = [count.get(i, 0) for i in x]

sorted_pairs = sorted(zip(brands, y), key=lambda p: p[1], reverse=True)
brands, y = zip(*sorted_pairs)

plt.figure(figsize=(10,6))
plt.bar(x, y, color='skyblue')
plt.xticks(x, brands, rotation=45, ha='right')
plt.tight_layout()
plt.savefig("colors.png", dpi=300)
plt.close()