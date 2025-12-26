#!/usr/bin/env python3
import shutil
from pathlib import Path

root = Path(__file__).resolve().parents[1]
decisions_path = root / 'decisions.txt'
full_label_path = root / 'annot' / 'full_label.txt'
backup_path = root / 'annot' / 'full_label.txt.bak'

# Read decisions
removes = set()
keeps = set()
with decisions_path.open('r', encoding='utf-8') as f:
    for ln in f:
        ln = ln.strip()
        if not ln:
            continue
        if ' ' not in ln:
            continue
        name, decision = ln.rsplit(' ', 1)
        name = name.strip()
        decision = decision.strip().lower()
        if decision == 'remove':
            removes.add(name)
        elif decision == 'keep':
            keeps.add(name)

print(f"Decisions: {len(removes)} remove, {len(keeps)} keep")

# Backup original
shutil.copy2(full_label_path, backup_path)
print(f"Backed up {full_label_path} -> {backup_path}")

# Process full_label
removed = 0
total = 0
out_lines = []
with full_label_path.open('r', encoding='utf-8') as f:
    for ln in f:
        total += 1
        s = ln.rstrip('\n')
        if not s:
            out_lines.append(ln)
            continue
        # First token is filename (may contain spaces? but here filenames have spaces)
        # We'll split by whitespace and reconstruct the filename tokens until we hit a token that ends with .jpg
        parts = s.split()
        # Find the index of the token that endswith .jpg
        idx = None
        for i, token in enumerate(parts):
            if token.lower().endswith('.jpg'):
                idx = i
                break
        if idx is None:
            # malformed line: cannot find jpg; keep as is
            out_lines.append(ln)
            continue
        filename = ' '.join(parts[: idx + 1])
        if filename in removes:
            removed += 1
            # skip line
        else:
            out_lines.append(ln)

# Write back
with full_label_path.open('w', encoding='utf-8') as f:
    f.writelines(out_lines)

print(f"Processed {total} lines, removed {removed} lines.")
print("Updated file written to", full_label_path)
print("Original backed up at", backup_path)
