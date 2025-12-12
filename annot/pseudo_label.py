import os
import gc
import torch
from transformers import (
    CLIPProcessor, CLIPModel
)
from PIL import Image
from tqdm import tqdm

# config
IMAGE_DIR = "/ssd1/team_thuctap/ntthai/car_brand_and_color/images"
INPUT_TXT = "/ssd1/team_thuctap/ntthai/car_brand_and_color/annot/br_label.txt"
OUTPUT_TXT = "/ssd1/team_thuctap/ntthai/car_brand_and_color/annot/full_pslabel.txt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LABELS = ["white", "black", "silver or gray", "red", "blue"]
LIMIT = 10  # test 10 first

# CLIP pseudo-label
def run_clip_large(image_paths):
    print("Loading CLIP Large...")
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(DEVICE)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    model.eval()

    clip_labels = {}

    text_prompts = [f"a photograph of a {label} car" for label in LABELS]
    with torch.no_grad():
        text_inputs = processor(text=text_prompts, return_tensors="pt", padding=True).to(DEVICE)
        text_features = model.get_text_features(**text_inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    for img_path in tqdm(image_paths, desc="CLIP Large"):
        try:
            image = Image.open(img_path).convert("RGB")
        except:
            clip_labels[img_path] = "5"  # other
            continue

        inputs = processor(images=image, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            img_features = model.get_image_features(**inputs)
            img_features = img_features / img_features.norm(dim=-1, keepdim=True)
            sims = (img_features @ text_features.T).squeeze(0)
        idx = sims.argmax().item()
        clip_labels[img_path] = str(idx)

    del model, processor
    torch.cuda.empty_cache(); gc.collect()
    return clip_labels

# read merged.txt
image_data = {}
with open(INPUT_TXT, 'r') as f:
    for line in f:
    # for i, line in enumerate(f):
    #     if i >= LIMIT:
    #         break
        line = line.strip()
        if ' ' in line:
            img_name, label = line.rsplit(' ', 1)
            image_data[img_name] = label
        else:
            print(f"Skipping invalid line: {line}")

image_files = list(image_data.keys())
image_paths = [os.path.join(IMAGE_DIR, f) for f in image_files if os.path.exists(os.path.join(IMAGE_DIR, f))]
print(f"Found {len(image_paths)} images.")

clip_labels = run_clip_large(image_paths)

with open(OUTPUT_TXT, 'w') as f:
    for img_path in image_paths:
        img_name = os.path.basename(img_path)
        original_label = image_data.get(img_name, "unknown")
        final_label = clip_labels.get(img_path, "other")
        f.write(f"{img_name} {original_label} {final_label}\n")

print(f"Saved pseudo-labels to {OUTPUT_TXT}.")
