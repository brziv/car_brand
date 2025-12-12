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
GROUND_TRUTH_TXT = "/ssd1/team_thuctap/ntthai/car_brand_and_color/annot/full_label.txt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load brand and color lists
BRAND_LIST = [line.strip() for line in open("annot/br_list.txt")]
COLOR_LIST = [line.strip() for line in open("annot/cl_list.txt")]

# Create combined labels
combined_labels = [f"{color} {brand}" for color in COLOR_LIST for brand in BRAND_LIST]

# CLIP prediction function
def predict_with_clip(image_paths, text_labels, model_name="openai/clip-vit-large-patch14"):
    print(f"Loading CLIP Large for combined labels ({len(text_labels)})...")
    model = CLIPModel.from_pretrained(model_name).to(DEVICE)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()

    predictions = {}

    text_prompts = [f"a photograph of a {label} car" for label in text_labels]
    with torch.no_grad():
        text_inputs = processor(text=text_prompts, return_tensors="pt", padding=True).to(DEVICE)
        text_features = model.get_text_features(**text_inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    for img_path in tqdm(image_paths, desc=f"Predicting combined labels ({len(text_labels)})"):
        try:
            image = Image.open(img_path).convert("RGB")
        except:
            predictions[img_path] = -1  # invalid
            continue

        inputs = processor(images=image, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            img_features = model.get_image_features(**inputs)
            img_features = img_features / img_features.norm(dim=-1, keepdim=True)
            sims = (img_features @ text_features.T).squeeze(0)
        idx = sims.argmax().item()
        predictions[img_path] = idx

    del model, processor
    torch.cuda.empty_cache(); gc.collect()
    return predictions

# Load ground truth
ground_truth = {}
with open(GROUND_TRUTH_TXT, 'r') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 3:
            img_name, brand_idx, color_idx = parts
            ground_truth[img_name] = (int(brand_idx), int(color_idx))

image_files = list(ground_truth.keys())
image_paths = [os.path.join(IMAGE_DIR, f) for f in image_files if os.path.exists(os.path.join(IMAGE_DIR, f))]
print(f"Found {len(image_paths)} images out of {len(ground_truth)}.")

# Predict combined
combined_predictions = predict_with_clip(image_paths, combined_labels)

# Calculate accuracy
brand_correct = 0
color_correct = 0
both_correct = 0
total = 0

for img_path in image_paths:
    img_name = os.path.basename(img_path)
    true_brand, true_color = ground_truth[img_name]
    combined_idx = combined_predictions[img_path]
    if combined_idx != -1:
        pred_color = combined_idx // len(BRAND_LIST)
        pred_brand = combined_idx % len(BRAND_LIST)
        total += 1
        if pred_brand == true_brand:
            brand_correct += 1
        if pred_color == true_color:
            color_correct += 1
        if pred_brand == true_brand and pred_color == true_color:
            both_correct += 1

print(f"Total valid predictions: {total}")
print(f"Brand accuracy: {brand_correct / total:.4f}")
print(f"Color accuracy: {color_correct / total:.4f}")
print(f"Both accuracy: {both_correct / total:.4f}")
