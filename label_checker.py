import streamlit as st
import os
from PIL import Image
import json

# Paths
images_by_color_dir = 'images_by_color'
full_txt = 'annot/full.txt'
corrected_labels_file = 'annot/corrected_labels.json'

# Load colors
with open('annot/color.txt', 'r') as f:
    colors = [line.strip() for line in f.readlines()]

# Load existing labels from full.txt
labels = {}
with open(full_txt, 'r') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 3:
            img_name = ' '.join(parts[:-2])
            color_label = int(parts[-1])
            labels[img_name] = color_label

# Load corrected labels if exists
if os.path.exists(corrected_labels_file):
    with open(corrected_labels_file, 'r') as f:
        corrected_labels = json.load(f)
else:
    corrected_labels = {}

# Function to save corrected labels
def save_corrected_labels():
    with open(corrected_labels_file, 'w') as f:
        json.dump(corrected_labels, f, indent=4)

# Streamlit app
st.title("Image Label Checker and Corrector")

# Sidebar for color selection
selected_color = st.sidebar.selectbox("Select Color Group", colors)

# Get images in selected color folder
color_folder = os.path.join(images_by_color_dir, selected_color)
if os.path.exists(color_folder):
    images = [f for f in os.listdir(color_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]
else:
    images = []
    st.error(f"Folder {color_folder} does not exist")

if images:
    # Session state for current image index
    if 'current_index' not in st.session_state:
        st.session_state.current_index = 0

    # Navigation
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("Previous") and st.session_state.current_index > 0:
            st.session_state.current_index -= 1
    with col3:
        if st.button("Next") and st.session_state.current_index < len(images) - 1:
            st.session_state.current_index += 1

    # Layout: Image on left, labels on right
    left_col, right_col = st.columns([2, 1])

    # Current image
    current_image = images[st.session_state.current_index]
    img_path = os.path.join(color_folder, current_image)
    img = Image.open(img_path)
    
    with left_col:
        st.image(img, caption=f"{current_image} ({st.session_state.current_index + 1}/{len(images)})", width=400)

    with right_col:
        # Current label
        current_label = labels.get(current_image, -1)
        if current_label != -1:
            st.write(f"Current Label: {colors[current_label]} ({current_label})")
        else:
            st.write("Current Label: Not found")

        # Corrected label
        corrected = corrected_labels.get(current_image, current_label)
        new_label = st.selectbox("Correct Label", range(len(colors)), index=corrected if corrected != -1 else 0, format_func=lambda x: f"{x}: {colors[x]}")

        if st.button("Save Correction"):
            corrected_labels[current_image] = new_label
            save_corrected_labels()
            st.success(f"Saved correction for {current_image}: {colors[new_label]}")

    # Progress
    st.progress((st.session_state.current_index + 1) / len(images))
else:
    st.write("No images in this group")