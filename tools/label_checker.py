import streamlit as st
import os
from PIL import Image

# Paths
images_by_color_dir = '../colors_ps'
full_pslabel_txt = '../annot/result.txt'
full_status_txt = '../annot/full_status.txt'

# Load colors
with open('../annot/cl_list.txt', 'r') as f:
    colors = [line.strip() for line in f.readlines()]

# Function to load from full_pslabel.txt and create/update full_status.txt
def load_and_create_status():
    lines = []
    with open(full_pslabel_txt, 'r') as f:
        for line in f:
            parts = line.strip().rsplit(' ', 1)  # split from right to handle spaces in img_name
            if len(parts) == 2:
                img_name, color = parts
                lines.append([img_name, color, 'no'])
    
    # Write to full_status.txt
    with open(full_status_txt, 'w') as f:
        for parts in lines:
            f.write(' '.join(parts) + '\n')
    
    return lines

# Load existing labels and status from full_status.txt (create if not exists)
if not os.path.exists(full_status_txt):
    status_data = load_and_create_status()
else:
    status_data = []
    with open(full_status_txt, 'r') as f:
        for line in f:
            parts = line.strip().rsplit(' ', 2)  # img_name color status
            if len(parts) == 3:
                status_data.append(parts)

labels = {}
statuses = {}
for parts in status_data:
    img_name = parts[0]  # img_name (may have spaces)
    color_label = int(parts[1])  # color
    status = parts[2]
    labels[img_name] = color_label
    statuses[img_name] = status

# Function to save corrected labels and update status
def save_corrected_labels_and_status(img_name, new_label):
    statuses[img_name] = 'yes'
    
    # Update full_status.txt
    lines = []
    with open(full_status_txt, 'r') as f:
        for line in f:
            parts = line.strip().rsplit(' ', 2)  # img_name color status
            if parts[0] == img_name:
                parts[1] = str(new_label)  # update color label
                parts[2] = 'yes'  # update status
            lines.append(' '.join(parts))
    
    with open(full_status_txt, 'w') as f:
        f.write('\n'.join(lines) + '\n')

# Sidebar for color selection
selected_color = st.sidebar.selectbox("Select Color Group", colors)

# Get images in selected color folder that have status 'no'
color_folder = os.path.join(images_by_color_dir, selected_color)
if os.path.exists(color_folder):
    all_images = [f for f in os.listdir(color_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]
    images = [img for img in all_images if statuses.get(img, '') == 'no']
else:
    images = []
    st.error(f"Folder {color_folder} does not exist")

if images:
    # Reset index if images list changed
    if 'last_images' not in st.session_state or st.session_state.last_images != images:
        st.session_state.current_index = 0
        st.session_state.last_images = images
    
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
    left_col, right_col = st.columns([3, 2])

    # Current image
    current_image = images[st.session_state.current_index]
    img_path = os.path.join(color_folder, current_image)
    img = Image.open(img_path)
    
    # Set default selected label to current label when image changes
    if 'last_image' not in st.session_state or st.session_state.last_image != current_image:
        st.session_state.selected_label = labels.get(current_image, -1)
        st.session_state.last_image = current_image
    
    with left_col:
        st.image(img, caption=f"{current_image} ({st.session_state.current_index + 1}/{len(images)})", width='stretch')

    with right_col:
        # Current label
        current_label = labels.get(current_image, -1)
        if current_label != -1:
            st.write(f"**Current Label: {colors[current_label]} ({current_label})**")
        else:
            st.write("**Current Label: Not found**")

        st.write("")  # spacing
        
        # Label selection with buttons - 2 columns for better readability
        st.write("**Select New Label:**")
        cols = st.columns(2)
        selected_label = None
        for i in range(len(colors)):
            with cols[i % 2]:
                if st.button(f"{i}: {colors[i]}", key=f"label_{i}", width='stretch'):
                    selected_label = i
        
        # If a label button was pressed, update
        if selected_label is not None:
            st.session_state.selected_label = selected_label
        
        st.write("")  # spacing
        
        # Display selected label
        if 'selected_label' in st.session_state:
            st.info(f"Selected: {colors[st.session_state.selected_label]} ({st.session_state.selected_label})")
            new_label = st.session_state.selected_label
        else:
            new_label = current_label

        st.write("")  # spacing
        
        # Save button
        if st.button("💾 Save Correction", key="save", width='stretch'):
            if 'selected_label' in st.session_state:
                save_corrected_labels_and_status(current_image, st.session_state.selected_label)
                st.success(f"✓ Saved: {colors[st.session_state.selected_label]}")
                st.rerun()  # Refresh to update the list
            else:
                st.warning("Please select a label first")

    # Progress
    st.progress((st.session_state.current_index + 1) / len(images))
else:
    st.write("No images in this group")