import streamlit as st
import cv2
import numpy as np
from LungSeg import LungSeg

lungseg = LungSeg()

st.title("Lung Segmentation")

st.write("Segment Lung from X-Ray Images")

uploaded_files = st.file_uploader("Choose images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    st.write(f"You have uploaded {len(uploaded_files)} file(s).")
    
    images = []
    for uploaded_file in uploaded_files:
        try:
            # Load image using OpenCV
            file_bytes = uploaded_file.read()
            np_array = np.frombuffer(file_bytes, np.uint8)
            image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)  # Read image in color (BGR)
            images.append(image)
            
            # Resize image for smaller display (e.g., 300x300)
            image_resized = cv2.resize(image, (300, 300))

            # Convert to RGB (since OpenCV uses BGR by default)
            image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
            
            # Display the image
            st.image(image_rgb, caption=uploaded_file.name, use_column_width=False)
        except Exception as e:
            st.error(f"Error loading {uploaded_file.name}: {e}")

    # Analyze button
    if st.button("Analyze"):
        st.write("Analyzing and preprocessing images...")

        # Convert images to a format compatible with LungSeg (OpenCV images)
        predicted_masks = lungseg(images)

        # Clear previous images
        st.empty()

        # Create columns for horizontal display of the output images
        num_columns = len(predicted_masks)
        cols = st.columns(num_columns)

        # Display the predicted masks or processed images horizontally
        for i, (predicted_mask, col) in enumerate(zip(predicted_masks, cols)):
            # Resize mask image for smaller display
            mask_resized = cv2.resize(predicted_mask, (300, 300))

            # Convert the mask to RGB for visualization (if necessary)
            mask_rgb = cv2.cvtColor(mask_resized, cv2.COLOR_BGR2RGB)

            with col:
                st.image(mask_rgb, caption=f"Processed Image {i + 1}", use_column_width=False)
else:
    st.info("No images uploaded yet.")
