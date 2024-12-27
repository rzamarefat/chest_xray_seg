import streamlit as st
import cv2
import numpy as np
from LungSeg import LungSeg

if "model_choice" not in st.session_state:
    st.session_state.model_choice = "YOLOv11"
if "lungseg" not in st.session_state:
    st.session_state.lungseg = LungSeg(model_name="yolov11")

model_choice = st.radio(
    "Choose the segmentation model:",
    options=["UNET++", "YOLOv11"],
    index=1 if st.session_state.model_choice == "YOLOv11" else 0
)

if model_choice != st.session_state.model_choice:
    st.session_state.model_choice = model_choice
    st.session_state.lungseg = LungSeg(model_name=model_choice.lower())
    st.experimental_rerun() 

lungseg = st.session_state.lungseg 

st.title("Lung Segmentation")
st.write("Segment Lung from X-Ray Images")

uploaded_files = st.file_uploader("Choose images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    st.write(f"You have uploaded {len(uploaded_files)} file(s).")
    
    images = []
    for uploaded_file in uploaded_files:
        try:
            file_bytes = uploaded_file.read()
            np_array = np.frombuffer(file_bytes, np.uint8)
            image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
            images.append(image)
            image_resized = cv2.resize(image, (300, 300))
            image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
            st.image(image_rgb, caption=uploaded_file.name, use_column_width=False)
        except Exception as e:
            st.error(f"Error loading {uploaded_file.name}: {e}")

    if st.button("Analyze"):
        st.write(f"Analyzing and preprocessing images using the {model_choice} model...")
        predicted_masks = lungseg(images)

        st.empty()

        num_columns = len(predicted_masks)
        cols = st.columns(num_columns)

        for i, (predicted_mask, col) in enumerate(zip(predicted_masks, cols)):
            print("predicted_mask.shape", predicted_mask.shape)
            mask_resized = cv2.resize(predicted_mask, (300, 300))
            mask_rgb = cv2.cvtColor(mask_resized, cv2.COLOR_BGR2RGB)

            with col:
                st.image(mask_rgb, caption=f"Processed Image {i + 1}", use_column_width=False)
else:
    st.info("No images uploaded yet.")
