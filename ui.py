import streamlit as st
from PIL import Image
from ultralytics import YOLO

st.title("🥔🥕 Potato & Carrot Defect Detection")

uploaded_file = st.file_uploader(
    "Upload image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is None:
    st.warning("⚠️ Image is required")
    st.stop()   
else:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

model = YOLO("yolo11n.pt")
results = model(image)

st.write("### Detection Results")
st.image(results[0].plot(), use_column_width=True)