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
    st.image(image, caption="Uploaded Image")

@st.cache_resource
def load_model():
    return YOLO("models/yolo11n.pt")

# if st.button("Train model"):
    # model = YOLO("models/yolo11n.pt",verbose=True)
    # trains = model.train(data="dataset/data.yaml", epochs=3,device="cpu")
    # st.success("Training finished")
    # st.write("### Training Results")
    # st.json(trains)
results = load_model()(image)
st.write("### Detection Results")
st.image(results[0].plot(), channels="BGR")