import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the model
model = tf.keras.models.load_model("pets_model.h5")

# Class labels (0–36)
NUM_CLASSES = 37

# Labels to map to IDs
LABEL_NAMES = [
    "Abyssinian", "american_bulldog", "american_pit_bull_terrier",
    "basset_hound", "beagle", "Bengal", "Birman", "Bombay", "boxer",
    "British_Shorthair", "chihuahua", "Egyptian_Mau", "english_cocker_spaniel",
    "english_setter", "german_shorthaired", "great_pyrenees", "havanese",
    "japanese_chin", "keeshond", "leonberger", "Maine_Coon", "miniature_pinscher",
    "newfoundland", "Persian", "pomeranian", "pug", "Ragdoll", "Russian_Blue",
    "saint_bernard", "samoyed", "scottish_terrier", "shiba_inu", "Siamese",
    "Sphynx", "staffordshire_bull_terrier", "wheaten_terrier", "yorkshire_terrier"
]

st.title("🐶🐱 Oxford Pets Classifier")
st.write("Upload an image of a pet, and the model will predict its breed.")

# Image uploader
uploaded_file = st.file_uploader("Choose a pet image", type=["jpg", "jpeg", "png"])

def preprocess_image(image_file):
    image = Image.open(image_file).convert("RGB")
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)  # [1, 224, 224, 3]
    return image

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
    
    with st.spinner("Predicting..."):
        image = preprocess_image(uploaded_file)
        prediction = model.predict(image)
        pred_class = np.argmax(prediction)

    st.success(f"Predicted breed: {LABEL_NAMES[pred_class]}")

