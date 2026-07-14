import streamlit as st
import json
import numpy as np
import tensorflow as tf
import os

from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import (
    preprocess_input as preprocess_input_mobilenet
)

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "model_car_logo_best.keras")
CLASS_PATH = os.path.join(BASE_DIR, "class_names.json")


def run():

    st.set_page_config(
        page_title="Car Logo Classification",
        page_icon="🚗",
        layout="centered"
    )

    st.title("🚗 Car Logo Classification")
    st.write("Upload a car logo image to predict its brand.")

    # Load model
    model = tf.keras.models.load_model(MODEL_PATH)

    # Load class names
    with open(CLASS_PATH, "r") as f:
        raw_map = json.load(f)
        idx_to_class = {int(k): v for k, v in raw_map.items()}

    img_height = 220
    img_width = 220

    def prediction(image):

        # simpan gambar asli untuk ditampilkan
        image_show = image.copy()

        # resize untuk model
        image = image.resize((img_width, img_height))

        x = tf.keras.utils.img_to_array(image)
        x = preprocess_input_mobilenet(x)
        x = np.expand_dims(x, axis=0)

        pred = model.predict(x, verbose=0)

        predicted_class = idx_to_class[np.argmax(pred[0])]
        confidence = float(np.max(pred))

        return image_show, predicted_class, confidence, pred[0]

    uploaded_file = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

        if st.button("Predict"):

            image_show, predicted_class, confidence, probabilities = prediction(image)

            st.success(f"Prediction : **{predicted_class.upper()}**")

            st.metric(
                "Confidence",
                f"{confidence*100:.2f}%"
            )

            st.subheader("Prediction Probability")

            for idx, prob in enumerate(probabilities):
                label = idx_to_class[idx]

                st.write(f"**{label.upper()}** : {prob*100:.2f}%")
                st.progress(float(prob))


if __name__ == "__main__":
    run()