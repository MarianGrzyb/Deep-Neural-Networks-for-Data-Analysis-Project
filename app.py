
from io import BytesIO

import streamlit as st
from PIL import Image

from src.inference import load_inference_model, predict_image


st.set_page_config(
    page_title="Fetal Ultrasound Classification",
    page_icon="🔬",
    layout="centered",
)

st.title("Fetal Ultrasound Classification")

st.write(
    "Upload an ultrasound image to classify it as benign, "
    "malignant, or normal."
)

checkpoint_path = st.text_input(
    "Model checkpoint",
    value="artifacts/sanity/training_v1_best_model.pth",
)

uploaded_file = st.file_uploader(
    "Upload an ultrasound image",
    type=["png", "jpg", "jpeg"],
)

if uploaded_file is not None:
    image_bytes = uploaded_file.getvalue()

    st.image(
        image_bytes,
        caption="Uploaded ultrasound image",
        use_container_width=True,
    )

    if st.button("Classify image", type="primary"):
        try:
            with st.spinner("Loading model and performing inference..."):
                model, transform, device = load_inference_model(
                    checkpoint_path
                )

                image = Image.open(BytesIO(image_bytes))

                predicted_class, probabilities = predict_image(
                    image,
                    model,
                    transform,
                    device,
                )

            st.subheader("Classification result")

            st.success(
                f"Predicted class: {predicted_class.capitalize()}"
            )

            st.subheader("Class probabilities")

            for class_name, probability in probabilities.items():
                st.write(
                    f"**{class_name.capitalize()}:** "
                    f"{probability * 100:.2f}%"
                )

                st.progress(probability)

            st.caption(f"Inference device: {device}")

        except Exception as error:
            st.error(f"Classification failed: {error}")
