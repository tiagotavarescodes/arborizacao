import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from keras.models import load_model

# Load the model and labels
model = load_model("keras_model.h5", compile=False)
class_names = open("labels.txt", "r").readlines()

# Define a function to handle image processing and prediction
def predict_class(image_path):
    size = (224, 224)
    image = Image.open(image_path).convert("RGB")
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data = np.expand_dims(normalized_image_array, axis=0)
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()  # Remove potential newline characters
    confidence_score = prediction[0][index]
    return class_name, confidence_score

# Create the Streamlit app layout
st.title("Pode Plantar?")

uploaded_file = st.file_uploader("Escolha uma imagem para classificação", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagem carregada", use_column_width=True)

    class_name, confidence_score = predict_class(uploaded_file)

    st.write("**Classificação:** ", class_name)
    st.write("**Confiança:** ", confidence_score)
