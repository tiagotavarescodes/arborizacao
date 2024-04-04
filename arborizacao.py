# -*- coding: utf-8 -*-
"""Untitled14.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/14gbsmdX-oORQg8qqKaHyQ_Imsf0_FEA0
"""

import streamlit as st
from keras.models import load_model
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("/arborizacao/keras_model.h5", compile=False)

# Define class names
class_names = ["Não", "Sim"]

def preprocess_image(image):
    # Redimensionar a imagem para 224x224 e recortar do centro
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    # Converter a imagem em uma matriz numpy
    image_array = np.asarray(image)
    # Normalizar a matriz da imagem
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    # Expandir a dimensão para atender aos requisitos do modelo (1, 224, 224, 3)
    return np.expand_dims(normalized_image_array, axis=0)

def predict(image):
    # Pré-processar a imagem
    data = preprocess_image(image)
    # Prever a classe
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    return class_name, confidence_score

def main():
    st.title("Classificador de Imagens Sim/Não")

    uploaded_file = st.file_uploader("Escolha uma imagem...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Imagem carregada', use_column_width=True)
        st.write("")
        st.write("Classificando...")

        class_name, confidence_score = predict(image)

        st.write(f"Classe: {class_name}")
        st.write(f"Pontuação de Confiança: {confidence_score}")

if __name__ == "__main__":
    main()
