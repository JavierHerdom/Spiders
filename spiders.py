import tensorflow as tf
import streamlit as st
import cv2
from PIL import Image, ImageOps
import numpy as np
import requests
import zipfile
import os


MODEL_URL = "https://inteligencia-artificial.s3.sa-east-1.amazonaws.com/spider_mnist.h5py.zip"
MODEL_FILE = "spider_mnist.h5py"

def import_and_predict(image_data, model):
        size = (300,300)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = np.asarray(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resize = (cv2.resize(img, dsize=(300, 300),    interpolation=cv2.INTER_CUBIC))/255.
        img_reshape = img_resize[np.newaxis,...]
        prediction = model.predict(img_reshape)
        return prediction

def download_file(url):
    with st.spinner('Downloading model...'):
        # from https://stackoverflow.com/a/16696317
        local_filename = url.split('/')[-1]
        # NOTE the stream=True parameter below
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    # If you have chunk encoded response uncomment if
                    # and set chunk_size parameter to None.
                    # if chunk:
                    f.write(chunk)

        return local_filename

def unzip_model():
    with zipfile.ZipFile(f"{MODEL_FILE}.zip", 'r') as zip_ref:
        zip_ref.extractall("./")

if __name__ == '__main__':
    if not os.path.isdir(MODEL_FILE):
        print(f"Descargando modelo desde {MODEL_URL}")
        _ = download_file(f'{MODEL_URL}')

    unzip_model()

    model = tf.keras.models.load_model(MODEL_FILE)
    st.write("SPIDER-IA")
    st.write("Esta es una red neuronal que determina si la imagen subida es o no una araña de rincón.")
    file = st.file_uploader("Please upload an image file", type=["jpg", "png"])

    if file is None:
        st.text("Please upload an image file")
    else:
        image = Image.open(file)
        st.image(image, use_column_width=True)
        prediction = import_and_predict(image, model)

        if np.argmax(prediction) == 0:
            st.write("Es de rincon")
        else:
            st.write("No es de rincon")
        st.text(f"Precision: {(prediction[0][np.argmax(prediction)] * 100):.2f}%")