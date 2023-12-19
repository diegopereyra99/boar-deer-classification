import streamlit as st
from PIL import Image
import tensorflow as tf
import os
import requests
import numpy as np

import numpy as np
from keras.src.utils import image_utils

classes = ["boar", "deer"]

def load_quantiles(model_fn):
    quants =  {c : np.loadtxt(f"data/quantiles/{model_fn}/{c}.txt") for c in classes}
    return quants
    
    
def confidence_from_prob(p, quant):
    for c, q in zip(*quant.T):
        if p > q:
            last_c = c
            continue
    
    return last_c
    
def class_to_name(pc):
    if pc == 0: 
        return 'boar'
    else:
        return 'deer'

def load_model(model_path):
    return tf.keras.models.load_model(model_path)

# Function to preprocess the image for model prediction
def preprocess_image(image):
    img = image.resize((224, 224))
    img = image_utils.img_to_array(img)
    img = img / 255.0
    
    img = img.reshape(-1,224,224,3)
    
    return img
        
# Function to make a prediction using the loaded model
def predict(model, image):
    image = preprocess_image(image)
    predictions = model.predict(image)
    return predictions.flatten()

def main():
    st.title("Boar - Deer classification")

    # Select a folder containing models
    selected_model = st.sidebar.selectbox("Select model:", os.listdir("data/models"))

    # Load the selected model
    model_path = os.path.join("data/models", selected_model)
    model = tf.keras.models.load_model(model_path)
    
    conf_level = st.sidebar.checkbox("Confidence Level", value=True)
    if conf_level:
        quants = load_quantiles(os.path.basename(model_path))

    # Upload an image or provide a URL
    uploaded_file = st.file_uploader("Choose an image file:", type=["jpg", "jpeg", "png"])
    image_url = st.text_input("Or enter an image URL:")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", use_column_width=True)
    elif image_url:
        image = Image.open(requests.get(image_url, stream=True).raw)
        st.image(image, caption="Image from URL.", use_column_width=True)
    else:
        st.warning("Please upload an image or provide a URL.")
        st.stop()

    predictions = predict(model, image)


    st.subheader("Class Probabilities:")
    for j, prob in enumerate(predictions):
        st.write(f"{classes[j]}: {prob:.2%}")
    
    if conf_level:
        class_prob = predictions.max()
        pred_class_name = class_to_name(predictions.argmax())
        conf = confidence_from_prob(class_prob, quants[pred_class_name])
            
        st.write(f"Model predicts {pred_class_name.upper()} with a confidence of {conf:.0%}")

if __name__ == "__main__":
    main()
