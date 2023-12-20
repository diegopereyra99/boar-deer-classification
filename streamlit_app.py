import cv2
import streamlit as st
from PIL import Image
import tensorflow as tf
import os
import requests
import numpy as np

import numpy as np
from keras.src.utils import image_utils
from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore


classes = ["boar", "deer"]

def load_quantiles(model_fn):
    quants =  {c : np.loadtxt(f"data/quantiles/{model_fn}/{c}.txt") for c in classes}
    return quants
    
    
def confidence_from_prob(p, quant):
    last_c = 0
    for c, q in zip(*quant.T):
        if p > q:
            last_c = c
            continue
    
    return last_c
    
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


def gradcam_heatmap(gradcam, img, class_ix):
    prep_img = preprocess_image(img)
    cam = gradcam(CategoricalScore(class_ix), prep_img)
    heatmap = cv2.applyColorMap(np.uint8(cam[0]*255), cv2.COLORMAP_JET)[..., ::-1]
    hmap = cv2.resize(heatmap, img.size)
    return hmap

def main():
    st.title("Boar - Deer classification")

    selected_model = st.sidebar.selectbox("Select model:", os.listdir("data/models"))

    model_path = os.path.join("data/models", selected_model)
    model = tf.keras.models.load_model(model_path)
    
    conf_level = st.sidebar.checkbox("Confidence Level", value=True)
    if conf_level:
        quants = load_quantiles(os.path.basename(model_path))
    
    apply_gradcam = st.sidebar.checkbox("GradCam++", value=False)
    if  apply_gradcam:
        gradcam = GradcamPlusPlus(
            model,
            model_modifier=ReplaceToLinear(),
            clone=True
        )
        gc_cl = st.sidebar.selectbox("GradCam looking at:", ["prediction", "boar", "deer"])
        
        
    # Upload an image or provide a URL
    uploaded_file = st.file_uploader("Choose an image file:", type=["jpg", "jpeg", "png"])
    image_url = st.text_input("Or enter an image URL:")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
    elif image_url:
        image = Image.open(requests.get(image_url, stream=True).raw)
        # st.image(image, caption="Image from URL.", use_column_width=True)
    else:
        st.warning("Please upload an image or provide a URL.")
        st.stop()

    predictions = predict(model, image)
    class_prob = predictions.max()
    pred_class_name = classes[predictions.argmax()]

    if apply_gradcam:
        cls_ix = classes.index(gc_cl) if gc_cl != "prediction" else predictions.argmax()
        hmap = gradcam_heatmap(gradcam, image, cls_ix)
        show_img = cv2.addWeighted(hmap, 0.3, np.array(image), 0.7, 0)
        show_img = Image.fromarray(show_img)
    else:
        show_img = image
        
    st.image(show_img, use_column_width=True)
    
    st.subheader("Class Probabilities:")
    for j, prob in enumerate(predictions):
        st.write(f"{classes[j]}: {prob:.2%}")
    
    if conf_level:
        conf = confidence_from_prob(class_prob, quants[pred_class_name])
        st.write(f"Model predicts {pred_class_name.upper()} with a confidence of {conf:.0%}")

    
    
    
if __name__ == "__main__":
    main()
