import os
import keras
import cv2
import urllib
import shutil

import numpy as np
from keras.src.utils import image_utils


names = os.listdir("data/models") #["model.h5", "model1.h5", "mobilenetv2-0.h5", "mobilenetv2-1.h5"]
model = keras.models.load_model("data/models/model.h5")

classes = ["boar", "deer"]
quant = {c : np.loadtxt(f"data/{c}_quantiles.txt") for c in classes}

def confidence_from_prob(p, cl):
    for c, q in zip(*quant[cl].T):
        if p > q:
            last_c = c
            continue
        return last_c
    
def class_to_name(pc):
    if pc == 0: 
        return 'boar'
    else:
        return 'deer'
    

while True:
    url = input('Please enter the url of the image: ')
    
    try:
        print('Downloading the image')
        if url.startswith("http"):
            urllib.request.urlretrieve(url, 'data/temp.jpg')
        else:
            shutil.copy(url, "data/temp.jpg")
            
        # img = cv2.imread('data/temp.jpg', 1)[:, :, ::-1]
        img = image_utils.load_img(
            "data/temp.jpg",
            target_size=(224, 224),
            # interpolation='bilinear',
        )
        if img is not None:    
            # img = cv2.resize(img, (224,224))
            
            img = image_utils.img_to_array(img)
            img = img / 255.0
            # img = image_utils.load_img(
            #     ,
            #     color_mode=self.color_mode,
            #     target_size=self.target_size,
            #     interpolation=self.interpolation,
            #     keep_aspect_ratio=self.keep_aspect_ratio,
            # )
            
            img = img.reshape(-1,224,224,3)
            
            pred_prob = model.predict(img).flatten()
            class_prob = pred_prob.max()
            pred_class_name = class_to_name(pred_prob.argmax())
            conf = confidence_from_prob(class_prob, pred_class_name)
                
            print(f"PROBS: boar - {pred_prob.flatten()[0]:.2%}  //  deer - {pred_prob.flatten()[1]:.2%}")
            print(f"Model predicts {pred_class_name.upper()} with a confidence of {conf:.0%}")

    except Exception as ex:
        print(ex)
# %%
