import os
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
import keras


def compute_quantiles(model, step=0.05):
    df_val = pd.read_csv("data/val.csv", names=["filename", "class"])
    datagen = ImageDataGenerator(rescale=1./255)

    classes = ["boar", "deer"]
    
    quant = {}
    for ix, c in enumerate(classes):    
        val_generator = datagen.flow_from_dataframe(
            df_val,
            'data/imgs',
            target_size=(224, 224),
            batch_size=64,
            class_mode='categorical',
            classes=[c],
            shuffle=False,
        )

        y_true = val_generator.labels
        y_pred = model.predict(val_generator)
        print(np.mean(0 == y_pred.argmax(axis=1)))

        E = y_pred[np.arange(len(y_pred)), ix]
        alphas = np.arange(0, 1, step=step)
        qs = np.quantile(E, alphas)
        quant[c] = (alphas, qs)
        
    return quant

if __name__ == "__main__":
    for fn in os.listdir("data/models/"):
        
        model = keras.models.load_model(f"data/models/{fn}")
        quant = compute_quantiles(model)
        
        os.makedirs(f"data/quantiles/{fn}", exist_ok=True)
        for c, (alphas, qs) in quant.items():
            with open(f"data/quantiles/{fn}/{c}.txt", "w") as file:
                for a, q in zip(alphas, qs):
                    file.write(f"{a} {q}\n")
    
        