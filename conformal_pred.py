from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
import keras

from keras.src.utils import image_utils


if __name__ == "__main__":

    df_val = pd.read_csv("data/val.csv", names=["filename", "class"])
    datagen = ImageDataGenerator(rescale=1./255)

    classes = ["boar", "deer"]
    
    for c in classes:    
        val_generator = datagen.flow_from_dataframe(
            df_val,
            'data/imgs',
            target_size=(224, 224),
            batch_size=64,
            class_mode='categorical',
            classes=[c],
            shuffle=False,
        )

        model = keras.models.load_model(f"data/models/model.h5")
        y_true = val_generator.labels
        y_pred = model.predict(val_generator)

        E = y_pred.flatten()   # [np.arange(len(y_pred)), y_true]
        alphas = np.arange(0, 1, step=0.05)
        qs = np.quantile(E, alphas)

        with open(f"data/{c}_quantiles.txt", "w") as file:
            for a, q in zip(alphas, qs):
                file.write(f"{a} {q}\n")
        