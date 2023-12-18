import argparse
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import ModelCheckpoint
from keras.applications import MobileNetV2, ResNet50, EfficientNetV2B3, InceptionV3
import pandas as pd


def load_backbone(arch="mobilenet"):
    
    base_model = MobileNetV2(
        include_top=False,
        weights="imagenet", 
        input_shape=(224,224,3),
        pooling="avg"
    )

    # base_model = InceptionV3(
    #     include_top=False,
    #     weights="imagenet", 
    #     input_shape=(224,224,3),
    #     pooling="avg"
    # )
    
    return base_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model with TensorFlow.')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--output-name', type=str, default='model.h5', help='Output name for saved model')
    parser.add_argument("--architecture", type=str, default="mobilenetv2", help="Architecture of the model to be trained")
    parser.add_argument("--train-backbone", action="store_true", help="Include this option to train the weights of the backbone")

    args = parser.parse_args()


    df_train = pd.read_csv("data/train.csv", names=["filename", "class"])
    df_val = pd.read_csv("data/val.csv", names=["filename", "class"])

    datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.1, rotation_range=20, horizontal_flip=True)
    # val_gen = ImageDataGenerator(rescale=1./255)

    classes = ["boar", "deer"]
    
    train_generator = datagen.flow_from_dataframe(
        df_train,
        'data/imgs', 
        target_size=(224, 224),
        batch_size=64,
        class_mode='categorical',
        shuffle=True,
        classes=classes,
    )

    val_generator = datagen.flow_from_dataframe(
        df_val,
        'data/imgs',
        target_size=(224, 224),
        batch_size=64,
        class_mode='categorical',
        classes=classes,
    )
    

    base_model = load_backbone(args.architecture)
    base_model.trainable = args.train_backbone

    classifier = Sequential([
        Dense(256, activation='relu'),
        Dropout(0.2),
        Dense(2, activation='sigmoid')
    ])
    
    model = Sequential([
        base_model,
        classifier
    ])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    os.makedirs("data/models/", exist_ok=True)
    model_checkpoint = ModelCheckpoint(
        filepath=f"data/models/{args.output_name}",
        save_best_only=True,
        monitor='val_loss',
        mode='min',
        verbose=1
    )
    
    model.fit(
        train_generator, 
        epochs=args.epochs,
        validation_data=val_generator,
        callbacks=[model_checkpoint]
    )