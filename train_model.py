import argparse
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dense, Dropout
from keras.callbacks import ModelCheckpoint
from keras.applications import MobileNetV2, ResNet50, EfficientNetV2B3, InceptionV3
import pandas as pd
from sklearn.model_selection import train_test_split

model_archs = {
    "mobilenet": MobileNetV2,
    "resnet": ResNet50,
    "inception": InceptionV3,
    "efficientnet": EfficientNetV2B3,
}

def load_backbone(arch="mobilenet"):
    
    ModelClass = model_archs[arch]
    base_model = ModelClass(
        include_top=False,
        weights="imagenet", 
        input_shape=(224,224,3),
        pooling="avg"
    )
    
    return base_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model with TensorFlow.')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--output-name', type=str, default='model.h5', help='Output name for saved model')
    parser.add_argument("--architecture", type=str, default="mobilenet", help="Architecture of the model to be trained", choices=list(model_archs.keys()))
    parser.add_argument("--train-backbone", action="store_true", help="Include this option to train the weights of the backbone")
    parser.add_argument("--split-seed", type=int, default=0, help="Random seed to split the dataset in train-val")

    args = parser.parse_args()

    df = pd.DataFrame(os.listdir("data/imgs"), columns=['filename'])
    df["class"] = df["filename"].str.split("_").str[0]

    df_train, df_val = train_test_split(df, test_size=0.2, stratify=df["class"], random_state=args.split_seed)

    df_train.to_csv("data/train.csv", index=False, header=False)
    df_val.to_csv("data/val.csv", index=False, header=False)

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

    # Has test time augmentation
    val_generator = datagen.flow_from_dataframe(
        df_val,
        'data/imgs',
        target_size=(224, 224),
        batch_size=64,
        class_mode='categorical',
        classes=classes,
    )
    

    base_model = load_backbone(args.architecture)
    for layer in base_model.layers:
        layer.trainable = args.train_backbone
        
    base_out = base_model.output
    x = Dense(256, activation='relu')(base_out)
    x = Dropout(0.2)(x)
    output = Dense(2, activation='sigmoid')(x)

    # classifier = Sequential([
    #     Dense(256, activation='relu'),
    #     Dropout(0.2),
    #     Dense(2, activation='sigmoid')
    # ])
    
    # model = Sequential([
    #     base_model,
    #     classifier
    # ])
    
    model = Model(inputs=base_model.inputs, outputs=[output])

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