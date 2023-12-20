# Boar - Deer classification
This repo contains the scripts to download boar/deer images, train a CNN to classify the images and evaluate the model.

## Preparation
The code was run using python 3.10 in Ubuntu 22.04. Build an environment and install dependencies (requirements.txt).

The original image urls provided are in the folder `data/sources`. If desired, these lists of urls can be expaneded using the script `add_urls.py` which adds approximately +1000 images for each class.

Run the script `download_data.py` to download all the images into the folder `data/imgs`.

## Training
I used Tensorflow to train the image classifier. I tried four different CNN architectures:

- MobileNetV2
- ResNet50
- EfficientNetV2
- InceptionV3

All the experiments were done using models pretreined on imagenet. On top of the backbone a linear classifier of 2 layers was added. Leaving the backbone freezed and training only the linear layers led to better results than training the whole network.

All these experiments can be reproduced by using the script `train_model.py`. This scripts splits the data in train-val (80-20), loads the backbone and trains the model for a number of epochs (adjustable). The model with the lower validation loss is saved. All the models are saved in the folder `data/models/`

## Evaluation and Reliablity
As a part of the evaluation of the model and the confidence of its predictions I tried to do a simple measure the reliability of the output using conformal prediction. Basically, using a calibration set (I used the validation set) I calculate the quantiles of the probabilites with which the model predicts the right output and I use this as a measure of confidence. This quantiles are calculated for all the saved models using the script `conformal_pred.py`

## Model deployment
To deploy the model I used fastapi to build a simple API where you can upload a file or input a new image URL to get the predictions of the model. The response includes the predicted class, the probabilities predicted by the model and the confidence of the predicted class.

I also built an app with streamlit to have a more interactive option. Here I included the possibility to show the GradCam++ heatmap over the input image to make a better analysis of the models.

### FastAPI
You can deploy your trained model (`model.h5`) by running:

```bash
uvicorn api:app --reload
```

### Streamlit
Run:

```bash
streamlit run streamlit_app.py
```

This will open an interactive page where you can chose between all the saved models and upload an image to try it out. You can also show (or not) the heatmap obtained using GradCam++.

