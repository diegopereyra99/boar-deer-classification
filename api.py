from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from io import BytesIO
import numpy as np
import requests
import tensorflow as tf
from streamlit_app import predict, confidence_from_prob

app = FastAPI()

model = tf.keras.models.load_model("data/models/model.h5")
classes = ["boar", "deer"]
quants =  {c : np.loadtxt(f"data/quantiles/model.h5/{c}.txt") for c in classes}
    

def classify_image(image):
    preds = predict(model, image)
    pred_class = classes[preds.argmax()]
    conf = confidence_from_prob(preds.max(), quants[pred_class])
    
    return {
        "pred_class": pred_class,
        "confidence": str(round(conf*100)), 
        "probs": dict(zip(classes, preds.astype(str))),
    }


@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    try:
        image = Image.open(BytesIO(await file.read()))
        result = classify_image(image)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.post("/imageurl/")
async def create_image_url(image_url: str):
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
        result = classify_image(image)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)