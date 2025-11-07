from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import io
import joblib
import uvicorn

app = FastAPI()

# Modeli yükle
model = joblib.load("model.joblib")

# Sınıf isimleri
classes = {
    0: "Bakteriyel Yaprak Yanıklığı",
    1: "Kahverengi Nokta",
    2: "Yaprak İsi"
}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Görseli oku
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('L')
        image = image.resize((28, 28))
        img_array = np.array(image).flatten().reshape(1, -1)

        # Tahmin
        prediction = model.predict(img_array)[0]
        disease_name = classes.get(int(prediction), "Bilinmeyen hastalık")

        return JSONResponse(content={"disease": disease_name})
    except Exception as e:
        return JSONResponse(content={"error": str(e)})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
