import io
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from preprocess import detect_and_crop_face, preprocess_image
from model import FaceRecognitionModel

app = FastAPI()
model = FaceRecognitionModel()

@app.post("/recognize")
async def recognize_celebrity(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    face = detect_and_crop_face(img)
    if face is None:
        return JSONResponse(content={"error": "No face detected in the image"}, status_code=400)
    
    preprocessed_face = preprocess_image(face)
    prediction = model.recognize_face(preprocessed_face)
    celebrity_name = model.get_celebrity_name(prediction)
    
    return {"celebrity": celebrity_name}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)