import io
import os
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
import tensorflow as tf

app = FastAPI(title="AgroScan API")

# Setup paths (files should be in the same directory as this script)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.tflite")
LABELS_PATH = os.path.join(BASE_DIR, "labels.txt")

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load labels
try:
    with open(LABELS_PATH, "r") as f:
        labels = [line.strip() for line in f.readlines() if line.strip()]
except Exception as e:
    print(f"Error loading labels: {e}")
    labels = []

@app.get("/")
def read_root():
    return {"message": "Welcome to AgroScan API. Send a POST request with an image to /analyze"}

@app.post("/predict")
async def analyze_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # 1. Preprocess Image
        image_150 = image.resize((150, 150))
        input_data = np.array(image_150, dtype=np.float32) / 255.0
        input_data = np.expand_dims(input_data, axis=0)
        
        # 2. Gatekeeper Check (Color Heuristic)
        # Convert a smaller version to HSV to check for plant colors (Green/Yellow/Brown)
        hsv_image = image.resize((100, 100)).convert('HSV')
        np_hsv = np.array(hsv_image)
        h = np_hsv[:,:,0]
        s = np_hsv[:,:,1]
        v = np_hsv[:,:,2]
        
        is_green = (h >= 30) & (h <= 110) & (s >= 30) & (v >= 30)
        is_yellow_brown = (h >= 10) & (h < 30) & (s >= 30) & (v >= 20)
        plant_ratio = np.sum(is_green | is_yellow_brown) / 10000.0
        
        # 3. Run inference
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])[0]
        
        max_index = np.argmax(output_data)
        max_confidence = float(output_data[max_index])
        
        if max_index >= len(labels):
            raise ValueError("Predicted index out of range for labels.")
            
        full_label = labels[max_index]
        confidence_percent = round(max_confidence * 100, 1)
        
        # إذا لم يكن بالصورة أي ألوان نباتية (أقل من 5%)، نطلب نسبة تأكد عالية جداً (90%) لكي نقبلها
        if plant_ratio < 0.05:
            if confidence_percent < 90.0:
                raise HTTPException(status_code=400, detail="الصورة غير واضحة أو لا تبدو كأنها ورقة نبات. يرجى التقاط صورة أوضح.")
        else:
            # إذا كانت ألوانها ألوان نبات، نقبلها حتى لو نسبة التأكد متوسطة (لأن النبات المريض قد تقل نسبة التأكد فيه)
            if confidence_percent < 40.0:
                raise HTTPException(status_code=400, detail="نسبة دقة التعرف ضعيفة جداً، يرجى التقاط صورة أقرب للورقة.")
        
        plant_name = "Unknown"
        disease_name = full_label
        is_healthy = False
        
        if '___' in full_label:
            parts = full_label.split('___')
            plant_name = parts[0].replace('_', ' ').replace('(', '').replace(')', '')
            disease_raw = parts[1].replace('_', ' ')
            is_healthy = 'healthy' in disease_raw.lower()
            disease_name = "Sleem (سليم)" if is_healthy else disease_raw
        else:
            plant_name = full_label
            is_healthy = 'healthy' in full_label.lower()
            
        treatment_msg = "الري بانتظام ومراقبة النبات." if is_healthy else "يرجى عزل النبات واستشارة مختص."
        
        return {
            "plantName": plant_name,
            "diseaseName": disease_name,
            "confidence": str(confidence_percent),
            "isHealthy": is_healthy,
            "treatment": treatment_msg
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
