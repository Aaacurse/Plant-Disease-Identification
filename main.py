from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
import uvicorn
from PIL import Image
from io import BytesIO
import traceback
from fastapi import HTTPException

app = FastAPI()

origins=[ 
"*"
] 

app.add_middleware( 
CORSMiddleware, 
allow_origins=origins, 
allow_credentials=True, 
    allow_methods=["*"], 
    allow_headers=["*"], 
)

class_names = ['Healthy', 'Rust', 'Scab']

try:
    model = tf.saved_model.load(r"D:\Aakarsh\Coding\Plant Disease Identification\model")
    infer = model.signatures['serving_default'] 
    model_loaded = True
    print("Model loaded successfully!")
except Exception as e:
    model_loaded = False
    load_error = str(e)
    print(f"Failed to load model: {load_error}")


def preprocess_image(image_data) -> tf.Tensor:
    image = Image.open(BytesIO(image_data))
    image = image.resize((256, 256))
    image = tf.convert_to_tensor(np.array(image), dtype=tf.float32)
    image = tf.expand_dims(image, 0)
    return image


@app.get("/")
async def root():
    return {"message": "Plant Disease Detection API is running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not model_loaded:
        return {"error": f"Failed to load the model: {load_error}"}
    
    try:
        file_content = await file.read()
        allowed_types = ["image/jpeg", "image/png"]
        if file.content_type not in allowed_types:
            raise HTTPException(status_code=400, detail="Only JPEG and PNG files are allowed")
        if not file_content:
            return {"error": "Could not read the file or file is empty"}
        
        processed_image = preprocess_image(file_content)        
        prediction = infer(processed_image)
        
        print(f"Available output keys: {list(prediction.keys())}")
        
        if 'output_0' in prediction:
            output_tensor = 'output_0'
        else:
            output_tensor = list(prediction.keys())[0]
            print(f"Using '{output_tensor}' as output key")
        
        prediction_array = prediction[output_tensor].numpy()
        pred_index = np.argmax(prediction_array[0])
        pred_class = class_names[pred_index]
        confidence = round(float(np.max(prediction_array[0])),4)
        
        return {
            "class": pred_class,
            "confidence": str(confidence)+"%",
        }
    
    except Exception as e:
        error_traceback = traceback.format_exc()
        print(f"Error in prediction: {error_traceback}")
        return {"error": str(e), "details": error_traceback}


if __name__ == "__main__":
    uvicorn.run(app=app, host="localhost", port=8000)