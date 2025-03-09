from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import torch
import traceback

from utils import (
    load_segmentation_model,
    load_efficientnet_model,
    load_lin_reg_model,
    process_image_for_inference,
    get_device,
)

app = FastAPI(title="Bone Age Prediction API")

# Add CORS middleware to allow cross-origin requests from frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to hold models and device
device = None
seg_model = None
final_model = None
lin_reg = None

@app.on_event("startup")
async def startup_event():
    global device, seg_model, final_model, lin_reg
    device = get_device()  # returns cuda if available, else cpu
    seg_model = load_segmentation_model(device, model_path="models/deeplab_weights.pth")
    final_model = load_efficientnet_model(device, model_path="models/efficientnet_weights.pth")
    lin_reg = load_lin_reg_model("models/lin_reg_model.pkl")
    print("Models loaded successfully.")

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "Bone Age Prediction API is running."}

@app.post("/predict")
async def predict(file: UploadFile = File(...), gender: float = Form(...)):
    """
    Predict bone age from X-ray image.
    
    - file: The X-ray image file (JPEG, PNG, or TIFF)
    - gender: Gender as a float (1.0 for male, 0.0 for female)
    
    Returns predicted bone age in months.
    """
    # Read image data from uploaded file
    image_bytes = await file.read()
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Invalid image file: {str(e)}"})
    
    try:
        import asyncio
        # Create a task with timeout
        try:
            # Process image and predict age with timeout
            predicted_age = await asyncio.wait_for(
                asyncio.to_thread(process_image_for_inference, image, gender, seg_model, final_model, lin_reg, device),
                timeout=58.0  # 58 seconds timeout (allow 2s for network)
            )
            
            # Return the prediction
            return JSONResponse(content={"bone_age": predicted_age})
        except asyncio.TimeoutError:
            print("Prediction timed out after 58 seconds")
            return JSONResponse(status_code=408, content={"error": "Unable To Process Image: Processing timeout after 60 seconds"})
    except Exception as e:
        print("Prediction error:", str(e))
        print(traceback.format_exc())
        return JSONResponse(status_code=500, content={"error": f"Prediction failed: {str(e)}"})

# For local testing
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
