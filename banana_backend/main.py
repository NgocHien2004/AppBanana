from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
from pathlib import Path
from predictor import BananaPredictor
import uvicorn

# Khởi tạo FastAPI
app = FastAPI(
    title="Banana Prediction API",
    description="API dự đoán thời hạn sử dụng chuối với 120+ features",
    version="3.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Khởi tạo predictor
MODELS_DIR = Path("models")
UPLOADS_DIR = Path("uploads")
UPLOADS_DIR.mkdir(exist_ok=True)

# Model paths
YOLO_MODEL_PATH = MODELS_DIR / "yolov11.pt"
PKL_MODEL_PATH = MODELS_DIR / "regression.pkl"  # Model với 120+ features

# Kiểm tra files
if not YOLO_MODEL_PATH.exists():
    print(f"❌ YOLO model not found: {YOLO_MODEL_PATH}")
    print(f"📁 Please place your YOLOv11 model at: {YOLO_MODEL_PATH.absolute()}")
    
if not PKL_MODEL_PATH.exists():
    print(f"❌ PKL model not found: {PKL_MODEL_PATH}")
    print(f"📁 Please place your regression.pkl at: {PKL_MODEL_PATH.absolute()}")

# Load models
try:
    predictor = BananaPredictor(
        yolo_path=str(YOLO_MODEL_PATH),
        pkl_path=str(PKL_MODEL_PATH)
    )
    print("✅ Server ready with 120+ features extraction!")
except Exception as e:
    print(f"❌ Failed to load models: {e}")
    predictor = None

@app.get("/")
def root():
    """Root endpoint"""
    return {
        "message": "🍌 Banana Prediction API v3.0",
        "version": "3.0.0",
        "features": [
            "YOLO banana detection",
            "120+ visual features (ResNet50, MobileNetV2, Classical)",
            "Ensemble regression prediction",
            "Auto error on no banana"
        ],
        "endpoints": {
            "predict": "/predict [POST]",
            "health": "/health [GET]",
            "docs": "/docs [GET]"
        }
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": predictor is not None,
        "yolo_model": str(YOLO_MODEL_PATH.exists()),
        "pkl_model": str(PKL_MODEL_PATH.exists()),
        "deep_learning": predictor is not None and hasattr(predictor, 'resnet') and predictor.resnet is not None
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Endpoint dự đoán với YOLO detection + 120+ features
    
    FLOW:
    1. Kiểm tra file type
    2. YOLO detect chuối
    3. Nếu KHÔNG có chuối → return error
    4. Nếu CÓ chuối → extract 120+ features → predict shelf life
    
    Features extracted:
    - Deep Learning: ResNet50 (36 features) + MobileNetV2 (31 features)
    - Classical: Color (HSV, LAB) + Texture (GLCM, LBP, Gradient) + Shape
    - Total: ~120-130 features
    
    Parameters:
    - file: Image file (jpg, png, jpeg)
    
    Returns:
    - JSON with prediction results or error message
    """
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Check server logs."
        )
    
    try:
        # Kiểm tra file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type: {file.content_type}. Please upload an image."
            )
        
        # Lưu file tạm
        file_path = UPLOADS_DIR / file.filename
        print(f"\n📥 Receiving file: {file.filename}")
        
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        print(f"💾 Saved to: {file_path}")
        
        # Dự đoán (bao gồm YOLO detection + 120+ features)
        result = predictor.predict(str(file_path))
        
        # Có thể xóa file tạm
        # file_path.unlink()
        
        return result
    
    except HTTPException:
        raise
    
    except Exception as e:
        print(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("🚀 Starting Banana Prediction API Server v3.0...")
    print(f"📁 Models directory: {MODELS_DIR.absolute()}")
    print(f"📁 Uploads directory: {UPLOADS_DIR.absolute()}")
    print("\n⚙️  FEATURES:")
    print("   - YOLO detection (detect banana first)")
    print("   - 120+ features extraction")
    print("   - Deep learning: ResNet50 + MobileNetV2")
    print("   - Classical: Color + Texture + Shape")
    print("   - Ensemble regression prediction\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )