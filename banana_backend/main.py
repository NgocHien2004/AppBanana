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
    description="API dự đoán thời hạn sử dụng chuối",
    version="1.0.0"
)

# CORS để Flutter có thể gọi
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

# QUAN TRỌNG: Thay đổi tên file model của bạn
YOLO_MODEL_PATH = MODELS_DIR / "yolov11.pt"  # Đổi tên nếu khác
PKL_MODEL_PATH = MODELS_DIR / "model.pkl"    # Đổi tên nếu khác

# Kiểm tra file model có tồn tại không
if not YOLO_MODEL_PATH.exists():
    print(f"❌ YOLO model not found: {YOLO_MODEL_PATH}")
    print(f"📁 Please place your YOLOv11 model at: {YOLO_MODEL_PATH.absolute()}")
    
if not PKL_MODEL_PATH.exists():
    print(f"❌ PKL model not found: {PKL_MODEL_PATH}")
    print(f"📁 Please place your PKL model at: {PKL_MODEL_PATH.absolute()}")

# Load models
try:
    predictor = BananaPredictor(
        yolo_path=str(YOLO_MODEL_PATH),
        pkl_path=str(PKL_MODEL_PATH)
    )
    print("✅ Server ready!")
except Exception as e:
    print(f"❌ Failed to load models: {e}")
    predictor = None

@app.get("/")
def root():
    """Root endpoint"""
    return {
        "message": "🍌 Banana Prediction API is running!",
        "version": "1.0.0",
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
        "pkl_model": str(PKL_MODEL_PATH.exists())
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Endpoint dự đoán
    
    Parameters:
    - file: Image file (jpg, png, jpeg)
    
    Returns:
    - JSON with prediction results
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
        
        # Dự đoán
        result = predictor.predict(str(file_path))
        
        # Có thể xóa file tạm sau khi dự đoán (optional)
        # file_path.unlink()
        
        return result
    
    except Exception as e:
        print(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("🚀 Starting Banana Prediction API Server...")
    print(f"📁 Models directory: {MODELS_DIR.absolute()}")
    print(f"📁 Uploads directory: {UPLOADS_DIR.absolute()}")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )