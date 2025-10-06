import pickle
import numpy as np
from PIL import Image
import os

class BananaPredictor:
    def __init__(self):
        self.yolo_model = None
        self.regression_model = None
        self.models_loaded = False

        # Mapping label sang tên giống chuối
        self.banana_types = {
            0: "Chuối tiêu",
            1: "Chuối sứ",
            2: "Chuối cau",
            3: "Chuối già"
        }

    def load_models(self, yolo_path, regression_path):
        """Load cả 2 models"""
        try:
            # Kiểm tra file tồn tại
            if not os.path.exists(yolo_path):
                return {"success": False, "error": f"YOLO model not found: {yolo_path}"}
            if not os.path.exists(regression_path):
                return {"success": False, "error": f"Regression model not found: {regression_path}"}

            # Load YOLO model
            import torch
            self.yolo_model = torch.load(yolo_path, map_location=torch.device('cpu'))

            # Nếu model là dict với key 'model', extract nó
            if isinstance(self.yolo_model, dict) and 'model' in self.yolo_model:
                self.yolo_model = self.yolo_model['model']

            self.yolo_model.eval()

            # Load regression model
            with open(regression_path, 'rb') as f:
                self.regression_model = pickle.load(f)

            self.models_loaded = True
            return {"success": True, "message": "Models loaded successfully"}

        except Exception as e:
            return {"success": False, "error": f"Load model error: {str(e)}"}

    def preprocess_image(self, image_path, target_size=(224, 224)):
        """Tiền xử lý ảnh"""
        try:
            img = Image.open(image_path).convert('RGB')
            img = img.resize(target_size)
            img_array = np.array(img)
            img_array = img_array / 255.0
            return img_array, img
        except Exception as e:
            raise Exception(f"Image preprocessing error: {str(e)}")

    def predict_banana_type(self, img_pil):
        """Dùng YOLO để phân loại giống chuối"""
        try:
            import torch

            # YOLO model thường nhận PIL Image hoặc numpy array
            # Thử predict trực tiếp
            with torch.no_grad():
                results = self.yolo_model(img_pil)

                # Lấy class với confidence cao nhất
                if hasattr(results, 'xyxy'):
                    # YOLOv5/v8 format
                    pred = results.xyxy[0]
                    if len(pred) > 0:
                        predicted_class = int(pred[0, 5].item())
                    else:
                        predicted_class = 0
                elif hasattr(results, 'boxes'):
                    # YOLOv11 format
                    boxes = results.boxes
                    if len(boxes) > 0:
                        predicted_class = int(boxes.cls[0].item())
                    else:
                        predicted_class = 0
                else:
                    # Fallback
                    predicted_class = 0

            return predicted_class

        except Exception as e:
            print(f"YOLO prediction error: {str(e)}")
            return 0  # Default về class 0 nếu lỗi

    def predict_days_remaining(self, img_array):
        """Dùng regression model để dự đoán số ngày"""
        try:
            # Flatten image cho regression model
            img_flat = img_array.reshape(1, -1)

            # Predict
            days = self.regression_model.predict(img_flat)[0]

            # Đảm bảo không âm
            days = max(0, days)

            return float(days)

        except Exception as e:
            print(f"Regression prediction error: {str(e)}")
            return 3.0  # Default 3 ngày nếu lỗi

    def predict(self, image_path):
        """Dự đoán tổng hợp"""
        try:
            # Kiểm tra models đã load
            if not self.models_loaded:
                return {
                    "success": False,
                    "error": "Models not loaded. Call load_models() first."
                }

            # Kiểm tra file ảnh tồn tại
            if not os.path.exists(image_path):
                return {
                    "success": False,
                    "error": f"Image file not found: {image_path}"
                }

            # Tiền xử lý ảnh
            img_array, img_pil = self.preprocess_image(image_path)

            # Bước 1: Phân loại giống chuối bằng YOLO
            banana_class = self.predict_banana_type(img_pil)
            banana_type = self.banana_types.get(banana_class, f"Loại {banana_class}")

            # Bước 2: Dự đoán số ngày còn lại bằng regression
            days_float = self.predict_days_remaining(img_array)
            days_remaining = int(round(days_float))

            # Phân loại trạng thái dựa trên số ngày
            if days_remaining <= 0:
                status = "Đã hỏng"
                color = "#F44336"
            elif days_remaining <= 2:
                status = "Gần hỏng (1-2 ngày)"
                color = "#FF9800"
            elif days_remaining <= 5:
                status = "Còn tốt (3-5 ngày)"
                color = "#FFC107"
            else:
                status = "Rất tươi (>5 ngày)"
                color = "#4CAF50"

            return {
                "success": True,
                "banana_type": banana_type,
                "banana_class": int(banana_class),
                "days": int(days_remaining),
                "days_exact": round(days_float, 1),
                "status": status,
                "color": color
            }

        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            return {
                "success": False,
                "error": str(e),
                "detail": error_detail
            }

# Instance toàn cục
predictor = BananaPredictor()

def load_models(yolo_path, regression_path):
    """Load models - gọi từ Kotlin"""
    return predictor.load_models(yolo_path, regression_path)

def predict_image(path):
    """Predict image - gọi từ Kotlin"""
    return predictor.predict(path)