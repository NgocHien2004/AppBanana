import pickle
import numpy as np
from PIL import Image

class BananaPredictor:
    def __init__(self):
        self.yolo_model = None
        self.regression_model = None

        # Mapping label sang tên giống chuối (placeholder)
        self.banana_types = {
            0: "Chuối tiêu",
            1: "Chuối sứ",
            2: "Chuối cau",
            3: "Chuối già"
        }

    def load_models(self, yolo_path, regression_path):
        """Load cả 2 models"""
        try:
            # Load YOLO model
            import torch
            self.yolo_model = torch.load(yolo_path, map_location=torch.device('cpu'))
            self.yolo_model.eval()

            # Load regression model
            with open(regression_path, 'rb') as f:
                self.regression_model = pickle.load(f)

            return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def preprocess_image(self, image_path, target_size=(224, 224)):
        """Tiền xử lý ảnh"""
        img = Image.open(image_path).convert('RGB')
        img = img.resize(target_size)
        img_array = np.array(img)
        img_array = img_array / 255.0
        return img_array

    def predict_banana_type(self, img_array):
        """Dùng YOLO để phân loại giống chuối"""
        try:
            import torch

            # Chuyển sang tensor và reshape cho YOLO
            img_tensor = torch.from_numpy(img_array).float()
            img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # BHWC -> BCHW

            with torch.no_grad():
                output = self.yolo_model(img_tensor)

                # Lấy class prediction (giả sử output shape là [1, num_classes])
                predicted_class = int(torch.argmax(output).item())

            return predicted_class
        except Exception as e:
            return 0  # Default về class 0 nếu lỗi

    def predict_days_remaining(self, img_array):
        """Dùng regression model để dự đoán số ngày"""
        try:
            # Flatten image cho regression model
            img_flat = img_array.reshape(1, -1)

            # Predict
            days = self.regression_model.predict(img_flat)[0]

            return float(days)
        except Exception as e:
            return 0.0

    def predict(self, image_path):
        """Dự đoán tổng hợp"""
        try:
            # Tiền xử lý ảnh
            img_array = self.preprocess_image(image_path)

            # Bước 1: Phân loại giống chuối bằng YOLO
            banana_class = self.predict_banana_type(img_array)
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
                "banana_class": banana_class,
                "days": days_remaining,
                "days_exact": round(days_float, 1),
                "status": status,
                "color": color
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

# Instance toàn cục
predictor = BananaPredictor()

def load_models(yolo_path, regression_path):
    return predictor.load_models(yolo_path, regression_path)

def predict_image(path):
    return predictor.predict(path)