import os
import cv2
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict
from ultralytics import YOLO
import traceback

class BananaPredictor:
    """
    Predictor kết hợp YOLO (phân loại giống) + Regression (dự đoán shelf life)
    """
    
    def __init__(self, yolo_path: str, pkl_path: str):
        """
        Khởi tạo models
        
        Args:
            yolo_path: Path to YOLOv11 model (.pt)
            pkl_path: Path to regression ensemble model (.pkl)
        """
        print("🔄 Loading models...")
        
        # Load YOLO model
        try:
            self.yolo_model = YOLO(yolo_path)
            print(f"✅ YOLO model loaded: {yolo_path}")
        except Exception as e:
            print(f"❌ Error loading YOLO: {e}")
            traceback.print_exc()
            raise
        
        # Load regression ensemble model
        try:
            self.regression_model = joblib.load(pkl_path)
            print(f"✅ Regression model loaded: {pkl_path}")
            
            # Check model structure
            if isinstance(self.regression_model, dict):
                if 'models' in self.regression_model and 'weights' in self.regression_model:
                    print(f"   📊 Ensemble model with {len(self.regression_model['models'])} base models")
                    print(f"   ⚖️ Model weights: {[f'{w:.3f}' for w in self.regression_model['weights']]}")
                else:
                    raise ValueError("Invalid ensemble model format")
            else:
                print(f"   📊 Single model loaded")
                
        except Exception as e:
            print(f"❌ Error loading regression model: {e}")
            traceback.print_exc()
            raise
        
        print("✅ All models loaded successfully!")
        
        # Class mapping (YOLO classes)
        self.banana_types = {
            0: "Chuối tiêu cao (Cao-Le)",
            1: "Chuối tiêu thấp (Cao-Nai)",
            2: "Chuối xiêm cao (Xiem-Le)",
            3: "Chuối xiêm thấp (Xiem-Nai)"
        }
        
        # Feature columns (SAME AS TRAINING)
        self.feature_columns = [
            'hue_mean', 'saturation_mean', 'value_mean',
            'a_mean', 'b_mean', 'yellowness_index',
            'gradient_mean', 'texture_contrast'
        ]
        
        print(f"📋 Class mapping: {self.banana_types}")
        print(f"🔧 Expected features: {self.feature_columns}")
    
    def extract_visual_features(self, image_path: str) -> Dict:
        """
        Extract same features as training data
        COPIED FROM test_regression.py
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None or image.size == 0:
            raise ValueError(f"Cannot load image: {image_path}")
        
        try:
            # Resize to standard size (224x224)
            image = cv2.resize(image, (224, 224))
            
            # Convert color spaces
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            features = {}
            
            # Core color features (same as training)
            features['hue_mean'] = float(np.mean(hsv[:,:,0]))
            features['saturation_mean'] = float(np.mean(hsv[:,:,1]))
            features['value_mean'] = float(np.mean(hsv[:,:,2]))
            features['a_mean'] = float(np.mean(lab[:,:,1]))  # Green-Red axis
            features['b_mean'] = float(np.mean(lab[:,:,2]))  # Blue-Yellow axis
            
            # Yellowness index (MOST IMPORTANT)
            features['yellowness_index'] = features['b_mean'] - features['a_mean']
            
            # Texture features
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            features['gradient_mean'] = float(np.mean(gradient_magnitude))
            features['texture_contrast'] = float(np.std(gray))
            
            print(f"   🔍 Features extracted:")
            print(f"      Yellowness: {features['yellowness_index']:.2f}")
            print(f"      Hue: {features['hue_mean']:.2f}")
            print(f"      Saturation: {features['saturation_mean']:.2f}")
            print(f"      Texture: {features['texture_contrast']:.2f}")
            
            return features
            
        except Exception as e:
            print(f"❌ Error extracting features: {e}")
            traceback.print_exc()
            raise
    
    def predict_banana_type(self, image_path: str, conf_threshold=0.25) -> tuple:
        """
        Predict banana type using YOLO
        
        Returns:
            tuple: (class_id, confidence)
        """
        try:
            print(f"🔍 Running YOLO prediction...")
            
            results = self.yolo_model.predict(
                source=image_path,
                conf=conf_threshold,
                iou=0.7,
                verbose=False,
                save=False
            )
            
            # Get detection with highest confidence
            if len(results) > 0 and hasattr(results[0], 'boxes') and len(results[0].boxes) > 0:
                boxes = results[0].boxes
                confidences = boxes.conf.cpu().numpy()
                classes = boxes.cls.cpu().numpy().astype(int)
                
                # Get highest confidence detection
                max_idx = confidences.argmax()
                predicted_class = int(classes[max_idx])
                confidence = float(confidences[max_idx])
                
                print(f"   ✅ YOLO: class={predicted_class}, conf={confidence:.3f}")
                return predicted_class, confidence
            else:
                print("   ⚠️ No banana detected, using default class 0")
                return 0, 0.5
                
        except Exception as e:
            print(f"❌ YOLO prediction error: {e}")
            traceback.print_exc()
            return 0, 0.5
    
    def predict_shelf_life(self, features: Dict) -> tuple:
        """
        Predict shelf life using ensemble regression model
        LOGIC FROM test_regression.py
        
        Returns:
            tuple: (prediction, individual_predictions)
        """
        try:
            print(f"🔍 Running shelf life prediction...")
            
            # Convert to DataFrame with correct column order
            feature_df = pd.DataFrame([features])[self.feature_columns]
            
            print(f"   Input shape: {feature_df.shape}")
            print(f"   Features: {list(feature_df.columns)}")
            
            # Make prediction with ensemble
            if isinstance(self.regression_model, dict):
                # Ensemble prediction
                predictions = []
                for model in self.regression_model['models']:
                    pred = model.predict(feature_df)[0]
                    predictions.append(pred)
                
                # Weighted average
                prediction = np.average(predictions, weights=self.regression_model['weights'])
                individual_preds = predictions
                
                print(f"   🤖 Individual predictions: {[f'{p:.2f}' for p in predictions]}")
                print(f"   📊 Ensemble result: {prediction:.2f} days")
            else:
                # Single model prediction
                prediction = self.regression_model.predict(feature_df)[0]
                individual_preds = [prediction]
                
                print(f"   📊 Prediction: {prediction:.2f} days")
            
            # Ensure non-negative
            prediction = max(0, float(prediction))
            
            return prediction, [float(p) for p in individual_preds]
            
        except Exception as e:
            print(f"❌ Shelf life prediction error: {e}")
            traceback.print_exc()
            raise
    
    def get_freshness_status(self, prediction: float) -> Dict:
        """
        Determine freshness status
        LOGIC FROM test_regression.py
        
        Returns:
            dict: status, color, recommendation
        """
        if prediction >= 4.5:
            return {
                "status": "🟢 Rất tươi",
                "color": "#4CAF50",
                "recommendation": "Chuối rất tươi, có thể bảo quản lâu. Tốt nhất trong vài ngày tới."
            }
        elif prediction >= 3.5:
            return {
                "status": "🟢 Tươi",
                "color": "#8BC34A",
                "recommendation": "Chuối còn tươi tốt, nên dùng trong 3-4 ngày."
            }
        elif prediction >= 2.5:
            return {
                "status": "🟡 Khá tốt",
                "color": "#FFC107",
                "recommendation": "Chuối vẫn ổn, nên ăn trong 2-3 ngày."
            }
        elif prediction >= 1.5:
            return {
                "status": "🟠 Chín - Nên dùng sớm",
                "color": "#FF9800",
                "recommendation": "Chuối đã chín, tốt nhất nên ăn trong 1-2 ngày."
            }
        elif prediction >= 0.5:
            return {
                "status": "🔴 Rất chín - Dùng ngay",
                "color": "#F44336",
                "recommendation": "Chuối rất chín, cần dùng ngay hoặc làm sinh tố/nướng."
            }
        else:
            return {
                "status": "🔴 Quá chín",
                "color": "#D32F2F",
                "recommendation": "Chuối đã quá chín, nên cân nhắc loại bỏ."
            }
    
    def predict(self, image_path: str) -> Dict:
        """
        Complete prediction pipeline
        
        Args:
            image_path: Path to banana image
            
        Returns:
            dict: Prediction results
        """
        try:
            print(f"\n{'='*60}")
            print(f"🍌 PROCESSING: {os.path.basename(image_path)}")
            print(f"{'='*60}")
            
            # Step 1: Classify banana type (YOLO)
            banana_class, yolo_confidence = self.predict_banana_type(image_path)
            banana_type = self.banana_types.get(banana_class, f"Loại {banana_class}")
            
            # Step 2: Extract visual features
            features = self.extract_visual_features(image_path)
            
            # Step 3: Predict shelf life
            days_float, individual_preds = self.predict_shelf_life(features)
            days_remaining = int(round(days_float))
            
            # Step 4: Determine freshness status
            freshness = self.get_freshness_status(days_float)
            
            # Build result
            result = {
                "success": True,
                "banana_type": banana_type,
                "banana_class": int(banana_class),
                "yolo_confidence": round(yolo_confidence, 3),
                "days": int(days_remaining),
                "days_exact": round(days_float, 1),
                "status": freshness["status"],
                "color": freshness["color"],
                "recommendation": freshness["recommendation"],
                "individual_predictions": individual_preds,
                "key_features": {
                    "yellowness_index": round(features['yellowness_index'], 2),
                    "hue_mean": round(features['hue_mean'], 2),
                    "saturation_mean": round(features['saturation_mean'], 2),
                    "texture_contrast": round(features['texture_contrast'], 2)
                }
            }
            
            print(f"\n✅ PREDICTION SUCCESS:")
            print(f"   🍌 Type: {banana_type}")
            print(f"   📅 Days: {days_remaining} ({days_float:.1f})")
            print(f"   🎯 Status: {freshness['status']}")
            print(f"   💡 {freshness['recommendation']}")
            print(f"{'='*60}\n")
            
            return result
        
        except Exception as e:
            error_msg = f"Prediction error: {str(e)}"
            print(f"❌ {error_msg}")
            traceback.print_exc()
            
            return {
                "success": False,
                "error": error_msg,
                "error_detail": traceback.format_exc()
            }