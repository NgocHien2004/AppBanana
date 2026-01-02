"""
TEST MODEL - CHÍNH XÁC 100%
Extract đúng features model cần
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings('ignore')

# Deep Learning
try:
    from tensorflow.keras.applications import ResNet50, MobileNetV2
    from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
    from tensorflow.keras.models import Model
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    DEEP_LEARNING = True
except:
    DEEP_LEARNING = False
    print("⚠️ TensorFlow not available")

from scipy import stats
from skimage import feature
from skimage.feature import graycomatrix, graycoprops


class BananaPredictor:
    """Predictor với features CHÍNH XÁC"""
    
    def __init__(self, model_path=r'D:\flutter_project\banana_backend\models\regression.pkl'):
        print("=" * 80)
        print("🍌 BANANA SHELF LIFE PREDICTOR")
        print("=" * 80)
        
        # Load model
        print(f"📂 Loading: {model_path}")
        with open(model_path, 'rb') as f:
            saved_data = pickle.load(f)
        
        self.models = saved_data['models']
        self.weights = saved_data['weights']
        self.scaler = saved_data['scaler']
        self.selected_features = saved_data['features']
        
        print(f"✓ Models: {len(self.models) if isinstance(self.models, dict) else len(self.models)}")
        print(f"✓ Features: {len(self.selected_features)}")
        
        # Load deep learning
        if DEEP_LEARNING:
            print("🔥 Loading ResNet50 & MobileNetV2...")
            base_resnet = ResNet50(weights='imagenet', include_top=False, pooling='avg')
            self.resnet = Model(inputs=base_resnet.input, outputs=base_resnet.output)
            
            base_mobile = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
            self.mobilenet = Model(inputs=base_mobile.input, outputs=base_mobile.output)
            print("✓ Deep learning loaded")
        else:
            self.resnet = None
            self.mobilenet = None
        
        print()
    
    def extract_deep_features(self, img):
        """Extract CHÍNH XÁC deep features model cần"""
        if not DEEP_LEARNING or self.resnet is None:
            return {}
        
        features = {}
        
        # RGB
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        elif img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        img_resized = cv2.resize(img, (224, 224))
        img_batch = np.expand_dims(img_resized, axis=0)
        
        # ResNet50 - CHỈ LẤY: f10, f25, f26, f27, f28, f29
        img_resnet = resnet_preprocess(img_batch.copy())
        resnet_feat = self.resnet.predict(img_resnet, verbose=0)[0]
        
        top_indices = np.argsort(np.abs(resnet_feat))[-30:]  # Top 30
        
        # Map theo index model cần: 10,25,26,27,28,29
        resnet_mapping = {10: 10, 25: 25, 26: 26, 27: 27, 28: 28, 29: 29}
        for model_idx in resnet_mapping.keys():
            if model_idx < len(top_indices):
                feat_idx = top_indices[model_idx]
                features[f'resnet_f{model_idx}'] = float(resnet_feat[feat_idx])
        
        features['resnet_mean'] = float(np.mean(resnet_feat))
        features['resnet_std'] = float(np.std(resnet_feat))
        features['resnet_max'] = float(np.max(resnet_feat))
        features['resnet_median'] = float(np.median(resnet_feat))
        features['resnet_energy'] = float(np.sum(resnet_feat**2))
        
        # MobileNetV2 - LẤY: f0-14, f23, f24 (17 features)
        img_mobile = mobilenet_preprocess(img_batch.copy())
        mobile_feat = self.mobilenet.predict(img_mobile, verbose=0)[0]
        
        top_indices = np.argsort(np.abs(mobile_feat))[-25:]  # Top 25
        
        # f0-14
        for i in range(15):
            if i < len(top_indices):
                feat_idx = top_indices[i]
                features[f'mobile_f{i}'] = float(mobile_feat[feat_idx])
        
        # f23, f24
        if 23 < len(top_indices):
            features['mobile_f23'] = float(mobile_feat[top_indices[23]])
        if 24 < len(top_indices):
            features['mobile_f24'] = float(mobile_feat[top_indices[24]])
        
        features['mobile_mean'] = float(np.mean(mobile_feat))
        features['mobile_std'] = float(np.std(mobile_feat))
        features['mobile_max'] = float(np.max(mobile_feat))
        features['mobile_median'] = float(np.median(mobile_feat))
        features['mobile_energy'] = float(np.sum(mobile_feat**2))
        
        features['resnet_mobile_corr'] = float(np.corrcoef(
            resnet_feat[:100], mobile_feat[:100]
        )[0, 1] if len(resnet_feat) >= 100 and len(mobile_feat) >= 100 else 0)
        
        return features
    
    def extract_classical_features(self, img):
        """Extract classical features"""
        features = {}
        
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            
            h, s, v = img_hsv[:, :, 0], img_hsv[:, :, 1], img_hsv[:, :, 2]
            total_pixels = h.size
            
            # HSV - KHÔNG CẦN hue_min, saturation_min, value_min
            for channel, name in [(h, 'hue'), (s, 'saturation'), (v, 'value')]:
                features[f'{name}_mean'] = float(np.mean(channel))
                features[f'{name}_std'] = float(np.std(channel))
                features[f'{name}_median'] = float(np.median(channel))
                features[f'{name}_max'] = float(np.max(channel))
                features[f'{name}_range'] = float(np.max(channel) - np.min(channel))
                features[f'{name}_p25'] = float(np.percentile(channel, 25))
                features[f'{name}_p75'] = float(np.percentile(channel, 75))
                features[f'{name}_iqr'] = float(np.percentile(channel, 75) - np.percentile(channel, 25))
            
            # value_min CẦN (kiểm tra list)
            features['value_min'] = float(np.min(v))
            features['hue_max'] = float(np.max(h))
            features['value_max'] = float(np.max(v))
            
            # LAB
            l, a, b = img_lab[:, :, 0], img_lab[:, :, 1], img_lab[:, :, 2]
            features['l_mean'] = float(np.mean(l))
            features['l_std'] = float(np.std(l))
            features['l_range'] = float(np.max(l) - np.min(l))
            features['a_mean'] = float(np.mean(a))
            features['a_std'] = float(np.std(a))
            features['a_range'] = float(np.max(a) - np.min(a))
            features['b_mean'] = float(np.mean(b))
            features['b_std'] = float(np.std(b))
            features['b_range'] = float(np.max(b) - np.min(b))
            features['yellowness_index'] = float(features['b_mean'] - features['a_mean'])
            
            # Color ratios
            brown_mask_strict = (h >= 10) & (h <= 30) & (s >= 50) & (v >= 20) & (v <= 150)
            brown_mask_medium = (h >= 8) & (h <= 35) & (s >= 40) & (v >= 15) & (v <= 180)
            brown_mask_loose = (h >= 5) & (h <= 40) & (s >= 30) & (v >= 10) & (v <= 200)
            
            features['brown_ratio_strict'] = float(np.sum(brown_mask_strict) / total_pixels)
            features['brown_ratio_medium'] = float(np.sum(brown_mask_medium) / total_pixels)
            features['brown_ratio_loose'] = float(np.sum(brown_mask_loose) / total_pixels)
            
            black_mask_strict = (s <= 50) & (v <= 50)
            black_mask_medium = (s <= 70) & (v <= 80)
            black_mask_loose = (s <= 100) & (v <= 120)
            
            features['black_ratio_strict'] = float(np.sum(black_mask_strict) / total_pixels)
            features['black_ratio_medium'] = float(np.sum(black_mask_medium) / total_pixels)
            features['black_ratio_loose'] = float(np.sum(black_mask_loose) / total_pixels)
            
            features['dark_ratio_v80'] = float(np.sum(v <= 80) / total_pixels)
            features['dark_ratio_v100'] = float(np.sum(v <= 100) / total_pixels)
            features['dark_ratio_v120'] = float(np.sum(v <= 120) / total_pixels)
            
            features['bright_ratio_v180'] = float(np.sum(v >= 180) / total_pixels)
            features['bright_ratio_v200'] = float(np.sum(v >= 200) / total_pixels)
            features['bright_ratio_v220'] = float(np.sum(v >= 220) / total_pixels)
            
            for sat_thresh in [15, 20, 30]:
                mask = s >= sat_thresh
                if np.sum(mask) > 0:
                    h_filtered = h[mask]
                    features[f'green_ratio_s{sat_thresh}'] = float(np.sum((h_filtered >= 35) & (h_filtered <= 85)) / total_pixels)
                    features[f'yellow_ratio_s{sat_thresh}'] = float(np.sum((h_filtered >= 20) & (h_filtered <= 35)) / total_pixels)
                    features[f'orange_ratio_s{sat_thresh}'] = float(np.sum((h_filtered >= 10) & (h_filtered <= 20)) / total_pixels)
                    features[f'red_ratio_s{sat_thresh}'] = float(np.sum(((h_filtered >= 0) & (h_filtered <= 10)) | (h_filtered >= 170)) / total_pixels)
                else:
                    features[f'green_ratio_s{sat_thresh}'] = 0.0
                    features[f'yellow_ratio_s{sat_thresh}'] = 0.0
                    features[f'orange_ratio_s{sat_thresh}'] = 0.0
                    features[f'red_ratio_s{sat_thresh}'] = 0.0
            
            gray_mask = s <= 30
            features['gray_ratio'] = float(np.sum(gray_mask) / total_pixels)
            
            # Texture
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient = np.sqrt(grad_x**2 + grad_y**2)
            
            features['gradient_mean'] = float(np.mean(gradient))
            features['gradient_std'] = float(np.std(gradient))
            features['gradient_max'] = float(np.max(gradient))
            
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            features['laplacian_mean'] = float(np.mean(np.abs(laplacian)))
            features['laplacian_std'] = float(np.std(laplacian))
            features['laplacian_var'] = float(np.var(laplacian))
            
            lbp = feature.local_binary_pattern(gray, 8, 1, method='uniform')
            features['lbp_mean'] = float(np.mean(lbp))
            features['lbp_std'] = float(np.std(lbp))
            
            for low, high in [(30, 100), (50, 150), (70, 200)]:
                edges = cv2.Canny(gray, low, high)
                features[f'edge_density_{low}_{high}'] = float(np.sum(edges > 0) / total_pixels)
            
            # GLCM
            gray_8bit = (gray / gray.max() * 255).astype(np.uint8) if gray.max() > 0 else gray.astype(np.uint8)
            gray_rescaled = (gray_8bit / 16).astype(np.uint8)
            glcm = graycomatrix(gray_rescaled, [1], [0], 16, symmetric=True, normed=True)
            
            features['haralick_contrast_mean'] = float(graycoprops(glcm, 'contrast')[0, 0])
            features['haralick_dissimilarity_mean'] = float(graycoprops(glcm, 'dissimilarity')[0, 0])
            features['haralick_homogeneity_mean'] = float(graycoprops(glcm, 'homogeneity')[0, 0])
            features['haralick_energy_mean'] = float(graycoprops(glcm, 'energy')[0, 0])
            
            # Spatial
            h_img, w_img = img.shape[:2]
            center_h, center_w = h_img // 2, w_img // 2
            margin = min(h_img, w_img) // 4
            
            center_region = img_hsv[center_h-margin:center_h+margin, center_w-margin:center_w+margin]
            edge_region_top = img_hsv[:margin, :]
            edge_region_bottom = img_hsv[-margin:, :]
            edge_region = np.vstack([edge_region_top, edge_region_bottom])
            
            if center_region.size > 0:
                features['center_value_mean'] = float(np.mean(center_region[:, :, 2]))
                features['center_sat_mean'] = float(np.mean(center_region[:, :, 1]))
            else:
                features['center_value_mean'] = 0.0
                features['center_sat_mean'] = 0.0
            
            if edge_region.size > 0:
                features['edge_value_mean'] = float(np.mean(edge_region[:, :, 2]))
            else:
                features['edge_value_mean'] = 0.0
            
            features['center_edge_diff'] = features['center_value_mean'] - features['edge_value_mean']
            
            for thresh in [50, 60, 70]:
                _, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)
                features[f'spot_ratio_t{thresh}'] = float(np.sum(binary > 0) / total_pixels)
            
            # Stats
            features['hue_skew'] = float(stats.skew(h.flatten()))
            features['hue_kurtosis'] = float(stats.kurtosis(h.flatten()))
            features['saturation_skew'] = float(stats.skew(s.flatten()))
            features['saturation_kurtosis'] = float(stats.kurtosis(s.flatten()))
            features['value_skew'] = float(stats.skew(v.flatten()))
            features['value_kurtosis'] = float(stats.kurtosis(v.flatten()))
            
        except Exception as e:
            print(f"⚠️ Error: {e}")
        
        return features
    
    def predict(self, image_path):
        """Predict"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Cannot load: {image_path}")
            
            img = cv2.resize(img, (224, 224))
            
            # Extract
            classical = self.extract_classical_features(img)
            
            if DEEP_LEARNING:
                deep = self.extract_deep_features(img)
                features = {**classical, **deep}
            else:
                features = classical
            
            # Prepare DataFrame
            feature_df = pd.DataFrame([features])
            
            # CRITICAL: Scaler expects 156 features (all features extracted during training)
            # But model only uses 120 selected features
            # Solution: Create DataFrame with ALL features scaler expects, then select 120
            
            # Get feature names scaler was trained on (if available)
            if hasattr(self.scaler, 'feature_names_in_'):
                scaler_features = list(self.scaler.feature_names_in_)
            else:
                # Scaler doesn't have feature names, create generic names
                scaler_features = [f'feature_{i}' for i in range(self.scaler.n_features_in_)]
            
            # Create full DataFrame with all scaler features
            full_feature_df = pd.DataFrame(0.0, index=[0], columns=scaler_features)
            
            # Fill with extracted features
            for col in feature_df.columns:
                if col in full_feature_df.columns:
                    full_feature_df[col] = feature_df[col].values[0]
            
            # Scale ALL features
            features_scaled_all = self.scaler.transform(full_feature_df)
            
            # Select only the 120 features model needs
            selected_indices = [scaler_features.index(feat) if feat in scaler_features else 0 
                               for feat in self.selected_features]
            features_for_model = features_scaled_all[:, selected_indices]
            
            # Predict
            predictions = []
            if isinstance(self.models, dict):
                for model in self.models.values():
                    pred = model.predict(features_for_model)[0]
                    predictions.append(float(pred))
            else:
                for model in self.models:
                    pred = model.predict(features_for_model)[0]
                    predictions.append(float(pred))
            
            # Ensemble
            if isinstance(self.weights, dict):
                weights = list(self.weights.values())
            else:
                weights = list(self.weights[:len(predictions)])
            
            days = np.average(predictions, weights=weights)
            days = max(0, float(days))
            
            # Calculate confidence metrics
            pred_std = np.std(predictions)
            pred_variance = np.var(predictions)
            
            # Individual model confidence (based on distance from thresholds)
            thresholds = [0.5, 1.5, 2.5, 3.5, 4.5]
            individual_confidences = []
            
            for pred in predictions:
                # Find closest threshold
                distances = [abs(pred - t) for t in thresholds]
                min_distance = min(distances)
                
                # Confidence: higher distance from threshold = higher confidence
                # Map distance to 0-100% (0.5 distance = 50%, 1.0+ = 100%)
                confidence = min(100, min_distance * 100)
                individual_confidences.append(confidence)
            
            # ============================================================
            # CONFIDENCE FILTER: Top 3 models with confidence > 25%
            # ============================================================
            
            model_names = ['XGBoost', 'LightGBM', 'CatBoost', 'RandomForest']
            
            # Create model info list
            model_info = []
            for i in range(len(predictions)):
                model_info.append({
                    'name': model_names[i] if i < len(model_names) else f'Model {i}',
                    'prediction': predictions[i],
                    'confidence': individual_confidences[i]
                })
            
            # Filter: confidence > 25%
            valid_models = [m for m in model_info if m['confidence'] > 25]
            
            if len(valid_models) == 0:
                # No models pass threshold - use all models as fallback
                print("⚠️ Warning: No models with confidence > 25%, using all models")
                filtered_models = model_info
            else:
                # Sort by confidence and take top 3
                sorted_models = sorted(valid_models, key=lambda x: x['confidence'], reverse=True)
                filtered_models = sorted_models[:3]
            
            # Calculate ensemble with SIMPLE AVERAGE (không dùng trọng số)
            filtered_preds = [m['prediction'] for m in filtered_models]
            filtered_confs = [m['confidence'] for m in filtered_models]
            
            # SIMPLE AVERAGE
            days = np.mean(filtered_preds)
            days = max(0, float(days))
            
            # Recalculate statistics with filtered models
            filtered_std = np.std(filtered_preds)
            filtered_variance = np.var(filtered_preds)
            
            # Ensemble confidence (based on agreement between filtered models)
            if filtered_std < 0.3:
                ensemble_confidence = 95
            elif filtered_std < 0.5:
                ensemble_confidence = 85
            elif filtered_std < 0.8:
                ensemble_confidence = 75
            elif filtered_std < 1.2:
                ensemble_confidence = 65
            else:
                ensemble_confidence = 50

            
            # Status (Day 1-6)
            if days >= 4.5:
                status = "🟢 Day 1 - Rất tươi"
                day = 1
            elif days >= 3.5:
                status = "🟢 Day 2 - Tươi"
                day = 2
            elif days >= 2.5:
                status = "🟡 Day 3 - Chín vừa"
                day = 3
            elif days >= 1.5:
                status = "🟠 Day 4 - Chín"
                day = 4
            elif days >= 0.5:
                status = "🔴 Day 5 - Rất chín"
                day = 5
            else:
                status = "⚫ Day 6 - Hư"
                day = 6
            
            return {
                'image': Path(image_path).name,
                'days': round(days, 2),
                'status': status,
                'day': day,
                'preds': [round(p, 2) for p in predictions],
                'confidences': [round(c, 1) for c in individual_confidences],
                'ensemble_confidence': round(ensemble_confidence, 1),
                'std': round(filtered_std, 2),
                'variance': round(filtered_variance, 2),
                # Filter info
                'models_used': len(filtered_models),
                'models_total': len(predictions),
                'filtered_models': filtered_models,
                'all_models': model_info
            }
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return None


def main():
    predictor = BananaPredictor()
    
    while True:
        print("\n" + "─" * 60)
        print("[1] Test image  [2] Test folder  [0] Exit")
        choice = input("Choice: ").strip()
        
        if choice == '0':
            break
        
        elif choice == '1':
            path = input("\nImage path: ").strip().strip('"')
            if not Path(path).exists():
                print("❌ Not found")
                continue
            
            print("\n🎯 Predicting...")
            result = predictor.predict(path)
            
            if result:
                print(f"\n{'='*80}")
                print(f"📸 IMAGE: {result['image']}")
                print(f"{'='*80}")
                
                # All models
                print(f"\n🤖 ALL MODELS:")
                print(f"{'─'*80}")
                
                for model in result['all_models']:
                    conf_icon = "🟢" if model['confidence'] >= 60 else "🟡" if model['confidence'] >= 25 else "🔴"
                    bar_length = int(model['prediction'] * 8) if model['prediction'] >= 0 else 0
                    bar = '█' * bar_length
                    
                    # Mark if used
                    used = "✓" if any(m['name'] == model['name'] for m in result['filtered_models']) else "✗"
                    
                    print(f"   [{used}] {model['name']:15s}: {model['prediction']:5.2f} days  "
                          f"{bar:<40}  {conf_icon} {model['confidence']:5.1f}%")
                
                print(f"{'─'*80}")
                
                # Filter info
                print(f"\n🔍 CONFIDENCE FILTER:")
                print(f"   Threshold: > 25%")
                print(f"   Models used: {result['models_used']}/{result['models_total']}")
                print(f"   Average method: SIMPLE AVERAGE (equal weight)")
                
                # Top models
                print(f"\n🏆 MODELS USED:")
                print(f"{'─'*80}")
                
                for i, model in enumerate(result['filtered_models'], 1):
                    conf_icon = "🟢" if model['confidence'] >= 60 else "🟡"
                    print(f"   #{i}. {model['name']:15s}: {model['prediction']:5.2f} days  {conf_icon} {model['confidence']:5.1f}%")
                
                print(f"{'─'*80}")
                
                # Agreement
                print(f"\n📈 MODEL AGREEMENT:")
                print(f"   Standard Deviation: {result['std']:.2f}")
                print(f"   Variance: {result['variance']:.2f}")
                
                # Final result
                print(f"\n📊 FINAL RESULT:")
                avg_parts = ' + '.join([f"{m['prediction']:.2f}" for m in result['filtered_models']])
                print(f"   Average: ({avg_parts}) / {len(result['filtered_models'])}")
                print(f"   Prediction: {result['days']:.2f} days")
                
                ens_conf = result['ensemble_confidence']
                ens_icon = "🟢" if ens_conf >= 85 else "🟡" if ens_conf >= 70 else "🔴"
                
                print(f"   Confidence: {ens_icon} {ens_conf:.1f}%")
                print(f"\n   Status: {result['status']}")
                print(f"   Day: {result['day']}/6")
                print(f"{'='*80}")
        
        elif choice == '2':
            folder = input("\nFolder path: ").strip().strip('"')
            folder = Path(folder)
            if not folder.exists():
                print("❌ Not found")
                continue
            
            images = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG']:
                images.extend(list(folder.glob(ext)))
            
            if not images:
                print("❌ No images")
                continue
            
            print(f"\n📁 Processing {len(images)} images...")
            results = []
            for img in images:
                r = predictor.predict(str(img))
                if r:
                    results.append(r)
            
            if results:
                print(f"\n{'='*100}")
                print(f"📊 SUMMARY ({len(results)} images)")
                print(f"{'='*100}")
                print(f"\n{'Image':<25} {'Final':>6}  {'Conf':>5}  {'XGB':>5} {'LGB':>5} {'Cat':>5} {'RF':>5}  {'Status':<25}")
                print("─" * 100)
                
                for r in sorted(results, key=lambda x: x['days'], reverse=True):
                    preds = r['preds']
                    while len(preds) < 4:
                        preds.append(0)
                    
                    # Confidence icon
                    conf = r['ensemble_confidence']
                    if conf >= 85:
                        conf_icon = "🟢"
                    elif conf >= 70:
                        conf_icon = "🟡"
                    else:
                        conf_icon = "🔴"
                    
                    print(f"{r['image']:<25} {r['days']:>6.2f}  {conf_icon}{conf:>4.0f}%  "
                          f"{preds[0]:>5.2f} {preds[1]:>5.2f} {preds[2]:>5.2f} {preds[3]:>5.2f}  "
                          f"{r['status']:<25}")
                
                print("=" * 100)
                
                # Statistics
                avg_conf = np.mean([r['ensemble_confidence'] for r in results])
                print(f"\n📈 STATISTICS:")
                print(f"   Average confidence: {avg_conf:.1f}%")
                print(f"   High confidence (≥85%): {sum(1 for r in results if r['ensemble_confidence'] >= 85)}/{len(results)}")
                print(f"   Medium confidence (70-85%): {sum(1 for r in results if 70 <= r['ensemble_confidence'] < 85)}/{len(results)}")
                print(f"   Low confidence (<70%): {sum(1 for r in results if r['ensemble_confidence'] < 70)}/{len(results)}")
                print("=" * 100)


if __name__ == "__main__":
    main()