import sys
import os
import cv2
import numpy as np
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

try:
    from src.anti_spoof_predict import AntiSpoofPredict
    from src.generate_patches import CropImage
    from src.utility import parse_model_name
except ImportError as e:
    logger.error(f"Import error: {str(e)}")
    raise

class LivenessDetector:
    """Using Silent-Face-Anti-Spoofing model"""
    
    def __init__(self, silent_face_path='/app/Silent-Face-Anti-Spoofing-master'):
        self.original_cwd = os.getcwd()
        self.silent_face_path = silent_face_path

        # Full paths to model files
        self.model_dir = os.path.join(self.silent_face_path, "resources", "anti_spoof_models")
        self.model_files = [
            os.path.join(self.model_dir, "2.7_80x80_MiniFASNetV2.pth"),
            os.path.join(self.model_dir, "4_0_0_80x80_MiniFASNetV1SE.pth")
        ]

        # Check for specific model files
        deploy_path = os.path.join(self.model_dir, "deploy.prototxt")
        caffemodel_path = os.path.join(self.model_dir, "model.caffemodel")

        if not os.path.exists(deploy_path) or not os.path.exists(caffemodel_path):
            logger.error("Model files missing: Ensure deploy.prototxt and model.caffemodel exist.")
            raise FileNotFoundError("Required model files are missing.")
        
        # Check if model files exist
        for model_file in self.model_files:
            if not os.path.exists(model_file):
                logger.error(f"Model file not found: {model_file}")
            else:
                logger.info(f"Model file found: {model_file}")

        self.model = AntiSpoofPredict(device_id=-1)  # Use CPU (-1) for compatibility
        self.image_cropper = CropImage()
        
    def analyze(self, image: np.ndarray) -> dict:
        """Analyze using pre-trained anti-spoofing model"""
        try:
            os.chdir(self.silent_face_path)
            image_bbox = self.model.get_bbox(image)
            prediction = np.zeros((1, 3))
            
            for model_file in self.model_files:
                if not os.path.exists(model_file):
                    logger.warning(f"Skipping missing model: {model_file}")
                    continue

                model_name = os.path.basename(model_file)
                h_input, w_input, model_type, scale = parse_model_name(model_name)
                
                param = {
                    "org_img": image,
                    "bbox": image_bbox,
                    "scale": scale,
                    "out_w": w_input,
                    "out_h": h_input,
                    "crop": True,
                }
                
                if scale is None:
                    param["crop"] = False
                
                img = self.image_cropper.crop(**param)
                prediction += self.model.predict(img, model_file)
            
            os.chdir(self.original_cwd)
            
            label = np.argmax(prediction)
            value = prediction[0][label] / 2
            is_live = label == 1
            confidence = float(value)
            
            return {
                "live": is_live,
                "score": confidence,
                "explanation": f"Model prediction: {'Real' if is_live else 'Fake'} ({confidence:.3f})",
                "message": "Live person detected" if is_live else "Spoof detected"
            }
            
        except Exception as e:
            os.chdir(self.original_cwd)
            logger.error(f"Liveness detection error: {str(e)}")
            return {
                "live": True,  # Fail open
                "score": 0.5,
                "explanation": f"Model error: {str(e)}",
                "message": "Liveness check skipped"
            }