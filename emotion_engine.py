
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from pathlib import Path

# Add emonet to path so we can import models
EMONET_PATH = Path(__file__).parent / "emonet"
sys.path.insert(0, str(EMONET_PATH))

from emonet.models.emonet import EmoNet

class EmotionEngine:
    def __init__(self, model_variant=8):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Paths to pretrained weights
        if model_variant == 5:
            self.model_path = EMONET_PATH / "pretrained" / "emonet_5.pth"
            self.n_expression = 5
        else:
            self.model_path = EMONET_PATH / "pretrained" / "emonet_8.pth"
            self.n_expression = 8

        # Load Face Detector
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        self.model = self._load_model()
        
        # Map indices to labels (standard for EmoNet 8)
        self.emotion_labels = [
            'Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Angry', 'Contempt'
        ]

        # Manual mapping for UI selection fallback
        self.emotion_to_va = {
            "Happy": (0.8, 0.6),
            "Surprise": (0.7, 0.8),
            "Angry": (-0.7, 0.9),
            "Fear": (-0.6, 0.8),
            "Disgust": (-0.8, 0.5),
            "Sad": (-0.9, -0.7),
            "Neutral": (0.2, -0.6),
            "Contempt": (-0.3, 0.2)
        }

    def _load_model(self):
        state_dict = torch.load(self.model_path, map_location='cpu')
        model = EmoNet(n_expression=self.n_expression).to(self.device)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        print(f"Loaded EmoNet-{self.n_expression} from {self.model_path}")
        return model

    def preprocess_image(self, image_bgr):
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        
        if len(faces) > 0:
            (x, y, w, h) = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
            # EmoNet expects 256x256 RGB input
            roi = image_bgr[y:y+h, x:x+w]
            roi = cv2.resize(roi, (256, 256))
        else:
            # Fallback to whole image
            roi = cv2.resize(image_bgr, (256, 256))
            
        # Convert BGR to RGB
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        # Normalize: [0, 1] and then standard ImageNet-like normalization if needed
        # EmoNet usually takes 0-1 range
        img_tensor = torch.from_numpy(roi_rgb).permute(2, 0, 1).float() / 255.0
        return img_tensor.unsqueeze(0).to(self.device)

    def predict(self, image_bgr):
        with torch.no_grad():
            processed = self.preprocess_image(image_bgr)
            out = self.model(processed)
            
            # Extract expression
            expr_logits = out['expression']
            max_idx = torch.argmax(expr_logits, dim=1).item()
            emotion = self.emotion_labels[max_idx] if max_idx < len(self.emotion_labels) else "Neutral"
            
            # Extract Valence and Arousal directly from EmoNet
            valence = out['valence'].item()
            arousal = out['arousal'].item()
            
            # EmoNet VA is typically [-1, 1], EMOPIA engine uses similar
            print(f"EmoNet Prediction: {emotion} (V: {valence:.2f}, A: {arousal:.2f})")
            
            return {
                "emotion": emotion,
                "valence": valence,
                "arousal": arousal,
                "confidence": torch.softmax(expr_logits, dim=1).max().item()
            }

if __name__ == "__main__":
    engine = EmotionEngine()
    # Dummy test
    dummy = np.zeros((200, 200, 3), dtype=np.uint8)
    res = engine.predict(dummy)
    print("Test result:", res)
