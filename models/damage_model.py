import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import imagehash
from datetime import datetime
import io
from ultralytics import YOLO
import warnings
warnings.filterwarnings('ignore')

class DamageDetectionModel:
    def __init__(self, model_path):
        try:
            self.model = YOLO(model_path)
            print("✅ Damage detection model loaded successfully!")
        except Exception as e:
            raise IOError(f"Failed to load damage model: {e}")
    
    def predict(self, image):
        try:
            results = self.model.predict(image, verbose=False)
            result = results[0]
            predicted_class_index = result.probs.top1
            predicted_class_name = self.model.names[predicted_class_index]
            confidence_score = result.probs.top1conf.item()
            
            return {
                "prediction": predicted_class_name,
                "confidence": confidence_score
            }
        except Exception as e:
            return {"error": f"Prediction failed: {e}"}