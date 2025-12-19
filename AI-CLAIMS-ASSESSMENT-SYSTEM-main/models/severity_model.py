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
from config.settings import Config

config = Config()

class SeverityAssessmentModel:
    def __init__(self, model_path):
        self.class_names = ["minor", "moderate", "severe"]
        try:
            self.model = YOLO(model_path)
            print("✅ Severity model loaded successfully!")
        except Exception as e:
            raise IOError(f"Failed to load severity model: {e}")
    
    def predict(self, image):
        try:
            results = self.model(image, verbose=False)
            probs = results[0].probs
            
            if probs is None:
                raise RuntimeError("Model did not return probabilities")
            
            prob_values = probs.data.tolist()
            predicted_idx = probs.top1
            predicted_class = self.class_names[predicted_idx]
            
            probabilities = dict(zip(self.class_names, prob_values))
            
            return probabilities, predicted_class
        except Exception as e:
            return {"error": str(e)}, "moderate"