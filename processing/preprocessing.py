

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt




import warnings
warnings.filterwarnings('ignore')


class ImagePreprocessor:
    def __init__(self, image_size=224):
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def preprocess(self, image):
        quality_score = self._check_image_quality(image)
        tensor = self.transform(image)
        return tensor, image, quality_score
    
    def _check_image_quality(self, image):
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        brightness = np.mean(gray)
        blur_quality = min(blur_score / 500, 1.0)
        brightness_quality = 1.0 - abs(brightness - 127.5) / 127.5
        overall_quality = (blur_quality * 0.6 + brightness_quality * 0.4)
        
        return {
            'blur_score': blur_score,
            'brightness': brightness,
            'overall_quality': overall_quality,
            'is_acceptable': blur_score > 100 and 30 < brightness < 225
        }

