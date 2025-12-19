import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
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

# ============================================================================
# SECTION 6: UPGRADED FRAUD DETECTION MODULE
# ============================================================================

from PIL import ExifTags # Make sure this import is at the top of your script

class FraudDetector:
    """Detects potential fraud through multiple, enhanced checks"""

    def __init__(self):
        self.image_hashes = {}
        self.submission_history = []

    def detect_fraud(self, image, claim_text=None):
        fraud_indicators = []
        fraud_score = 0.0

        # --- UPDATED: Call all new and old checks ---
        checks = {
            'duplicate': (self._check_duplicate(image), 0.4), # High weight
            'ela_tampering': (self._detect_ela_tampering(image), 0.5), # Very high weight
            'exif_metadata': (self._analyze_metadata(image), 0.6), # Very high weight
            'ai_generated': (self._detect_ai_generation(image), 0.4), # High weight
            'frequency': (self._check_submission_frequency(), 0.2) # Lower weight
        }

        details = {}
        for check_name, (result, weight) in checks.items():
            score, reason = result
            details[f'{check_name}_score'] = score
            if score > 0:
                fraud_indicators.append(reason)
                fraud_score += score * weight
        
        fraud_score = min(fraud_score, 1.0) # Cap the score at 1.0

        return {
            'fraud_score': fraud_score,
            'fraud_level': self._get_fraud_level(fraud_score),
            'indicators': fraud_indicators if fraud_indicators else ["No obvious fraud indicators found."],
            'details': details
        }

    # --- NEW: Error Level Analysis ---
    def _detect_ela_tampering(self, image, quality=90, scale=15):
        temp_filename = "temp_ela.jpg"
        try:
            image.save(temp_filename, 'JPEG', quality=quality)
            resaved_image = Image.open(temp_filename)
            ela_image = Image.fromarray(np.abs(np.array(image) - np.array(resaved_image)) * scale)
            extrema = ela_image.getextrema()
            max_diff = max([ex[1] for ex in extrema]) if extrema else 0
        finally:
            if os.path.exists(temp_filename):
                os.remove(temp_filename)

        if max_diff > 100:
            return 0.7, f"High ELA variance detected (max diff: {max_diff}), indicates potential tampering."
        return 0.0, ""

    # --- NEW: AI Generation Check ---
    def _detect_ai_generation(self, image):
        img_array = np.array(image.convert('RGB'))
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        noise = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
        freq_energy = np.mean(magnitude_spectrum)
        
        score = 0.0
        reasons = []
        if noise < 100:
            score += 0.4
            reasons.append(f"Unusually low noise variance ({noise:.2f}), suggesting a synthetic origin.")
        if freq_energy < 8500:
            score += 0.3
            reasons.append(f"Smooth frequency spectrum ({freq_energy:.2f}), common in generated images.")
        return score, ", ".join(reasons) if reasons else ""

    # --- UPDATED: Deep EXIF Analysis ---
    def _analyze_metadata(self, image):
        score = 0.0
        reasons = []
        try:
            exif_data = image._getexif()
            if exif_data:
                exif = {ExifTags.TAGS[k]: v for k, v in exif_data.items() if k in ExifTags.TAGS}
                software = exif.get('Software', '').lower()
                if 'photoshop' in software or 'gimp' in software:
                    score = 1.0
                    reasons.append(f"Editing software found in metadata: {exif.get('Software')}")
                if 'Make' not in exif or 'Model' not in exif:
                    score = max(score, 0.2)
                    reasons.append("Missing camera make/model metadata.")
            else:
                score = 0.3
                reasons.append("No EXIF metadata found.")
        except Exception:
            score = 0.2
            reasons.append("Could not parse EXIF metadata (potentially stripped).")
        return score, ", ".join(reasons) if reasons else ""
    
    # --- UNCHANGED METHODS ---
    def _check_duplicate(self, image):
        img_hash = str(imagehash.phash(image))
        for stored_hash, info in self.image_hashes.items():
            if imagehash.hex_to_hash(img_hash) - imagehash.hex_to_hash(stored_hash) < 5:
                return 0.9, f"High similarity to a previously submitted image."
        if img_hash in self.image_hashes:
            self.image_hashes[img_hash]['count'] += 1
        else:
            self.image_hashes[img_hash] = {'count': 1}
        return 0.0, ""

    def _check_submission_frequency(self):
        self.submission_history.append(datetime.now())
        cutoff = datetime.now().timestamp() - 86400
        self.submission_history = [ts for ts in self.submission_history if ts.timestamp() > cutoff]
        if len(self.submission_history) > 10:
            score = min((len(self.submission_history) - 10) / 20, 1.0)
            return score, f"High submission frequency ({len(self.submission_history)} claims in 24h)."
        return 0.0, ""

    def _get_fraud_level(self, score):
        if score >= config.FRAUD_THRESHOLD_HIGH: return "HIGH"
        elif score >= config.FRAUD_THRESHOLD_MEDIUM: return "MEDIUM"
        else: return "LOW"