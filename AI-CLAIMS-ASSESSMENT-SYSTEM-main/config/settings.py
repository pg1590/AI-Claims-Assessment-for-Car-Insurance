import torch

class Config:
    IMAGE_SIZE = 224
    DAMAGE_TYPES = [
        'scratch', 'dent', 'broken_glass', 'broken_lamp', 
        'bumper_damage', 'door_damage', 'hood_damage', 'no_damage'
    ]
    SEVERITY_LEVELS = ['minor', 'moderate', 'severe']
    COST_BANDS = {
        'minor': (100, 500),
        'moderate': (500, 2000),
        'severe': (2000, 10000)
    }
    FRAUD_THRESHOLD_HIGH = 0.7
    FRAUD_THRESHOLD_MEDIUM = 0.4
    
    # Model paths - UPDATE THESE WITH YOUR ACTUAL MODEL PATHS
    DAMAGE_MODEL_PATH = "best.pt"  # Your YOLO damage detection model
    SEVERITY_MODEL_PATH = "car-damage.pt"  # Your YOLO severity model
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')