from config.settings import Config
config = Config()

class CostEstimator:
    def __init__(self):
        self.base_costs = {
            'scratch': 200, 'dent': 400, 'broken_glass': 300,
            'broken_lamp': 250, 'bumper_damage': 600,
            'door_damage': 800, 'hood_damage': 700, 'no_damage': 0
        }
        self.severity_multipliers = {
            'minor': 1.0, 'moderate': 2.0, 'severe': 4.0
        }
    
    def estimate_cost(self, damage_type, severity, confidence):
        base_cost = self.base_costs.get(damage_type, 500)
        multiplier = self.severity_multipliers.get(severity, 1.5)
        estimated_cost = base_cost * multiplier
        confidence_factor = 0.8 + (confidence * 0.4)
        estimated_cost *= confidence_factor
        cost_band = self._get_cost_band(severity)
        
        return {
            'estimated_cost': round(estimated_cost, 2),
            'cost_band': cost_band,
            'range': config.COST_BANDS[severity],
            'currency': 'USD'
        }
    
    def _get_cost_band(self, severity):
        bands = {
            'minor': 'Low ($100-$500)',
            'moderate': 'Medium ($500-$2000)',
            'severe': 'High ($2000-$10000)'
        }
        return bands.get(severity, 'Medium')
