
# processing/text_image_consistency.py
import re

class TextImageConsistencyChecker:
    def __init__(self):
        # Base cost mapping for different damages
        self.base_costs = {
            'scratch': 200,
            'dent': 400,
            'broken_glass': 300,
            'broken_lamp': 250,
            'bumper_damage': 600,
            'door_damage': 800,
            'hood_damage': 700,
            'no_damage': 0
        }

        # Convert keys into clean words for matching (e.g. "broken_lamp" -> "broken lamp")
        self.damage_keywords = {
            key: key.replace('_', ' ')
            for key in self.base_costs.keys()
        }

    def extract_damage_parts(self, text: str):
        """
        Extract damage parts mentioned in text based on known damage keywords.
        Returns a list of matched damage keys.
        """
        text_lower = text.lower()
        matched_parts = []

        for damage_key, damage_phrase in self.damage_keywords.items():
            # Use regex word boundaries to match whole words/phrases
            if re.search(r'\b' + re.escape(damage_phrase) + r'\b', text_lower):
                matched_parts.append(damage_key)

        return matched_parts

    def estimate_cost(self, matched_parts: list):
        """
        Estimate total repair cost based on matched damage parts.
        """
        total_cost = sum(self.base_costs[part] for part in matched_parts)
        return total_cost

    def process(self, text: str):
        """
        Main pipeline: extract damage parts and calculate total cost.
        Returns dictionary with parts and cost.
        """
        matched_parts = self.extract_damage_parts(text)
        total_cost = self.estimate_cost(matched_parts)

        return {
            "matched_parts": matched_parts,
            "estimated_cost": total_cost
        }
