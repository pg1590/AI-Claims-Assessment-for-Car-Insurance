import streamlit as st

def assess_claim(image, damage_model, severity_model, preprocessor, 
                 fraud_detector, cost_estimator, explainer, user_description=""):
    """Assess insurance claim with optional text description"""
    
    # Preprocess image
    image_tensor, original_image, quality = preprocessor.preprocess(image)
    
    # Damage detection using YOLO
    damage_results = damage_model.predict(original_image)
    if "error" in damage_results:
        st.error(f"Damage detection error: {damage_results['error']}")
        return None
    
    damage_type = damage_results['prediction']
    damage_confidence = damage_results['confidence']
    
    # Severity assessment using YOLO
    severity_probs_dict, severity = severity_model.predict(original_image)
    if isinstance(severity_probs_dict, dict) and "error" in severity_probs_dict:
        st.error(f"Severity assessment error: {severity_probs_dict['error']}")
        severity_confidence = 0.5
    else:
        severity_confidence = max(severity_probs_dict.values())
    
    # Fraud detection
    fraud_results = fraud_detector.detect_fraud(original_image)
    
    # Import text consistency checker
    from processing.text_image_consistency import TextImageConsistencyChecker
    text_checker = TextImageConsistencyChecker()
    
    # Process user description if provided
    text_result = None
    if user_description and user_description.strip():
        text_result = text_checker.process(user_description)
    
    # Calculate image-based cost
    image_cost = cost_estimator.estimate_cost(damage_type, severity, damage_confidence)
    
    # Reconcile costs if text description provided
    final_cost = image_cost['estimated_cost']
    cost_breakdown = {
        'image_based_cost': image_cost['estimated_cost'],
        'text_based_cost': None,
        'reconciliation_method': 'image_only'
    }
    
    if text_result and text_result.get('text_provided', False):
        text_cost = text_result['estimated_cost']
        cost_breakdown['text_based_cost'] = text_cost
        
        cost_difference = abs(image_cost['estimated_cost'] - text_cost)
        
        if cost_difference > 1000:
            # Use average if difference > $1000
            final_cost = (image_cost['estimated_cost'] + text_cost) / 2
            cost_breakdown['reconciliation_method'] = 'average'
        else:
            # Use maximum if difference <= $1000
            final_cost = max(image_cost['estimated_cost'], text_cost)
            cost_breakdown['reconciliation_method'] = 'maximum'
        
        cost_breakdown['cost_difference'] = cost_difference
    
    # Update cost result with reconciled value
    final_cost_result = {
        'estimated_cost': final_cost,
        'range': (final_cost * 0.8, final_cost * 1.2),
        'cost_band': get_cost_band(final_cost),
        'breakdown': cost_breakdown,
        'text_analysis': text_result
    }
    
    # Generate explainability heatmap
    explanation_viz = None
    heatmap_array = None
    if damage_type != 'no_damage':
        try:
            heatmap = explainer.generate_heatmap(image_tensor)
            explanation_viz = explainer.visualize(original_image, heatmap)
            heatmap_array = heatmap
        except Exception as e:
            print(f"Explainability generation failed: {e}")
    
    # Calculate final confidence
    model_confidence = (damage_confidence + severity_confidence) / 2
    fraud_penalty = fraud_results['fraud_score']
    quality_factor = quality['overall_quality']
    final_confidence = model_confidence * (1 - fraud_penalty * 0.5) * (0.7 + quality_factor * 0.3)
    final_confidence = min(max(final_confidence, 0), 1)
    
    # Generate recommendation
    if fraud_results['fraud_level'] == 'HIGH':
        recommendation = "REJECT"
        rec_detail = "High fraud risk detected. Manual review required."
    elif fraud_results['fraud_level'] == 'MEDIUM':
        recommendation = "MANUAL REVIEW"
        rec_detail = "Moderate fraud indicators. Verify with additional documentation."
    elif final_confidence < 0.5:
        recommendation = "MANUAL REVIEW"
        rec_detail = "Low confidence in assessment. Additional images recommended."
    elif final_confidence < 0.7:
        recommendation = "APPROVE WITH VERIFICATION"
        rec_detail = "Moderate confidence. Quick manual check advised."
    else:
        recommendation = "APPROVE"
        rec_detail = "High confidence assessment. Process claim."
    
    results = {
        'damage': {
            'type': damage_type,
            'confidence': damage_confidence
        },
        'severity': {
            'level': severity,
            'confidence': severity_confidence,
            'all_probabilities': severity_probs_dict if isinstance(severity_probs_dict, dict) and "error" not in severity_probs_dict else {}
        },
        'fraud': fraud_results,
        'cost': final_cost_result,
        'quality': quality,
        'final_confidence': final_confidence,
        'recommendation': recommendation,
        'recommendation_detail': rec_detail,
        'explanation': explanation_viz,
        'heatmap': heatmap_array
    }
    
    return results

def get_cost_band(cost):
    """Helper function to determine cost band"""
    if cost < 500:
        return "Minor Repair"
    elif cost < 2000:
        return "Moderate Repair"
    elif cost < 5000:
        return "Major Repair"
    else:
        return "Extensive Repair"
