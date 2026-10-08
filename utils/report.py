from datetime import datetime

def generate_report(results, filename):
    """Generate a comprehensive text report"""
    
    report = f"""
================================================================================
                    VEHICLE DAMAGE ASSESSMENT REPORT
================================================================================

REPORT METADATA:
  Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
  Image File: {filename}
  Assessment ID: {datetime.now().strftime('%Y%m%d%H%M%S')}

ASSESSMENT SUMMARY:
  Recommendation: {results['recommendation']}
  Details: {results['recommendation_detail']}
  Overall Confidence: {results['final_confidence']:.1%}

DAMAGE ANALYSIS:
  Damage Type: {results['damage']['type'].replace('_', ' ').title()}
  Detection Confidence: {results['damage']['confidence']:.1%}
  
SEVERITY ASSESSMENT:
  Severity Level: {results['severity']['level'].upper()}
  Assessment Confidence: {results['severity']['confidence']:.1%}

FRAUD DETECTION:
  Risk Level: {results['fraud']['fraud_level']}
  Fraud Score: {results['fraud']['fraud_score']:.1%}
  Indicators: {', '.join(results['fraud']['indicators']) if results['fraud']['indicators'] else 'None detected'}

COST ESTIMATION:
  Estimated Cost: ${results['cost']['estimated_cost']:,.2f} USD
  Cost Band: {results['cost']['cost_band']}
  Expected Range: ${results['cost']['range'][0]:,.2f} - ${results['cost']['range'][1]:,.2f} USD
"""

    breakdown = results['cost'].get('breakdown', {})
    if breakdown.get('text_based_cost') is not None:
        report += f"""
COST RECONCILIATION:
  Image-Based Estimate: ${breakdown['image_based_cost']:,.2f} USD
  Text-Based Estimate: ${breakdown['text_based_cost']:,.2f} USD
  Cost Difference: ${breakdown.get('cost_difference', 0):,.2f}
  Reconciliation Method: {breakdown['reconciliation_method'].upper()}
"""
        
        if results['cost'].get('text_analysis') and results['cost']['text_analysis'].get('matched_parts'):
            matched_parts = results['cost']['text_analysis']['matched_parts']
            parts_list = ', '.join([part.replace('_', ' ').title() for part in matched_parts])
            report += f"""  Damage Detected from Text: {parts_list}
"""

    report += f"""
IMAGE QUALITY METRICS:
  Overall Quality: {results['quality']['overall_quality']:.1%}
  Blur Score: {results['quality']['blur_score']:.2f}
  Brightness: {results['quality']['brightness']:.2f}

DETAILED FRAUD ANALYSIS:
"""
    
    for key, value in results['fraud']['details'].items():
        label = key.replace('_', ' ').title()
        report += f"  {label}: {value:.1%}\n"
    
    if results['severity']['all_probabilities']:
        report += "\nSEVERITY PROBABILITIES:\n"
        for level, prob in results['severity']['all_probabilities'].items():
            report += f"  {level.upper()}: {prob:.1%}\n"
    
    report += """
================================================================================
                            END OF REPORT
================================================================================

DISCLAIMER:
This automated assessment is for reference purposes only. Final repair costs
may vary based on local labor rates, parts availability, and additional damage
discovered during physical inspection. Manual verification is recommended for
high-value claims or cases flagged for review.

================================================================================
"""
    
    return report