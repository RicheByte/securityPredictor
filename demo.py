# demo_predictions.py
import pandas as pd
import joblib
from main import UltraOptimizedSecurityPredictor
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SecurityDemo:
    def __init__(self):
        self.predictor = UltraOptimizedSecurityPredictor()
        self.load_models()
    
    def load_models(self):
        """Load trained models"""
        try:
            self.incident_model = joblib.load('incident_model.joblib')
            self.severity_model = joblib.load('severity_model.joblib')
            
            # Load feature names
            train_df = pd.read_parquet("train_optimized.parquet")
            self.feature_names = self.predictor.select_features_correlation(
                self.predictor.efficient_feature_engineering(train_df), 
                'IsIncident', 
                top_k=20
            ).columns.tolist()
            
            logger.info("✅ Demo models loaded!")
        except Exception as e:
            logger.error(f"❌ Error: {e}")
    
    def create_demo_alerts(self):
        """Create realistic demo security alerts"""
        demo_alerts = [
            {
                'name': 'Normal User Activity',
                'description': 'Regular user login during business hours',
                'features': {'SuspicionLevel': 0.1, 'mitre_count': 0, 'is_business_hours': 1}
            },
            {
                'name': 'Suspicious Login',
                'description': 'Multiple failed login attempts after hours',
                'features': {'SuspicionLevel': 0.8, 'mitre_count': 2, 'is_business_hours': 0}
            },
            {
                'name': 'Potential Malware',
                'description': 'Unusual process execution with MITRE techniques',
                'features': {'SuspicionLevel': 0.9, 'mitre_count': 5, 'is_business_hours': 1}
            }
        ]
        return demo_alerts
    
    def run_demo(self):
        """Run the demo with sample alerts"""
        logger.info("\n" + "🔒 SECURITY PREDICTION DEMO")
        logger.info("="*40)
        
        demo_alerts = self.create_demo_alerts()
        
        for alert in demo_alerts:
            logger.info(f"\n📋 Alert: {alert['name']}")
            logger.info(f"   Description: {alert['description']}")
            
            # Create full feature set
            full_features = {}
            for feature in self.feature_names:
                if feature in alert['features']:
                    full_features[feature] = alert['features'][feature]
                else:
                    # Set default values for missing features
                    if 'freq' in feature or 'count' in feature:
                        full_features[feature] = 1
                    else:
                        full_features[feature] = 0.5
            
            # Get prediction
            try:
                prediction = self.predictor.predict_single_optimized(full_features)
                
                if "error" not in prediction:
                    logger.info(f"   🎯 Prediction: {'INCIDENT' if prediction['incident_prediction'] else 'NO INCIDENT'}")
                    logger.info(f"   📊 Confidence: {prediction['incident_confidence']:.1%}")
                    logger.info(f"   🚨 Severity: Level {prediction['severity_level']}")
                    logger.info(f"   ⚠️  Risk: {prediction['risk_category']}")
                    
                    # Interpretation
                    if prediction['incident_confidence'] > 0.7:
                        logger.info("   💡 ACTION: Investigate immediately!")
                    elif prediction['incident_confidence'] > 0.3:
                        logger.info("   💡 ACTION: Monitor closely")
                    else:
                        logger.info("   💡 ACTION: Normal activity")
                else:
                    logger.error(f"   ❌ Prediction error: {prediction['error']}")
                    
            except Exception as e:
                logger.error(f"   ❌ Processing error: {e}")

if __name__ == "__main__":
    demo = SecurityDemo()
    demo.run_demo()