# test_model_workings.py
import pandas as pd
import numpy as np
import joblib
from main import UltraOptimizedSecurityPredictor
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelTester:
    def __init__(self):
        self.predictor = UltraOptimizedSecurityPredictor()
        self.load_models()
        
    def load_models(self):
        """Load trained models"""
        try:
            self.incident_model = joblib.load('incident_model.joblib')
            self.severity_model = joblib.load('severity_model.joblib')
            self.anomaly_model = joblib.load('anomaly_model.joblib')
            logger.info("✅ Models loaded successfully!")
            
            # Load feature names from training data
            train_df = pd.read_parquet("train_optimized.parquet")
            self.feature_names = self.predictor.select_features_correlation(
                self.predictor.efficient_feature_engineering(train_df), 
                'IsIncident', 
                top_k=20
            ).columns.tolist()
            logger.info(f"📊 Loaded {len(self.feature_names)} feature names")
            
        except Exception as e:
            logger.error(f"❌ Error loading models: {e}")
            raise
    
    def test_model_architecture(self):
        """Test and understand model architecture"""
        logger.info("\n" + "="*50)
        logger.info("🧠 MODEL ARCHITECTURE ANALYSIS")
        logger.info("="*50)
        
        # Incident Model Analysis
        logger.info("\n🎯 INCIDENT PREDICTION MODEL:")
        logger.info(f"Model Type: {type(self.incident_model).__name__}")
        logger.info(f"Number of Trees: {self.incident_model.n_estimators}")
        logger.info(f"Number of Features: {len(self.incident_model.feature_importances_)}")
        logger.info(f"Classes: {self.incident_model.classes_}")
        
        # Severity Model Analysis
        logger.info("\n📊 SEVERITY PREDICTION MODEL:")
        logger.info(f"Model Type: {type(self.severity_model).__name__}")
        logger.info(f"Number of Trees: {self.severity_model.n_estimators}")
        logger.info(f"Classes: {self.severity_model.classes_}")
        
        # Anomaly Model Analysis
        logger.info("\n🔍 ANOMALY DETECTION MODEL:")
        logger.info(f"Model Type: {type(self.anomaly_model).__name__}")
        logger.info(f"Number of Trees: {self.anomaly_model.n_estimators}")
        logger.info(f"Contamination: {self.anomaly_model.contamination}")
    
    def test_feature_importance(self):
        """Analyze feature importance"""
        logger.info("\n" + "="*50)
        logger.info("📈 FEATURE IMPORTANCE ANALYSIS")
        logger.info("="*50)
        
        # Get feature importances
        incident_importance = self.incident_model.feature_importances_
        severity_importance = self.severity_model.feature_importances_
        
        # Create feature importance DataFrame
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Incident_Importance': incident_importance,
            'Severity_Importance': severity_importance
        })
        
        # Sort by incident importance
        importance_df = importance_df.sort_values('Incident_Importance', ascending=False)
        
        logger.info("\n🔝 TOP 10 MOST IMPORTANT FEATURES:")
        for idx, row in importance_df.head(10).iterrows():
            logger.info(f"  {row['Feature']}: Incident={row['Incident_Importance']:.4f}, Severity={row['Severity_Importance']:.4f}")
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        
        plt.subplot(1, 2, 1)
        top_10_incident = importance_df.head(10)
        plt.barh(top_10_incident['Feature'], top_10_incident['Incident_Importance'])
        plt.title('Top 10 Features - Incident Prediction')
        plt.xlabel('Importance')
        plt.gca().invert_yaxis()
        
        plt.subplot(1, 2, 2)
        top_10_severity = importance_df.head(10)
        plt.barh(top_10_severity['Feature'], top_10_severity['Severity_Importance'])
        plt.title('Top 10 Features - Severity Prediction')
        plt.xlabel('Importance')
        plt.gca().invert_yaxis()
        
        plt.tight_layout()
        plt.savefig('feature_importance_analysis.png', dpi=150, bbox_inches='tight')
        logger.info("💾 Feature importance plot saved as 'feature_importance_analysis.png'")
        plt.show()
        
        return importance_df
    
    def generate_test_alerts(self, num_alerts=5):
        """Generate realistic test alerts based on training data patterns"""
        logger.info("\n" + "="*50)
        logger.info("🚨 GENERATING TEST ALERTS")
        logger.info("="*50)
        
        # Load training data to understand data patterns
        train_df = pd.read_parquet("train_optimized.parquet")
        processed_train = self.predictor.efficient_feature_engineering(train_df)
        
        test_alerts = []
        
        for i in range(num_alerts):
            alert = {}
            
            # Generate realistic values based on training data patterns
            for feature in self.feature_names:
                if feature in processed_train.columns:
                    if processed_train[feature].dtype in ['int8', 'int16', 'int32', 'int64']:
                        # Use most frequent values for categorical-like features
                        alert[feature] = processed_train[feature].mode().iloc[0] if len(processed_train[feature].mode()) > 0 else 0
                    elif processed_train[feature].dtype in ['float32', 'float64']:
                        # Use mean for numerical features
                        alert[feature] = float(processed_train[feature].mean())
                    else:
                        alert[feature] = 0
                else:
                    alert[feature] = 0  # Default value for missing features
            
            # Make some alerts suspicious by modifying key features
            if i % 2 == 0:  # Make every other alert suspicious
                if 'SuspicionLevel' in alert:
                    alert['SuspicionLevel'] = min(alert.get('SuspicionLevel', 0) + 0.8, 1.0)
                if 'mitre_count' in alert:
                    alert['mitre_count'] = alert.get('mitre_count', 0) + 3
            
            test_alerts.append(alert)
            logger.info(f"📝 Generated alert {i+1}: {list(alert.items())[:3]}...")  # Show first 3 features
        
        return test_alerts
    
    def test_single_predictions(self, test_alerts):
        """Test single alert predictions"""
        logger.info("\n" + "="*50)
        logger.info("🔮 SINGLE ALERT PREDICTIONS")
        logger.info("="*50)
        
        results = []
        
        for i, alert in enumerate(test_alerts):
            logger.info(f"\n🔍 Analyzing Alert {i+1}:")
            
            try:
                prediction = self.predictor.predict_single_optimized(alert)
                
                if "error" not in prediction:
                    logger.info(f"   ✅ Incident Prediction: {prediction['incident_prediction']}")
                    logger.info(f"   📊 Confidence: {prediction['incident_confidence']:.3f}")
                    logger.info(f"   🚨 Severity Level: {prediction['severity_level']}")
                    logger.info(f"   ⚠️  Risk Category: {prediction['risk_category']}")
                    
                    results.append({
                        'alert_id': i+1,
                        **prediction,
                        'suspicion_level': alert.get('SuspicionLevel', 'N/A'),
                        'mitre_count': alert.get('mitre_count', 'N/A')
                    })
                else:
                    logger.error(f"   ❌ Prediction error: {prediction['error']}")
                    
            except Exception as e:
                logger.error(f"   ❌ Error processing alert {i+1}: {e}")
        
        return results
    
    def test_batch_predictions(self):
        """Test batch predictions on test data"""
        logger.info("\n" + "="*50)
        logger.info("📊 BATCH PREDICTION TEST")
        logger.info("="*50)
        
        try:
            # Load test data
            test_df = pd.read_parquet("test_optimized.parquet")
            processed_test = self.predictor.efficient_feature_engineering(test_df)
            
            # Select features
            X_test = processed_test[self.feature_names].fillna(0).astype('float32')
            y_incident_test = processed_test['IsIncident']
            y_severity_test = processed_test['Severity']
            
            logger.info(f"📈 Test dataset: {X_test.shape}")
            
            # Make batch predictions
            incident_pred = self.incident_model.predict(X_test)
            incident_proba = self.incident_model.predict_proba(X_test)[:, 1]
            severity_pred = self.severity_model.predict(X_test)
            
            # Calculate metrics
            from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
            
            incident_accuracy = accuracy_score(y_incident_test, incident_pred)
            severity_accuracy = accuracy_score(y_severity_test, severity_pred)
            
            logger.info(f"🎯 Incident Prediction Accuracy: {incident_accuracy:.4f}")
            logger.info(f"📊 Severity Prediction Accuracy: {severity_accuracy:.4f}")
            
            # Detailed classification report
            logger.info("\n📋 INCIDENT CLASSIFICATION REPORT:")
            logger.info(classification_report(y_incident_test, incident_pred, target_names=['No Incident', 'Incident']))
            
            logger.info("\n📋 SEVERITY CLASSIFICATION REPORT:")
            logger.info(classification_report(y_severity_test, severity_pred))
            
            return {
                'incident_accuracy': incident_accuracy,
                'severity_accuracy': severity_accuracy,
                'incident_predictions': incident_pred,
                'severity_predictions': severity_pred
            }
            
        except Exception as e:
            logger.error(f"❌ Batch prediction error: {e}")
            return None
    
    def test_anomaly_detection(self, test_alerts):
        """Test anomaly detection on generated alerts"""
        logger.info("\n" + "="*50)
        logger.info("🕵️ ANOMALY DETECTION TEST")
        logger.info("="*50)
        
        try:
            # Convert alerts to DataFrame
            alert_df = pd.DataFrame(test_alerts)[self.feature_names].fillna(0).astype('float32')
            
            # Get anomaly scores (-1 for outliers, 1 for inliers)
            anomaly_pred = self.anomaly_model.predict(alert_df)
            anomaly_scores = self.anomaly_model.decision_function(alert_df)
            
            for i, (pred, score) in enumerate(zip(anomaly_pred, anomaly_scores)):
                status = "🚨 ANOMALY" if pred == -1 else "✅ NORMAL"
                logger.info(f"Alert {i+1}: {status} (Score: {score:.3f})")
            
            return {
                'anomaly_predictions': anomaly_pred,
                'anomaly_scores': anomaly_scores
            }
            
        except Exception as e:
            logger.error(f"❌ Anomaly detection error: {e}")
            return None
    
    def test_model_confidence(self, test_alerts):
        """Analyze model confidence levels"""
        logger.info("\n" + "="*50)
        logger.info("🎯 MODEL CONFIDENCE ANALYSIS")
        logger.info("="*50)
        
        confidence_data = []
        
        for i, alert in enumerate(test_alerts):
            try:
                # Convert to DataFrame for prediction
                input_df = pd.DataFrame([alert])[self.feature_names].fillna(0).astype('float32')
                
                # Get probability predictions
                incident_proba = self.incident_model.predict_proba(input_df)[0]
                severity_proba = self.severity_model.predict_proba(input_df)[0]
                
                max_incident_conf = np.max(incident_proba)
                max_severity_conf = np.max(severity_proba)
                
                confidence_data.append({
                    'alert_id': i+1,
                    'incident_confidence': max_incident_conf,
                    'severity_confidence': max_severity_conf,
                    'incident_prediction': self.incident_model.predict(input_df)[0],
                    'severity_prediction': self.severity_model.predict(input_df)[0]
                })
                
                logger.info(f"Alert {i+1}: Incident Confidence={max_incident_conf:.3f}, Severity Confidence={max_severity_conf:.3f}")
                
            except Exception as e:
                logger.error(f"❌ Confidence analysis error for alert {i+1}: {e}")
        
        return confidence_data
    
    def run_comprehensive_test(self):
        """Run all tests comprehensively"""
        logger.info("🚀 STARTING COMPREHENSIVE MODEL TEST")
        logger.info("="*60)
        
        # 1. Test model architecture
        self.test_model_architecture()
        
        # 2. Analyze feature importance
        importance_df = self.test_feature_importance()
        
        # 3. Generate test alerts
        test_alerts = self.generate_test_alerts(5)
        
        # 4. Test single predictions
        single_results = self.test_single_predictions(test_alerts)
        
        # 5. Test batch predictions
        batch_results = self.test_batch_predictions()
        
        # 6. Test anomaly detection
        anomaly_results = self.test_anomaly_detection(test_alerts)
        
        # 7. Test model confidence
        confidence_results = self.test_model_confidence(test_alerts)
        
        # Summary
        logger.info("\n" + "="*50)
        logger.info("📋 COMPREHENSIVE TEST SUMMARY")
        logger.info("="*50)
        
        logger.info("✅ Model Architecture: Loaded and analyzed successfully")
        logger.info(f"✅ Feature Importance: Analyzed {len(importance_df)} features")
        logger.info(f"✅ Single Predictions: Tested {len(single_results)} alerts")
        logger.info(f"✅ Batch Predictions: Accuracy = {batch_results['incident_accuracy']:.3f}" if batch_results else "❌ Failed")
        logger.info(f"✅ Anomaly Detection: Tested on {len(test_alerts)} alerts")
        logger.info(f"✅ Confidence Analysis: Completed for {len(confidence_results)} alerts")
        
        return {
            'importance': importance_df,
            'single_results': single_results,
            'batch_results': batch_results,
            'anomaly_results': anomaly_results,
            'confidence_results': confidence_results
        }

def quick_test():
    """Quick test function for fast verification"""
    logger.info("⚡ QUICK MODEL TEST")
    
    tester = ModelTester()
    
    # Quick feature importance check
    importance_df = tester.test_feature_importance()
    
    # Test with just 2 alerts
    test_alerts = tester.generate_test_alerts(2)
    single_results = tester.test_single_predictions(test_alerts)
    
    logger.info("✅ Quick test completed!")
    return single_results

if __name__ == "__main__":
    print("Security Model Testing Suite")
    print("Choose test type:")
    print("1. Quick Test")
    print("2. Comprehensive Test")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        results = quick_test()
    elif choice == "2":
        tester = ModelTester()
        results = tester.run_comprehensive_test()
    else:
        print("Invalid choice. Running quick test...")
        results = quick_test()
    
    print(f"\n🎉 Testing completed! Check logs for detailed results.")
    print("📊 Generated files:")
    print("   - feature_importance_analysis.png")
    print("   - Check console for prediction results")