import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, f1_score, roc_curve
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import lightgbm as lgb
import joblib
from sklearn.utils.class_weight import compute_class_weight
import warnings
import logging
import gc
import os

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# GPU configuration
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class UltraOptimizedSecurityPredictor:
    def __init__(self):
        self.incident_model = None
        self.severity_model = None
        self.anomaly_model = None
        self.feature_names = []
        self.categorical_cols = ['Category', 'EntityType', 'OSFamily', 'CountryCode']
        self.numerical_cols = ['DetectorId', 'DeviceId', 'SuspicionLevel']
        self.preprocessor = None
        
    def load_data_chunked(self, dataset_path, sample_frac=0.2, chunksize=50000):
        """Optimized chunked loading with dtype optimization"""
        logger.info("📥 Loading data in chunks...")
        
        try:
            all_files = [f for f in os.listdir(dataset_path) if f.endswith('.csv')]
            train_files = [f for f in all_files if 'train' in f.lower()]
            test_files = [f for f in all_files if 'test' in f.lower()]
            
            if not train_files or not test_files:
                if len(all_files) >= 2:
                    train_files, test_files = [all_files[0]], [all_files[1]]
                else:
                    raise FileNotFoundError("Could not find train/test CSV files")
            
            train_path = os.path.join(dataset_path, train_files[0])
            test_path = os.path.join(dataset_path, test_files[0])
            
            # Define dtypes for memory efficiency
            dtype_dict = {
                'DetectorId': 'int32',
                'DeviceId': 'int32',
                'SuspicionLevel': 'float32',
                'hour': 'int8',
                'day_of_week': 'int8'
            }
            
            # Load training data
            logger.info("Processing training data in chunks...")
            train_chunks = [
                self.optimize_data_types(chunk.sample(frac=sample_frac, random_state=42))
                for chunk in pd.read_csv(train_path, chunksize=chunksize, dtype=dtype_dict)
            ]
            train_df = pd.concat(train_chunks, ignore_index=True)
            del train_chunks
            gc.collect()
            
            # Load test data
            logger.info("Processing test data in chunks...")
            test_chunks = [
                self.optimize_data_types(chunk.sample(frac=sample_frac, random_state=42))
                for chunk in pd.read_csv(test_path, chunksize=chunksize, dtype=dtype_dict)
            ]
            test_df = pd.concat(test_chunks, ignore_index=True)
            del test_chunks
            gc.collect()
            
            logger.info(f"✅ Final dataset: Train {train_df.shape}, Test {test_df.shape}")
            logger.info(f"💿 Memory usage: {train_df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
            
            # Save as parquet
            train_df.to_parquet("train_optimized.parquet", index=False, compression='snappy')
            test_df.to_parquet("test_optimized.parquet", index=False, compression='snappy')
            logger.info("💾 Saved optimized datasets as Parquet files")
            
            return train_df, test_df
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            try:
                train_df = pd.read_parquet("train_optimized.parquet")
                test_df = pd.read_parquet("test_optimized.parquet")
                logger.info("✅ Loaded from cached Parquet files")
                return train_df, test_df
            except Exception as e2:
                logger.error(f"Failed to load cached data: {e2}")
                return None, None

    def optimize_data_types(self, df):
        """Optimize data types for memory efficiency"""
        for col in df.columns:
            if df[col].dtype == 'object':
                unique_ratio = df[col].nunique() / len(df)
                if unique_ratio < 0.5:
                    df[col] = df[col].astype('category')
            elif df[col].dtype in ['int64', 'int32']:
                df[col] = pd.to_numeric(df[col], downcast='integer')
            elif df[col].dtype in ['float64', 'float32']:
                df[col] = pd.to_numeric(df[col], downcast='float')
        return df

    def efficient_feature_engineering(self, df):
        """Efficient feature engineering with reduced memory footprint"""
        logger.info("🔧 Efficient feature engineering...")
        
        # Convert categories to objects for processing
        for col in df.columns:
            if df[col].dtype.name == 'category':
                df[col] = df[col].astype('object')
        
        df_processed = df.copy()
        
        # 1. Create target variables
        if 'IncidentGrade' in df_processed.columns:
            df_processed['IsIncident'] = df_processed['IncidentGrade'].notna().astype('int8')
            
            severity_mapping = {'Low': 1, 'Medium': 2, 'High': 3, 'Critical': 4}
            if df_processed['IncidentGrade'].dtype == 'object':
                df_processed['Severity'] = df_processed['IncidentGrade'].map(severity_mapping).fillna(0).astype('int8')
            else:
                df_processed['Severity'] = self._create_severity_binning(df_processed['IncidentGrade'])
        
        # 2. Time features
        if 'Timestamp' in df_processed.columns:
            df_processed['Timestamp'] = pd.to_datetime(df_processed['Timestamp'], errors='coerce')
            df_processed['hour'] = df_processed['Timestamp'].dt.hour.astype('int8')
            df_processed['day_of_week'] = df_processed['Timestamp'].dt.dayofweek.astype('int8')
            df_processed['is_weekend'] = (df_processed['day_of_week'] >= 5).astype('int8')
            df_processed['is_business_hours'] = ((df_processed['hour'] >= 9) & (df_processed['hour'] <= 17)).astype('int8')
        
        # 3. Categorical encoding with frequency encoding
        for col in self.categorical_cols:
            if col in df_processed.columns and df_processed[col].dtype == 'object':
                freq_map = df_processed[col].value_counts().to_dict()
                df_processed[f'{col}_freq'] = df_processed[col].map(freq_map).fillna(0).astype('int16')
                df_processed[f'{col}_is_rare'] = (df_processed[f'{col}_freq'] < 10).astype('int8')
        
        # 4. MITRE techniques - simple count
        if 'MitreTechniques' in df_processed.columns:
            df_processed['mitre_count'] = df_processed['MitreTechniques'].apply(
                lambda x: len(str(x).split(';')) if pd.notna(x) else 0
            ).astype('int8')
            df_processed['has_mitre'] = (df_processed['mitre_count'] > 0).astype('int8')
        
        # 5. Numerical features
        for col in self.numerical_cols:
            if col in df_processed.columns:
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce').fillna(0).astype('float32')
        
        return df_processed

    def _create_severity_binning(self, incident_grades):
        """Create severity binning with error handling"""
        severity = pd.Series(0, index=incident_grades.index, dtype='int8')
        incident_mask = incident_grades.notna()
        
        if incident_mask.sum() == 0:
            return severity
        
        incident_vals = incident_grades[incident_mask]
        
        try:
            severity_bins = pd.qcut(incident_vals, q=4, labels=[1, 2, 3, 4], duplicates='drop')
            severity.loc[incident_mask] = severity_bins.astype('int8').values
        except ValueError:
            logger.warning("Quantile binning failed, using threshold-based approach")
            mean_val = incident_vals.mean()
            std_val = incident_vals.std()
            conditions = [
                incident_vals <= mean_val - std_val,
                incident_vals <= mean_val,
                incident_vals <= mean_val + std_val,
                incident_vals > mean_val + std_val
            ]
            severity.loc[incident_mask] = np.select(conditions, [1, 2, 3, 4], default=2).astype('int8')
        
        return severity

    def select_features_correlation(self, df, target_col='IsIncident', top_k=20):
        """Select top-k features by correlation - TRAIN ONLY"""
        logger.info(f"🎯 Selecting top {top_k} features by correlation...")
        
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numerical_cols = [col for col in numerical_cols if col != target_col]
        
        if target_col not in df.columns:
            logger.warning(f"Target column '{target_col}' not found. Using all numerical features.")
            self.feature_names = numerical_cols[:top_k]
            return df[self.feature_names]
        
        correlations = []
        for col in numerical_cols:
            if df[col].nunique() > 1:
                try:
                    corr = np.abs(df[col].corr(df[target_col]))
                    correlations.append((col, corr))
                except Exception as e:
                    logger.debug(f"Skipping {col}: {e}")
                    continue
        
        correlations.sort(key=lambda x: x[1], reverse=True)
        self.feature_names = [col for col, corr in correlations[:top_k]]
        
        logger.info(f"✅ Selected {len(self.feature_names)} features")
        logger.info("Top 10 features by correlation:")
        for col, corr in correlations[:10]:
            logger.info(f"  {col}: {corr:.4f}")
        
        return df[self.feature_names]

    def train_ultra_optimized_models(self, X_train, y_incident_train, y_severity_train):
        """Ultra-optimized model training with early stopping"""
        logger.info("🚀 Training Ultra-Optimized Models...")
        
        # Fill missing values
        X_train = X_train.fillna(0).astype('float32')
        
        classes = np.unique(y_incident_train)
        weights = compute_class_weight('balanced', classes=classes, y=y_incident_train)
        class_weights = dict(zip(classes, weights))
        
        logger.info(f"⚖️ Class weights: {class_weights}")
        
        # Common LightGBM parameters
        common_params = {
            'n_estimators': 80,
            'learning_rate': 0.15,
            'max_depth': 5,
            'num_leaves': 15,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': -1,
            'deterministic': True,
            'max_bin': 255
        }
        
        # 1. LightGBM for Incident Prediction
        logger.info("🎯 Training LightGBM for Incident Prediction...")
        try:
            self.incident_model = lgb.LGBMClassifier(
                **common_params,
                class_weight=class_weights,
                device='gpu',
                gpu_platform_id=0,
                gpu_device_id=0
            )
            self.incident_model.fit(
                X_train, y_incident_train,
                eval_set=[(X_train, y_incident_train)],
                eval_metric='auc',
                callbacks=[lgb.early_stopping(10)]
            )
            logger.info("✅ GPU training completed for incident model")
        except Exception as e:
            logger.warning(f"GPU training failed: {e}. Falling back to CPU...")
            self.incident_model = lgb.LGBMClassifier(**common_params, class_weight=class_weights)
            self.incident_model.fit(
                X_train, y_incident_train,
                eval_set=[(X_train, y_incident_train)],
                eval_metric='auc',
                callbacks=[lgb.early_stopping(10)]
            )
        
        # 2. LightGBM for Severity
        logger.info("📊 Training LightGBM for Severity Prediction...")
        try:
            self.severity_model = lgb.LGBMClassifier(
                n_estimators=60,
                learning_rate=0.15,
                max_depth=5,
                num_leaves=15,
                random_state=42,
                n_jobs=-1,
                verbosity=-1,
                deterministic=True,
                device='gpu'
            )
            self.severity_model.fit(
                X_train, y_severity_train,
                eval_set=[(X_train, y_severity_train)],
                callbacks=[lgb.early_stopping(10)]
            )
        except Exception:
            self.severity_model = lgb.LGBMClassifier(
                n_estimators=60,
                learning_rate=0.15,
                max_depth=5,
                num_leaves=15,
                random_state=42,
                n_jobs=-1,
                verbosity=-1,
                deterministic=True
            )
            self.severity_model.fit(X_train, y_severity_train)
        
        # 3. Anomaly Detection
        logger.info("🔍 Training Anomaly Detection...")
        anomaly_sample_size = min(5000, len(X_train))
        X_anomaly = X_train.sample(n=anomaly_sample_size, random_state=42)
        
        self.anomaly_model = IsolationForest(
            n_estimators=30,
            contamination=0.1,
            random_state=42,
            n_jobs=-1,
            max_samples=256
        )
        self.anomaly_model.fit(X_anomaly)
        
        # Save models with compression
        joblib.dump(self.incident_model, 'incident_model.joblib', compress=3)
        joblib.dump(self.severity_model, 'severity_model.joblib', compress=3)
        joblib.dump(self.anomaly_model, 'anomaly_model.joblib', compress=3)
        
        logger.info("✅ All models trained and saved!")

    def evaluate_with_plots(self, X_test, y_incident_test, y_severity_test):
        """Comprehensive evaluation with proper error handling"""
        logger.info("📈 Model Evaluation")
        logger.info("=" * 50)
        
        if self.incident_model is None:
            try:
                self.incident_model = joblib.load('incident_model.joblib')
                self.severity_model = joblib.load('severity_model.joblib')
                self.anomaly_model = joblib.load('anomaly_model.joblib')
                logger.info("✅ Loaded pre-trained models")
            except FileNotFoundError:
                logger.error("❌ No trained models found")
                return None
        
        # Ensure feature alignment and fill missing values
        X_test = X_test[self.feature_names].fillna(0).astype('float32')
        
        # Incident Prediction
        incident_pred = self.incident_model.predict(X_test)
        incident_proba = self.incident_model.predict_proba(X_test)[:, 1]
        
        # Check for single class in test set
        if len(np.unique(y_incident_test)) < 2:
            incident_auc = float('nan')
            logger.warning("⚠️ Test set has only one class - ROC-AUC is undefined")
        else:
            incident_auc = roc_auc_score(y_incident_test, incident_proba)
        
        logger.info("🎯 INCIDENT PREDICTION:")
        logger.info(f"Accuracy: {accuracy_score(y_incident_test, incident_pred):.4f}")
        logger.info(f"ROC-AUC: {incident_auc:.4f}")
        logger.info(f"F1-Score: {f1_score(y_incident_test, incident_pred, zero_division=0):.4f}")
        
        # Severity Prediction
        severity_pred = self.severity_model.predict(X_test)
        logger.info(f"📊 SEVERITY PREDICTION:")
        logger.info(f"Accuracy: {accuracy_score(y_severity_test, severity_pred):.4f}")
        
        self.create_memory_efficient_plots(X_test, y_incident_test, y_severity_test, 
                                         incident_pred, severity_pred, incident_proba)
        
        return {
            'incident_accuracy': accuracy_score(y_incident_test, incident_pred),
            'incident_auc': incident_auc,
            'severity_accuracy': accuracy_score(y_severity_test, severity_pred)
        }

    def create_memory_efficient_plots(self, X_test, y_incident_test, y_severity_test, 
                                    incident_pred, severity_pred, incident_proba):
        """Create memory-efficient visualizations"""
        logger.info("📊 Creating visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Confusion Matrix
        cm = confusion_matrix(y_incident_test, incident_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
        axes[0, 0].set_title('Incident Confusion Matrix')
        axes[0, 0].set_ylabel('True Label')
        axes[0, 0].set_xlabel('Predicted Label')
        
        # 2. Severity Distribution
        severity_counts = pd.Series(severity_pred).value_counts().sort_index()
        colors = ['green', 'lightblue', 'orange', 'red', 'darkred']
        axes[0, 1].bar(severity_counts.index, severity_counts.values, color=colors[:len(severity_counts)])
        axes[0, 1].set_title('Predicted Severity Distribution')
        axes[0, 1].set_xlabel('Severity Level')
        axes[0, 1].set_ylabel('Count')
        
        # 3. Feature Importance
        if hasattr(self.incident_model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.incident_model.feature_importances_
            }).nlargest(8, 'importance')
            
            axes[1, 0].barh(importance_df['feature'], importance_df['importance'], color='steelblue')
            axes[1, 0].set_title('Top 8 Feature Importances')
            axes[1, 0].set_xlabel('Importance')
        
        # 4. ROC Curve
        if len(np.unique(y_incident_test)) >= 2:
            fpr, tpr, _ = roc_curve(y_incident_test, incident_proba)
            auc_score = roc_auc_score(y_incident_test, incident_proba)
            
            axes[1, 1].plot(fpr, tpr, label=f'AUC = {auc_score:.3f}', linewidth=2)
            axes[1, 1].plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
            axes[1, 1].set_title('ROC Curve')
            axes[1, 1].set_xlabel('False Positive Rate')
            axes[1, 1].set_ylabel('True Positive Rate')
            axes[1, 1].legend()
            axes[1, 1].grid(alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'Not enough classes for ROC curve', ha='center', va='center')
        
        plt.tight_layout()
        plt.savefig('optimized_security_dashboard.png', dpi=150, bbox_inches='tight')
        logger.info("📊 Dashboard saved as 'optimized_security_dashboard.png'")
        plt.show()

    def predict_single_optimized(self, features_dict):
        """Optimized prediction for single alert using DataFrame"""
        if self.incident_model is None:
            try:
                self.incident_model = joblib.load('incident_model.joblib')
                self.severity_model = joblib.load('severity_model.joblib')
            except FileNotFoundError:
                return {"error": "Models not trained yet"}
        
        try:
            # Convert to DataFrame and align with training features
            input_df = pd.DataFrame([features_dict])[self.feature_names].fillna(0).astype('float32')
            
            incident_pred = self.incident_model.predict(input_df)[0]
            incident_proba = self.incident_model.predict_proba(input_df)[0, 1]
            severity_pred = self.severity_model.predict(input_df)[0]
            
            return {
                "incident_prediction": bool(incident_pred),
                "incident_confidence": float(incident_proba),
                "severity_level": int(severity_pred),
                "risk_category": "HIGH" if incident_proba > 0.7 else "MEDIUM" if incident_proba > 0.3 else "LOW"
            }
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {"error": str(e)}

def check_system_resources():
    """Check system resources"""
    logger.info("🩺 BASIC SYSTEM CHECK")
    logger.info("=" * 40)
    
    try:
        import psutil
        memory = psutil.virtual_memory()
        logger.info(f"💾 RAM: {memory.used/1024**3:.1f}GB / {memory.total/1024**3:.1f}GB ({memory.percent}%)")
        
        if memory.percent > 80:
            logger.warning("⚠️ High memory usage - consider reducing sample size")
        else:
            logger.info("✅ Memory adequate for processing")
    except ImportError:
        logger.info("💾 psutil not installed - using default memory settings")
    
    logger.info("✅ System check completed")

def cleanup_memory():
    """Clean up memory"""
    gc.collect()
    logger.info("🧹 Memory cleaned up")

def download_microsoft_security_data():
    """Download dataset with kagglehub"""
    logger.info("📥 Downloading Microsoft Security dataset...")
    try:
        path = kagglehub.dataset_download("microsoft/microsoft-security-incident-prediction")
        logger.info("✅ Dataset downloaded successfully!")
        return path
    except Exception as e:
        logger.error(f"Download error: {e}")
        logger.info("💡 Trying alternative download method...")
        try:
            import kaggle
            kaggle.api.dataset_download_files('microsoft/microsoft-security-incident-prediction', 
                                            path='.', unzip=True)
            logger.info("✅ Dataset downloaded via kaggle API!")
            return '.'
        except Exception as e2:
            logger.error(f"Both methods failed: {e2}")
            logger.info("📌 Download from: https://www.kaggle.com/datasets/microsoft/microsoft-security-incident-prediction")
            return None

def main_ultra_optimized():
    """Main optimized pipeline"""
    logger.info("🚀 ULTRA-OPTIMIZED SECURITY PREDICTION SYSTEM")
    logger.info("💻 Config: i5-12th Gen + RTX 3050 + 8GB RAM")
    logger.info("=" * 60)
    
    predictor = UltraOptimizedSecurityPredictor()
    
    # Load data
    dataset_path = download_microsoft_security_data()
    if dataset_path is None:
        logger.info("🔄 Trying cached data...")
        try:
            train_df = pd.read_parquet("train_optimized.parquet")
            test_df = pd.read_parquet("test_optimized.parquet")
        except FileNotFoundError:
            logger.error("❌ No cached data found. Exiting.")
            return
    else:
        train_df, test_df = predictor.load_data_chunked(dataset_path, sample_frac=0.2, chunksize=50000)
        if train_df is None:
            return
    
    # Feature engineering
    logger.info("Processing training data...")
    train_processed = predictor.efficient_feature_engineering(train_df)
    logger.info("Processing test data...")
    test_processed = predictor.efficient_feature_engineering(test_df)
    
    # Feature selection (train only)
    logger.info("Selecting features from training data...")
    X_train = predictor.select_features_correlation(train_processed, 'IsIncident', top_k=20)
    X_test = test_processed[predictor.feature_names]  # Apply same features to test
    
    y_incident_train = train_processed['IsIncident']
    y_incident_test = test_processed['IsIncident']
    y_severity_train = train_processed['Severity']
    y_severity_test = test_processed['Severity']
    
    logger.info(f"📊 Final dataset shape:")
    logger.info(f"   X_train: {X_train.shape}, X_test: {X_test.shape}")
    logger.info(f"   Memory: {X_train.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    # Training
    logger.info("⏰ Starting training...")
    predictor.train_ultra_optimized_models(X_train, y_incident_train, y_severity_train)
    
    # Evaluation
    logger.info("Evaluating models...")
    results = predictor.evaluate_with_plots(X_test, y_incident_test, y_severity_test)
    
    if results:
        # Sample prediction
        logger.info("🎯 SAMPLE PREDICTIONS:")
        if len(X_test) > 0:
            sample_features = dict(zip(predictor.feature_names, X_test.iloc[0].values))
            prediction = predictor.predict_single_optimized(sample_features)
            logger.info(f"Sample Alert Analysis: {prediction}")
        
        logger.info("✅ PIPELINE COMPLETED!")
        logger.info(f"📊 Final Results:")
        logger.info(f"   Incident AUC: {results['incident_auc']:.4f}")
        logger.info(f"   Incident Accuracy: {results['incident_accuracy']:.4f}")
        logger.info(f"   Severity Accuracy: {results['severity_accuracy']:.4f}")

if __name__ == "__main__":
    check_system_resources()
    
    try:
        main_ultra_optimized()
    except MemoryError:
        logger.error("Memory error - reducing dataset size")
        cleanup_memory()
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    
    cleanup_memory()
    logger.info("🎉 Pipeline execution completed!")