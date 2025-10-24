"""
Real-Time Security Analysis Dashboard
Single-file implementation for monitoring security incidents with ML predictions
Author: RicheByte
Date: 2025-10-24
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import logging
from collections import deque
import json
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dashboard.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Security Analysis Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .alert-critical {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .alert-high {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .alert-medium {
        background-color: #fff9c4;
        border-left: 4px solid #ffeb3b;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .alert-low {
        background-color: #e8f5e9;
        border-left: 4px solid #4caf50;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)


class RealtimeSecurityDashboard:
    """Main dashboard class for real-time security monitoring"""
    
    def __init__(self):
        self.initialize_session_state()
        self.load_models()
        self.load_feature_names()
        
    def initialize_session_state(self):
        """Initialize Streamlit session state variables"""
        if 'alert_history' not in st.session_state:
            st.session_state.alert_history = deque(maxlen=1000)
        if 'prediction_history' not in st.session_state:
            st.session_state.prediction_history = deque(maxlen=1000)
        if 'anomaly_history' not in st.session_state:
            st.session_state.anomaly_history = deque(maxlen=1000)
        if 'start_time' not in st.session_state:
            st.session_state.start_time = datetime.now()
        if 'alert_counter' not in st.session_state:
            st.session_state.alert_counter = 0
        if 'is_monitoring' not in st.session_state:
            st.session_state.is_monitoring = False
            
    def load_models(self):
        """Load pre-trained ML models"""
        try:
            self.incident_model = joblib.load('incident_model.joblib')
            self.severity_model = joblib.load('severity_model.joblib')
            self.anomaly_model = joblib.load('anomaly_model.joblib')
            logger.info("✅ Models loaded successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Error loading models: {e}")
            st.error(f"Failed to load models: {e}")
            return False
            
    def load_feature_names(self):
        """Load feature names from training data"""
        try:
            train_df = pd.read_parquet("train_optimized.parquet")
            # Assuming feature engineering similar to test_model_workings.py
            self.feature_names = [col for col in train_df.columns 
                                if col not in ['IsIncident', 'Severity', 'AlertId']][:20]
            logger.info(f"📊 Loaded {len(self.feature_names)} features")
        except Exception as e:
            logger.error(f"❌ Error loading features: {e}")
            self.feature_names = []
            
    def generate_realtime_alert(self):
        """Generate a simulated real-time security alert"""
        alert = {}
        
        # Generate realistic feature values
        alert['SuspicionLevel'] = np.random.choice([0.1, 0.3, 0.5, 0.7, 0.9], p=[0.4, 0.3, 0.15, 0.1, 0.05])
        alert['mitre_count'] = np.random.poisson(2)
        alert['ThreatScore'] = np.random.uniform(0, 1)
        alert['FailedLoginAttempts'] = np.random.poisson(3)
        alert['NetworkTraffic'] = np.random.exponential(1000)
        alert['ProcessCount'] = np.random.randint(1, 50)
        alert['FileModifications'] = np.random.poisson(5)
        alert['RegistryChanges'] = np.random.poisson(2)
        alert['PortScans'] = np.random.poisson(1)
        alert['DNSRequests'] = np.random.poisson(10)
        
        # Add more features to match training data
        for feature in self.feature_names:
            if feature not in alert:
                alert[feature] = np.random.uniform(0, 1)
        
        # Add metadata
        alert['timestamp'] = datetime.now()
        alert['alert_id'] = f"ALT-{st.session_state.alert_counter:06d}"
        st.session_state.alert_counter += 1
        
        return alert
    
    def predict_alert(self, alert):
        """Make predictions on a single alert"""
        try:
            # Prepare features
            feature_values = [alert.get(f, 0) for f in self.feature_names]
            X = np.array(feature_values).reshape(1, -1)
            
            # Predictions
            incident_pred = self.incident_model.predict(X)[0]
            incident_proba = self.incident_model.predict_proba(X)[0]
            severity_pred = self.severity_model.predict(X)[0]
            severity_proba = self.severity_model.predict_proba(X)[0]
            anomaly_pred = self.anomaly_model.predict(X)[0]
            anomaly_score = self.anomaly_model.decision_function(X)[0]
            
            prediction = {
                'alert_id': alert['alert_id'],
                'timestamp': alert['timestamp'],
                'incident_prediction': bool(incident_pred),
                'incident_confidence': float(np.max(incident_proba)),
                'severity_level': int(severity_pred),
                'severity_confidence': float(np.max(severity_proba)),
                'is_anomaly': bool(anomaly_pred == -1),
                'anomaly_score': float(anomaly_score),
                'risk_score': self.calculate_risk_score(incident_proba, severity_pred, anomaly_score)
            }
            
            return prediction
            
        except Exception as e:
            logger.error(f"❌ Prediction error: {e}")
            return None
    
    def calculate_risk_score(self, incident_proba, severity, anomaly_score):
        """Calculate overall risk score"""
        incident_weight = np.max(incident_proba) * 0.4
        severity_weight = (severity / 4) * 0.4  # Assuming severity 0-4
        anomaly_weight = (1 if anomaly_score < 0 else 0) * 0.2
        return min(incident_weight + severity_weight + anomaly_weight, 1.0)
    
    def render_header(self):
        """Render dashboard header"""
        st.markdown('<h1 class="main-header">🛡️ Real-Time Security Analysis Dashboard</h1>', 
                   unsafe_allow_html=True)
        st.markdown(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        st.markdown("---")
        
    def render_control_panel(self):
        """Render monitoring control panel"""
        st.sidebar.title("⚙️ Control Panel")
        
        # Start/Stop monitoring
        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button("▶️ Start", use_container_width=True):
                st.session_state.is_monitoring = True
        with col2:
            if st.button("⏸️ Pause", use_container_width=True):
                st.session_state.is_monitoring = False
        
        # Refresh rate
        refresh_rate = st.sidebar.slider("Refresh Rate (seconds)", 1, 10, 3)
        
        # Alert generation rate
        alert_rate = st.sidebar.slider("Alerts per Update", 1, 10, 3)
        
        # Filters
        st.sidebar.markdown("### 🔍 Filters")
        show_incidents_only = st.sidebar.checkbox("Show Incidents Only", False)
        show_anomalies_only = st.sidebar.checkbox("Show Anomalies Only", False)
        min_risk_score = st.sidebar.slider("Minimum Risk Score", 0.0, 1.0, 0.0)
        
        # Export options
        st.sidebar.markdown("### 💾 Export")
        if st.sidebar.button("Export History to CSV"):
            self.export_history()
        
        if st.sidebar.button("Export to JSON"):
            self.export_json()
            
        # Clear history
        if st.sidebar.button("🗑️ Clear History", type="secondary"):
            self.clear_history()
            
        return {
            'refresh_rate': refresh_rate,
            'alert_rate': alert_rate,
            'show_incidents_only': show_incidents_only,
            'show_anomalies_only': show_anomalies_only,
            'min_risk_score': min_risk_score
        }
    
    def render_metrics(self):
        """Render key metrics"""
        col1, col2, col3, col4, col5 = st.columns(5)
        
        total_alerts = len(st.session_state.prediction_history)
        incidents = sum(1 for p in st.session_state.prediction_history if p['incident_prediction'])
        anomalies = sum(1 for p in st.session_state.prediction_history if p['is_anomaly'])
        
        avg_risk = np.mean([p['risk_score'] for p in st.session_state.prediction_history]) if total_alerts > 0 else 0
        high_risk = sum(1 for p in st.session_state.prediction_history if p['risk_score'] > 0.7)
        
        with col1:
            st.metric("📊 Total Alerts", total_alerts)
        with col2:
            st.metric("🚨 Incidents", incidents, delta=f"{(incidents/total_alerts*100):.1f}%" if total_alerts > 0 else "0%")
        with col3:
            st.metric("⚠️ Anomalies", anomalies, delta=f"{(anomalies/total_alerts*100):.1f}%" if total_alerts > 0 else "0%")
        with col4:
            st.metric("📈 Avg Risk Score", f"{avg_risk:.2f}")
        with col5:
            st.metric("🔴 High Risk Alerts", high_risk)
    
    def render_realtime_chart(self):
        """Render real-time prediction chart"""
        st.subheader("📊 Real-Time Predictions")
        
        if len(st.session_state.prediction_history) == 0:
            st.info("Waiting for alerts...")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(list(st.session_state.prediction_history))
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Incident Predictions", "Risk Scores", "Severity Distribution", "Anomaly Detection"),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"type": "pie"}, {"secondary_y": False}]]
        )
        
        # Incident predictions over time
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['incident_confidence'],
                mode='lines+markers',
                name='Incident Confidence',
                line=dict(color='red', width=2),
                marker=dict(size=6)
            ),
            row=1, col=1
        )
        
        # Risk scores
        colors = ['red' if r > 0.7 else 'orange' if r > 0.4 else 'green' for r in df['risk_score']]
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['risk_score'],
                mode='markers',
                name='Risk Score',
                marker=dict(size=10, color=colors, line=dict(width=1, color='white'))
            ),
            row=1, col=2
        )
        
        # Severity distribution
        severity_counts = df['severity_level'].value_counts()
        fig.add_trace(
            go.Pie(
                labels=[f"Level {i}" for i in severity_counts.index],
                values=severity_counts.values,
                name='Severity'
            ),
            row=2, col=1
        )
        
        # Anomaly scores
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['anomaly_score'],
                mode='markers',
                name='Anomaly Score',
                marker=dict(
                    size=8,
                    color=df['is_anomaly'],
                    colorscale=['green', 'red'],
                    showscale=True
                )
            ),
            row=2, col=2
        )
        
        fig.update_layout(height=600, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
    
    def render_feature_importance(self):
        """Render feature importance chart"""
        st.subheader("📈 Feature Importance")
        
        try:
            importance = self.incident_model.feature_importances_[:10]
            features = self.feature_names[:10]
            
            fig = go.Figure(go.Bar(
                x=importance,
                y=features,
                orientation='h',
                marker=dict(color=importance, colorscale='Viridis')
            ))
            
            fig.update_layout(
                title="Top 10 Most Important Features",
                xaxis_title="Importance",
                yaxis_title="Feature",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error rendering feature importance: {e}")
    
    def render_alert_feed(self, filters):
        """Render live alert feed"""
        st.subheader("🚨 Live Alert Feed")
        
        if len(st.session_state.prediction_history) == 0:
            st.info("No alerts yet...")
            return
        
        # Filter predictions
        predictions = list(st.session_state.prediction_history)
        
        if filters['show_incidents_only']:
            predictions = [p for p in predictions if p['incident_prediction']]
        if filters['show_anomalies_only']:
            predictions = [p for p in predictions if p['is_anomaly']]
        if filters['min_risk_score'] > 0:
            predictions = [p for p in predictions if p['risk_score'] >= filters['min_risk_score']]
        
        # Show latest 10 alerts
        for pred in reversed(predictions[-10:]):
            self.render_alert_card(pred)
    
    def render_alert_card(self, prediction):
        """Render individual alert card"""
        risk_score = prediction['risk_score']
        
        if risk_score > 0.7:
            alert_class = "alert-critical"
            emoji = "🔴"
            risk_label = "CRITICAL"
        elif risk_score > 0.5:
            alert_class = "alert-high"
            emoji = "🟠"
            risk_label = "HIGH"
        elif risk_score > 0.3:
            alert_class = "alert-medium"
            emoji = "🟡"
            risk_label = "MEDIUM"
        else:
            alert_class = "alert-low"
            emoji = "🟢"
            risk_label = "LOW"
        
        incident_status = "✅ INCIDENT" if prediction['incident_prediction'] else "❌ No Incident"
        anomaly_status = "⚠️ ANOMALY" if prediction['is_anomaly'] else "✓ Normal"
        
        st.markdown(f"""
        <div class="{alert_class}">
            <strong>{emoji} {prediction['alert_id']}</strong> - {risk_label} RISK
            <br>
            <small>Time: {prediction['timestamp'].strftime('%H:%M:%S')}</small>
            <br>
            {incident_status} | Confidence: {prediction['incident_confidence']:.2%}
            <br>
            Severity: Level {prediction['severity_level']} | {anomaly_status}
            <br>
            Risk Score: {risk_score:.2%}
        </div>
        """, unsafe_allow_html=True)
    
    def render_statistics(self):
        """Render detailed statistics"""
        st.subheader("📊 Detailed Statistics")
        
        if len(st.session_state.prediction_history) == 0:
            st.info("No data available yet...")
            return
        
        df = pd.DataFrame(list(st.session_state.prediction_history))
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Incident Statistics")
            st.write(f"- Total Incidents: {df['incident_prediction'].sum()}")
            st.write(f"- Average Confidence: {df['incident_confidence'].mean():.2%}")
            st.write(f"- Max Confidence: {df['incident_confidence'].max():.2%}")
            st.write(f"- Min Confidence: {df['incident_confidence'].min():.2%}")
            
        with col2:
            st.markdown("#### Anomaly Statistics")
            st.write(f"- Total Anomalies: {df['is_anomaly'].sum()}")
            st.write(f"- Anomaly Rate: {(df['is_anomaly'].sum() / len(df)):.2%}")
            st.write(f"- Average Anomaly Score: {df['anomaly_score'].mean():.3f}")
        
        # Time-based analysis
        st.markdown("#### Time-Based Analysis")
        uptime = datetime.now() - st.session_state.start_time
        alerts_per_minute = len(df) / (uptime.total_seconds() / 60) if uptime.total_seconds() > 0 else 0
        
        st.write(f"- Uptime: {str(uptime).split('.')[0]}")
        st.write(f"- Alerts per Minute: {alerts_per_minute:.2f}")
        st.write(f"- Total Alerts Processed: {len(df)}")
    
    def render_model_performance(self):
        """Render model performance metrics"""
        st.subheader("🎯 Model Performance")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### Incident Model")
            st.write(f"- Type: {type(self.incident_model).__name__}")
            st.write(f"- Trees: {self.incident_model.n_estimators}")
            st.write(f"- Features: {len(self.feature_names)}")
            
        with col2:
            st.markdown("#### Severity Model")
            st.write(f"- Type: {type(self.severity_model).__name__}")
            st.write(f"- Trees: {self.severity_model.n_estimators}")
            st.write(f"- Classes: {len(self.severity_model.classes_)}")
            
        with col3:
            st.markdown("#### Anomaly Model")
            st.write(f"- Type: {type(self.anomaly_model).__name__}")
            st.write(f"- Trees: {self.anomaly_model.n_estimators}")
            st.write(f"- Contamination: {self.anomaly_model.contamination}")
    
    def export_history(self):
        """Export prediction history to CSV"""
        if len(st.session_state.prediction_history) == 0:
            st.warning("No data to export")
            return
        
        df = pd.DataFrame(list(st.session_state.prediction_history))
        filename = f"security_alerts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(filename, index=False)
        st.success(f"✅ Exported to {filename}")
        logger.info(f"Exported {len(df)} records to {filename}")
    
    def export_json(self):
        """Export prediction history to JSON"""
        if len(st.session_state.prediction_history) == 0:
            st.warning("No data to export")
            return
        
        data = []
        for pred in st.session_state.prediction_history:
            pred_copy = pred.copy()
            pred_copy['timestamp'] = pred_copy['timestamp'].isoformat()
            data.append(pred_copy)
        
        filename = f"security_alerts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        st.success(f"✅ Exported to {filename}")
        logger.info(f"Exported {len(data)} records to {filename}")
    
    def clear_history(self):
        """Clear all history"""
        st.session_state.alert_history.clear()
        st.session_state.prediction_history.clear()
        st.session_state.anomaly_history.clear()
        st.session_state.alert_counter = 0
        st.session_state.start_time = datetime.now()
        st.success("✅ History cleared")
        logger.info("History cleared")
    
    def run(self):
        """Main dashboard run loop"""
        self.render_header()
        
        # Control panel
        filters = self.render_control_panel()
        
        # Main content
        self.render_metrics()
        st.markdown("---")
        
        # Create tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Real-Time", "🚨 Alert Feed", "📈 Analytics", "⚙️ System"])
        
        with tab1:
            self.render_realtime_chart()
            
        with tab2:
            self.render_alert_feed(filters)
            
        with tab3:
            col1, col2 = st.columns(2)
            with col1:
                self.render_feature_importance()
            with col2:
                self.render_statistics()
                
        with tab4:
            self.render_model_performance()
        
        # Auto-refresh and alert generation
        if st.session_state.is_monitoring:
            with st.spinner("Monitoring..."):
                # Generate new alerts
                for _ in range(filters['alert_rate']):
                    alert = self.generate_realtime_alert()
                    prediction = self.predict_alert(alert)
                    
                    if prediction:
                        st.session_state.prediction_history.append(prediction)
                        st.session_state.alert_history.append(alert)
                
                # Auto-refresh
                time.sleep(filters['refresh_rate'])
                st.rerun()
        else:
            st.info("⏸️ Monitoring paused. Click 'Start' to resume.")


def main():
    """Main entry point"""
    try:
        dashboard = RealtimeSecurityDashboard()
        dashboard.run()
    except Exception as e:
        st.error(f"❌ Dashboard error: {e}")
        logger.error(f"Dashboard error: {e}", exc_info=True)


if __name__ == "__main__":
    main()