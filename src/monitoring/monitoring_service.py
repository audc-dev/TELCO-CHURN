# monitoring_service.py - Basic monitoring service
import os
import time
import json
import requests
import logging
from datetime import datetime, timedelta
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ChurnModelMonitor:
    """Basic monitoring service for churn prediction model"""
    
    def __init__(self):
        self.fastapi_url = os.getenv('FASTAPI_URL', 'http://fastapi:8000')
        self.monitoring_interval = int(os.getenv('MONITORING_INTERVAL', 300))  # 5 minutes
        self.reports_dir = Path('/app/reports')
        self.reports_dir.mkdir(exist_ok=True)
        
        # Monitoring metrics
        self.last_check = None
        self.health_checks = []
        self.alerts = []
    
    def check_api_health(self):
        """Check FastAPI service health"""
        try:
            response = requests.get(f"{self.fastapi_url}/health", timeout=10)
            
            if response.status_code == 200:
                health_data = response.json()
                logger.info(f"✅ API Health Check: {health_data}")
                
                self.health_checks.append({
                    'timestamp': datetime.now().isoformat(),
                    'status': 'healthy',
                    'response_time_ms': response.elapsed.total_seconds() * 1000,
                    'data': health_data
                })
                
                return True, health_data
            else:
                logger.warning(f"⚠️ API returned status {response.status_code}")
                return False, None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ API Health Check Failed: {e}")
            self.health_checks.append({
                'timestamp': datetime.now().isoformat(),
                'status': 'unhealthy',
                'error': str(e)
            })
            return False, None
    
    def check_model_performance(self):
        """Check model performance metrics"""
        try:
            response = requests.get(f"{self.fastapi_url}/monitoring/performance", timeout=10)
            
            if response.status_code == 200:
                perf_data = response.json()
                logger.info(f"📊 Performance Check: {perf_data}")
                
                # Check for performance issues
                avg_confidence = perf_data.get('average_confidence', 0)
                if avg_confidence < 0.7:
                    self.create_alert(
                        "LOW_CONFIDENCE",
                        f"Average model confidence is low: {avg_confidence:.2%}",
                        "warning"
                    )
                
                recent_predictions = perf_data.get('recent_predictions', 0)
                if recent_predictions == 0:
                    self.create_alert(
                        "NO_PREDICTIONS", 
                        "No recent predictions detected",
                        "info"
                    )
                
                return perf_data
            else:
                logger.warning(f"Performance check returned status {response.status_code}")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Performance Check Failed: {e}")
            return None
    
    def check_data_drift(self):
        """Check for data drift"""
        try:
            response = requests.get(f"{self.fastapi_url}/monitoring/drift", timeout=30)
            
            if response.status_code == 200:
                drift_data = response.json()
                logger.info(f"🔍 Drift Check: {drift_data}")
                
                if drift_data.get('drift_detected', False):
                    self.create_alert(
                        "DATA_DRIFT_DETECTED",
                        "Data drift detected in recent predictions",
                        "warning"
                    )
                
                return drift_data
            else:
                logger.warning(f"Drift check returned status {response.status_code}")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Drift Check Failed: {e}")
            return None
    
    def create_alert(self, alert_type, message, severity):
        """Create monitoring alert"""
        alert = {
            'timestamp': datetime.now().isoformat(),
            'type': alert_type,
            'message': message,
            'severity': severity
        }
        
        self.alerts.append(alert)
        logger.warning(f"🚨 ALERT [{severity.upper()}]: {message}")
        
        # Keep only last 100 alerts
        if len(self.alerts) > 100:
            self.alerts = self.alerts[-100:]
    
    def save_monitoring_report(self):
        """Save monitoring report to file"""
        try:
            report = {
                'monitoring_summary': {
                    'last_check': self.last_check,
                    'total_health_checks': len(self.health_checks),
                    'active_alerts': len([a for a in self.alerts if a['severity'] in ['warning', 'error']]),
                    'monitoring_interval': self.monitoring_interval
                },
                'recent_health_checks': self.health_checks[-10:],  # Last 10
                'active_alerts': [a for a in self.alerts if a['severity'] in ['warning', 'error']],
                'generated_at': datetime.now().isoformat()
            }
            
            report_file = self.reports_dir / f"monitoring_report_{datetime.now().strftime('%Y%m%d')}.json"
            
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"📄 Monitoring report saved: {report_file}")
            
        except Exception as e:
            logger.error(f"Failed to save monitoring report: {e}")
    
    def run_monitoring_cycle(self):
        """Run a complete monitoring cycle"""
        logger.info("🔄 Starting monitoring cycle")
        
        # Check API health
        api_healthy, health_data = self.check_api_health()
        
        if api_healthy:
            # Check model performance
            self.check_model_performance()
            
            # Check data drift (less frequently due to computation cost)
            if len(self.health_checks) % 3 == 0:  # Every 3rd check
                self.check_data_drift()
        
        self.last_check = datetime.now().isoformat()
        
        # Save report daily
        if len(self.health_checks) % 20 == 0:  # Roughly daily with 5min intervals
            self.save_monitoring_report()
        
        logger.info("✅ Monitoring cycle completed")
    
    def run(self):
        """Main monitoring loop"""
        logger.info("🚀 Starting Churn Model Monitoring Service")
        logger.info(f"📍 FastAPI URL: {self.fastapi_url}")
        logger.info(f"⏰ Monitoring Interval: {self.monitoring_interval} seconds")
        
        while True:
            try:
                self.run_monitoring_cycle()
                
            except KeyboardInterrupt:
                logger.info("🛑 Monitoring service stopped by user")
                break
                
            except Exception as e:
                logger.error(f"❌ Monitoring cycle error: {e}")
                self.create_alert("MONITORING_ERROR", f"Monitoring cycle failed: {str(e)}", "error")
            
            # Wait for next cycle
            logger.info(f"😴 Sleeping for {self.monitoring_interval} seconds...")
            time.sleep(self.monitoring_interval)

def main():
    """Main entry point"""
    monitor = ChurnModelMonitor()
    monitor.run()

if __name__ == "__main__":
    main()