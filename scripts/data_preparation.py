#!/usr/bin/env python3
"""
Data Preparation Script for Telco Churn Prediction
This script handles data cleaning, validation, and preprocessing
"""

import os
import pandas as pd
import numpy as np
import yaml
import json
from pathlib import Path
import logging
from typing import Dict, List, Any

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataPreparator:
    """Class to handle data preparation and validation"""
    
    def __init__(self, params_path: str = 'params.yaml'):
        """Initialize with parameters from params.yaml"""
        self.params = self.load_params(params_path)
        self.data_prep_params = self.params['data_prep']
        
        # Create output directories
        self.create_output_dirs()
        
    def load_params(self, params_path: str) -> Dict[str, Any]:
        """Load parameters from YAML file"""
        try:
            with open(params_path, 'r') as file:
                params = yaml.safe_load(file)
            logger.info(f"Parameters loaded from {params_path}")
            return params
        except FileNotFoundError:
            logger.error(f"Parameters file {params_path} not found")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML file: {e}")
            raise
    
    def create_output_dirs(self):
        """Create necessary output directories"""
        dirs_to_create = [
            'data/processed',
            'reports'
        ]
        
        for dir_path in dirs_to_create:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {dir_path}")
    
    def load_raw_data(self, file_path: str) -> pd.DataFrame:
        """Load raw data from CSV file"""
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Raw data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
        except FileNotFoundError:
            logger.error(f"Raw data file not found: {file_path}")
            raise
        except pd.errors.EmptyDataError:
            logger.error(f"Empty data file: {file_path}")
            raise
    
    def validate_raw_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate raw data quality"""
        validation_results = {}
        validation_rules = self.data_prep_params['validation_rules']
        
        # Check minimum samples
        min_samples = validation_rules['min_samples']
        validation_results['min_samples_check'] = df.shape[0] >= min_samples
        validation_results['actual_samples'] = df.shape[0]
        
        # Check missing value percentage
        max_missing_pct = validation_rules['max_missing_percentage']
        missing_pct = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
        validation_results['missing_percentage_check'] = missing_pct <= max_missing_pct
        validation_results['actual_missing_percentage'] = missing_pct
        
        # Check required columns
        required_cols = validation_rules['required_columns']
        missing_required_cols = [col for col in required_cols if col not in df.columns]
        validation_results['required_columns_check'] = len(missing_required_cols) == 0
        validation_results['missing_required_columns'] = missing_required_cols
        
        # Overall validation status
        validation_results['overall_valid'] = all([
            validation_results['min_samples_check'],
            validation_results['missing_percentage_check'],
            validation_results['required_columns_check']
        ])
        
        logger.info(f"Data validation completed. Valid: {validation_results['overall_valid']}")
        return validation_results
    
    def drop_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop specified columns from dataframe"""
        columns_to_drop = self.data_prep_params['drop_columns']
        
        # Only drop columns that exist in the dataframe
        existing_cols_to_drop = [col for col in columns_to_drop if col in df.columns]
        non_existing_cols = [col for col in columns_to_drop if col not in df.columns]
        
        if non_existing_cols:
            logger.warning(f"Columns not found in data: {non_existing_cols}")
        
        if existing_cols_to_drop:
            df_cleaned = df.drop(columns=existing_cols_to_drop)
            logger.info(f"Dropped {len(existing_cols_to_drop)} columns: {existing_cols_to_drop}")
        else:
            df_cleaned = df.copy()
            logger.info("No columns to drop")
        
        return df_cleaned
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values based on strategy"""
        missing_strategy = self.data_prep_params['missing_value_strategy']
        df_filled = df.copy()
        
        # Handle numeric columns
        numeric_cols = df_filled.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0 and missing_strategy['numeric'] == 'median':
            df_filled[numeric_cols] = df_filled[numeric_cols].fillna(df_filled[numeric_cols].median())
            logger.info(f"Filled missing values in numeric columns with median")
        
        # Handle categorical columns
        categorical_cols = df_filled.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0 and missing_strategy['categorical'] == 'most_frequent':
            for col in categorical_cols:
                if df_filled[col].isnull().any():
                    most_frequent = df_filled[col].mode().iloc[0] if not df_filled[col].mode().empty else 'Unknown'
                    df_filled[col] = df_filled[col].fillna(most_frequent)
            logger.info(f"Filled missing values in categorical columns with most frequent values")
        
        return df_filled
    
    def generate_data_quality_metrics(self, df_original: pd.DataFrame, df_processed: pd.DataFrame) -> Dict[str, Any]:
        """Generate data quality metrics"""
        metrics = {
            'original_shape': list(df_original.shape),
            'processed_shape': list(df_processed.shape),
            'columns_dropped': df_original.shape[1] - df_processed.shape[1],
            'missing_values_original': int(df_original.isnull().sum().sum()),
            'missing_values_processed': int(df_processed.isnull().sum().sum()),
            'missing_values_filled': int(df_original.isnull().sum().sum() - df_processed.isnull().sum().sum()),
            'data_types': df_processed.dtypes.astype(str).to_dict(),
            'numeric_columns': list(df_processed.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(df_processed.select_dtypes(include=['object']).columns),
        }
        
        # Add basic statistics for numeric columns
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            metrics['numeric_statistics'] = df_processed[numeric_cols].describe().to_dict()
        
        return metrics
    
    def generate_validation_report_html(self, validation_results: Dict[str, Any], 
                                       metrics: Dict[str, Any]) -> str:
        """Generate HTML validation report"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Data Validation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
                .section {{ margin: 20px 0; }}
                .pass {{ color: #27ae60; font-weight: bold; }}
                .fail {{ color: #e74c3c; font-weight: bold; }}
                .metrics {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1 class="header">Data Validation Report</h1>
            
            <div class="section">
                <h2>Validation Summary</h2>
                <p>Overall Status: <span class="{'pass' if validation_results['overall_valid'] else 'fail'}">
                    {'PASSED' if validation_results['overall_valid'] else 'FAILED'}</span></p>
                
                <table>
                    <tr><th>Check</th><th>Status</th><th>Details</th></tr>
                    <tr>
                        <td>Minimum Samples</td>
                        <td class="{'pass' if validation_results['min_samples_check'] else 'fail'}">
                            {'PASS' if validation_results['min_samples_check'] else 'FAIL'}</td>
                        <td>Required: {self.data_prep_params['validation_rules']['min_samples']}, 
                            Actual: {validation_results['actual_samples']}</td>
                    </tr>
                    <tr>
                        <td>Missing Value Percentage</td>
                        <td class="{'pass' if validation_results['missing_percentage_check'] else 'fail'}">
                            {'PASS' if validation_results['missing_percentage_check'] else 'FAIL'}</td>
                        <td>Max Allowed: {self.data_prep_params['validation_rules']['max_missing_percentage']}, 
                            Actual: {validation_results['actual_missing_percentage']:.4f}</td>
                    </tr>
                    <tr>
                        <td>Required Columns</td>
                        <td class="{'pass' if validation_results['required_columns_check'] else 'fail'}">
                            {'PASS' if validation_results['required_columns_check'] else 'FAIL'}</td>
                        <td>Missing: {validation_results['missing_required_columns']}</td>
                    </tr>
                </table>
            </div>
            
            <div class="section">
                <h2>Data Processing Summary</h2>
                <div class="metrics">
                    <p><strong>Original Shape:</strong> {metrics['original_shape'][0]} rows × {metrics['original_shape'][1]} columns</p>
                    <p><strong>Processed Shape:</strong> {metrics['processed_shape'][0]} rows × {metrics['processed_shape'][1]} columns</p>
                    <p><strong>Columns Dropped:</strong> {metrics['columns_dropped']}</p>
                    <p><strong>Missing Values Filled:</strong> {metrics['missing_values_filled']}</p>
                    <p><strong>Numeric Columns:</strong> {len(metrics['numeric_columns'])}</p>
                    <p><strong>Categorical Columns:</strong> {len(metrics['categorical_columns'])}</p>
                </div>
            </div>
        </body>
        </html>
        """
        return html_content
    
    def save_outputs(self, df_processed: pd.DataFrame, validation_results: Dict[str, Any], 
                    metrics: Dict[str, Any]):
        """Save all outputs"""
        # Save processed data
        processed_data_path = 'data/processed/telco_churn.csv'
        df_processed.to_csv(processed_data_path, index=False)
        logger.info(f"Processed data saved to {processed_data_path}")
        
        # Save validation report
        html_report = self.generate_validation_report_html(validation_results, metrics)
        report_path = 'data/processed/data_validation_report.html'
        with open(report_path, 'w') as f:
            f.write(html_report)
        logger.info(f"Validation report saved to {report_path}")
        
        # Save metrics
        metrics_path = 'reports/data_quality_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Data quality metrics saved to {metrics_path}")
    
    def run(self, input_file: str = 'data/raw/telco_churn_raw.csv'):
        """Main execution method"""
        logger.info("Starting data preparation process...")
        
        try:
            # Load raw data
            df_original = self.load_raw_data(input_file)
            
            # Validate raw data
            validation_results = self.validate_raw_data(df_original)
            
            if not validation_results['overall_valid']:
                logger.warning("Data validation failed, but continuing with processing...")
            
            # Process data
            df_processed = self.drop_columns(df_original)
            df_processed = self.handle_missing_values(df_processed)
            
            # Generate metrics
            metrics = self.generate_data_quality_metrics(df_original, df_processed)
            
            # Save outputs
            self.save_outputs(df_processed, validation_results, metrics)
            
            logger.info("Data preparation completed successfully!")
            
        except Exception as e:
            logger.error(f"Data preparation failed: {str(e)}")
            raise

def main():
    """Main function"""
    try:
        # Initialize data preparator
        preparator = DataPreparator()
        
        # Run data preparation
        preparator.run()
        
    except Exception as e:
        logger.error(f"Script execution failed: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()