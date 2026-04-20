#!/usr/bin/env python3
"""
Process extracted UNSODA 2.0 data for ML training
Assumes data has been extracted to CSV files in data_UNSODA_extracted/
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns

class UNSODADataProcessor:
    """Process and prepare UNSODA data for ML training"""
    
    def __init__(self, data_dir="data_UNSODA_extracted"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path("data_processed")
        self.output_dir.mkdir(exist_ok=True)
        
    def check_data_availability(self):
        """Check if extracted data is available"""
        if not self.data_dir.exists():
            print(f"✗ Data directory not found: {self.data_dir}")
            print("  Please extract UNSODA data first (see UNSODA_Data_Extraction_Guide.md)")
            return False
        
        csv_files = list(self.data_dir.glob("*.csv"))
        if not csv_files:
            print(f"✗ No CSV files found in {self.data_dir}")
            print("  Please extract UNSODA data first")
            return False
        
        print(f"✓ Found {len(csv_files)} CSV files:")
        for f in csv_files:
            print(f"    - {f.name}")
        return True
    
    def load_tables(self):
        """Load all UNSODA tables"""
        tables = {}
        csv_files = list(self.data_dir.glob("*.csv"))
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                table_name = csv_file.stem
                tables[table_name] = df
                print(f"✓ Loaded {table_name}: {len(df)} rows, {len(df.columns)} columns")
            except Exception as e:
                print(f"✗ Error loading {csv_file.name}: {e}")
        
        return tables
    
    def identify_key_tables(self, tables):
        """Identify key tables for SWCC prediction"""
        key_tables = {}
        
        # Look for soils/samples table
        for name, df in tables.items():
            name_lower = name.lower()
            if 'soil' in name_lower or 'sample' in name_lower:
                key_tables['soils'] = (name, df)
            elif 'retention' in name_lower or 'swcc' in name_lower or 'water' in name_lower:
                key_tables['retention'] = (name, df)
            elif 'particle' in name_lower or 'gsd' in name_lower or 'grain' in name_lower:
                key_tables['particles'] = (name, df)
            elif 'conductivity' in name_lower:
                key_tables['conductivity'] = (name, df)
        
        print("\n✓ Identified key tables:")
        for key, (name, df) in key_tables.items():
            print(f"    {key}: {name} ({len(df)} rows)")
        
        return key_tables
    
    def merge_data(self, key_tables):
        """Merge data from different tables"""
        # This will need to be customized based on actual table structure
        print("\nMerging data...")
        
        # Placeholder - will need actual column names from UNSODA
        merged_data = {}
        
        return merged_data
    
    def extract_features(self, tables):
        """Extract features for ML training"""
        features = {}
        
        # Extract GSD features
        # Extract physical properties
        # Extract SWCC data
        
        return features
    
    def create_swcc_dataset(self, tables):
        """Create dataset with SWCC curves and soil properties"""
        print("\n" + "="*60)
        print("Creating SWCC Dataset for ML Training")
        print("="*60)
        
        # This will be customized based on actual UNSODA structure
        dataset = {
            'samples': [],
            'features': [],
            'swcc_curves': []
        }
        
        return dataset
    
    def analyze_data_quality(self, df):
        """Analyze data quality"""
        print("\n" + "="*60)
        print("Data Quality Analysis")
        print("="*60)
        
        print(f"\nDataset Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"\nMissing Values:")
        missing = df.isnull().sum()
        print(missing[missing > 0])
        
        print(f"\nData Types:")
        print(df.dtypes)
        
        print(f"\nBasic Statistics:")
        print(df.describe())
        
    def save_processed_data(self, data, filename="unsoda_processed.csv"):
        """Save processed data"""
        output_file = self.output_dir / filename
        if isinstance(data, pd.DataFrame):
            data.to_csv(output_file, index=False)
        else:
            # Handle other data types
            pass
        print(f"\n✓ Saved processed data: {output_file}")
    
    def generate_report(self, tables):
        """Generate data summary report"""
        report = {
            'total_tables': len(tables),
            'table_names': list(tables.keys()),
            'total_samples': 0,
            'data_quality': {}
        }
        
        for name, df in tables.items():
            report['data_quality'][name] = {
                'rows': len(df),
                'columns': len(df.columns),
                'missing_values': df.isnull().sum().to_dict()
            }
        
        # Save report
        report_file = self.output_dir / "data_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n✓ Generated report: {report_file}")
        return report

def main():
    """Main processing function"""
    print("="*60)
    print("UNSODA 2.0 Data Processing")
    print("="*60)
    print()
    
    processor = UNSODADataProcessor()
    
    # Check if data is available
    if not processor.check_data_availability():
        return
    
    # Load tables
    print("\nLoading tables...")
    tables = processor.load_tables()
    
    if not tables:
        print("✗ No tables loaded. Please check data extraction.")
        return
    
    # Identify key tables
    key_tables = processor.identify_key_tables(tables)
    
    # Analyze each table
    print("\n" + "="*60)
    print("Table Analysis")
    print("="*60)
    
    for name, df in tables.items():
        print(f"\nTable: {name}")
        processor.analyze_data_quality(df)
    
    # Generate report
    report = processor.generate_report(tables)
    
    print("\n" + "="*60)
    print("Processing Complete!")
    print("="*60)
    print(f"\nNext steps:")
    print("1. Review the data report: data_processed/data_report.json")
    print("2. Customize feature extraction based on actual table structure")
    print("3. Create SWCC dataset for ML training")
    print("4. Proceed with data preprocessing (Phase 1 of research plan)")

if __name__ == "__main__":
    main()
