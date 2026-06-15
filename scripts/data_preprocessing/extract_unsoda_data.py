#!/usr/bin/env python3
"""
Script to extract and analyze UNSODA 2.0 database
Extracts SWCC data, soil properties, and prepares for ML training
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Try different methods to read Access database
try:
    import pyodbc
    USE_PYODBC = True
except ImportError:
    USE_PYODBC = False
    print("Warning: pyodbc not available. Trying alternative methods...")

def check_unsoda_structure():
    """Check UNSODA database file structure"""
    data_dir = Path("data_UNSODA 2.0")
    mdb_file = data_dir / "unsoda.mdb"
    
    if not mdb_file.exists():
        print(f"Error: {mdb_file} not found!")
        return False
    
    print(f"✓ Found UNSODA database: {mdb_file}")
    print(f"  File size: {mdb_file.stat().st_size / (1024*1024):.2f} MB")
    
    # Try to read with pyodbc (Windows/Linux with mdbtools)
    if USE_PYODBC:
        try:
            # Connection string for Access database
            # On Linux, may need mdbtools or wine
            conn_str = f'DRIVER={{Microsoft Access Driver (*.mdb, *.accdb)}};DBQ={mdb_file.absolute()};'
            
            try:
                conn = pyodbc.connect(conn_str)
                cursor = conn.cursor()
                
                # Get table names
                tables = [table.table_name for table in cursor.tables(tableType='TABLE')]
                print(f"\n✓ Database connection successful!")
                print(f"  Found {len(tables)} tables:")
                for table in tables:
                    print(f"    - {table}")
                
                # Get sample data from first table
                if tables:
                    sample_query = f"SELECT TOP 5 * FROM [{tables[0]}]"
                    df_sample = pd.read_sql(sample_query, conn)
                    print(f"\n  Sample from '{tables[0]}':")
                    print(f"    Columns: {list(df_sample.columns)}")
                    print(f"    Rows: {len(df_sample)}")
                
                conn.close()
                return True
                
            except pyodbc.Error as e:
                print(f"  Warning: Could not connect with pyodbc: {e}")
                print("  This is normal on Linux. Trying alternative methods...")
        except Exception as e:
            print(f"  Error with pyodbc: {e}")
    
    # Alternative: Check if we can use mdb-export (mdbtools)
    print("\nTrying alternative extraction methods...")
    print("  Note: On Linux, you may need to:")
    print("    1. Install mdbtools: sudo apt-get install mdbtools")
    print("    2. Or convert .mdb to .csv using online tools")
    print("    3. Or use Windows/Mac with Microsoft Access")
    
    return False

def extract_with_mdbtools():
    """Extract data using mdbtools command-line tools"""
    data_dir = Path("data_UNSODA 2.0")
    mdb_file = data_dir / "unsoda.mdb"
    output_dir = Path("data_UNSODA_extracted")
    output_dir.mkdir(exist_ok=True)
    
    # Check if mdb-tables exists
    import subprocess
    try:
        result = subprocess.run(['mdb-tables', str(mdb_file)], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            tables = result.stdout.strip().split()
            print(f"✓ Found {len(tables)} tables using mdbtools:")
            for table in tables:
                print(f"    - {table}")
            
            # Export each table to CSV
            for table in tables:
                csv_file = output_dir / f"{table}.csv"
                try:
                    result = subprocess.run(
                        ['mdb-export', str(mdb_file), table],
                        stdout=open(csv_file, 'w'),
                        text=True,
                        timeout=30
                    )
                    if result.returncode == 0:
                        df = pd.read_csv(csv_file)
                        print(f"  ✓ Exported {table}: {len(df)} rows, {len(df.columns)} columns")
                    else:
                        print(f"  ✗ Failed to export {table}")
                except Exception as e:
                    print(f"  ✗ Error exporting {table}: {e}")
            
            return True
    except FileNotFoundError:
        print("  ✗ mdbtools not installed")
        return False
    except Exception as e:
        print(f"  ✗ Error with mdbtools: {e}")
        return False

def analyze_extracted_data():
    """Analyze extracted UNSODA data"""
    output_dir = Path("data_UNSODA_extracted")
    
    if not output_dir.exists():
        print(f"✗ Extracted data directory not found: {output_dir}")
        return
    
    csv_files = list(output_dir.glob("*.csv"))
    if not csv_files:
        print(f"✗ No CSV files found in {output_dir}")
        return
    
    print(f"\n{'='*60}")
    print("UNSODA Data Analysis")
    print(f"{'='*60}\n")
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            print(f"Table: {csv_file.stem}")
            print(f"  Rows: {len(df):,}")
            print(f"  Columns: {len(df.columns)}")
            print(f"  Columns: {list(df.columns)[:10]}..." if len(df.columns) > 10 else f"  Columns: {list(df.columns)}")
            print(f"  Missing values: {df.isnull().sum().sum()}")
            print()
        except Exception as e:
            print(f"  Error reading {csv_file}: {e}")

def create_data_summary():
    """Create summary of available data"""
    summary = {
        "database_file": "unsoda.mdb",
        "location": "data_UNSODA 2.0/",
        "expected_samples": 790,
        "status": "needs_extraction"
    }
    
    # Save summary
    import json
    with open("unsoda_data_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print("✓ Created data summary: unsoda_data_summary.json")

if __name__ == "__main__":
    print("="*60)
    print("UNSODA 2.0 Database Extraction and Analysis")
    print("="*60)
    print()
    
    # Check database structure
    if check_unsoda_structure():
        print("\n✓ Database structure checked successfully")
    else:
        print("\n⚠ Could not read database directly")
        print("  Attempting alternative extraction methods...")
        
        # Try mdbtools
        if extract_with_mdbtools():
            print("\n✓ Data extracted successfully using mdbtools")
            analyze_extracted_data()
        else:
            print("\n⚠ Automatic extraction not available")
            print("\nManual extraction options:")
            print("  1. Install mdbtools: sudo apt-get install mdbtools")
            print("  2. Use Microsoft Access (Windows/Mac)")
            print("  3. Use online MDB to CSV converters")
            print("  4. Export tables manually from Access")
    
    create_data_summary()
    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)
