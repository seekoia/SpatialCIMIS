#!/usr/bin/env python3
"""
Interactive runner for Spatial CIMIS processing
Can be run in VS Code terminal or Jupyter notebook
"""

import subprocess
import os
import sys
from datetime import datetime

def print_header(title):
    """Print a formatted header"""
    print("\n" + "="*60)
    print(title)
    print("="*60 + "\n")

def run_command(cmd, description):
    """Run a command and stream output"""
    print(f"▶ {description}")
    print(f"Command: {' '.join(cmd)}\n")
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Stream output in real-time
        for line in process.stdout:
            print(line, end='')
        
        process.wait()
        
        if process.returncode == 0:
            print(f"\n✓ {description} completed successfully!\n")
            return True
        else:
            print(f"\n✗ {description} failed with return code {process.returncode}\n")
            return False
            
    except Exception as e:
        print(f"\n✗ Error running {description}: {e}\n")
        return False

def test_grid_standardization():
    """Run grid standardization tests"""
    print_header("Grid Standardization Tests")
    return run_command(
        ["python", "test_grid_standardization.py"],
        "Testing grid standardization"
    )

def process_variable(variable, config="scp_config.txt", options=None):
    """Process a variable"""
    print_header(f"Processing Variable: {variable}")
    
    cmd = ["python", "spatial_cimis_processing.py", "--config", config, "--variable", variable]
    
    if options:
        cmd.extend(options)
    else:
        cmd.append("--process-all")
    
    return run_command(cmd, f"Processing {variable}")

def quick_test_year(variable, year):
    """Quick test on a single year"""
    print_header(f"Quick Test: {variable} for {year}")
    
    cmd = [
        "python", "spatial_cimis_processing.py",
        "--variable", variable,
        "--start-year", str(year),
        "--end-year", str(year),
        "--process-mask",
        "--process-data"
    ]
    
    return run_command(cmd, f"Testing {variable} for year {year}")

def main_menu():
    """Display main menu"""
    print_header("Spatial CIMIS Processing - Interactive Mode")
    
    print("Available operations:")
    print("  1. Run grid standardization tests")
    print("  2. Quick test: Process single year (Rs 2010)")
    print("  3. Quick test: Process single year (custom)")
    print("  4. Create mask only")
    print("  5. Process full pipeline (all steps)")
    print("  6. Custom processing")
    print("  0. Exit")
    print()
    
    choice = input("Select option (0-6): ").strip()
    return choice

def run_interactive():
    """Run interactive menu"""
    
    while True:
        choice = main_menu()
        
        if choice == "0":
            print("\nExiting...")
            break
            
        elif choice == "1":
            test_grid_standardization()
            
        elif choice == "2":
            # Test with Rs 2010 (has both 500m and 2km resolution)
            print("\nTesting with Rs 2010 (includes resolution changes)...")
            quick_test_year("Rs", 2010)
            
        elif choice == "3":
            variable = input("Variable (e.g., Tx, Rs, ETo, Tn): ").strip()
            year = input("Year: ").strip()
            if variable and year:
                quick_test_year(variable, int(year))
            
        elif choice == "4":
            variable = input("Variable (e.g., Tx, Rs, ETo, Tn): ").strip()
            if variable:
                process_variable(variable, options=["--process-mask"])
                
        elif choice == "5":
            variable = input("Variable (e.g., Tx, Rs, ETo, Tn): ").strip()
            if variable:
                confirm = input(f"Process full pipeline for {variable}? This may take a while. (y/n): ")
                if confirm.lower() == 'y':
                    process_variable(variable)
                    
        elif choice == "6":
            print("\nCustom processing options:")
            variable = input("Variable: ").strip()
            start_year = input("Start year (default from config): ").strip()
            end_year = input("End year (default from config): ").strip()
            
            print("\nSelect operations (y/n):")
            mask = input("  Create mask? ").strip().lower() == 'y'
            data = input("  Process data? ").strip().lower() == 'y'
            two_d = input("  Create 2D files? ").strip().lower() == 'y'
            daily_clim = input("  Daily climatology? ").strip().lower() == 'y'
            monthly_clim = input("  Monthly climatology? ").strip().lower() == 'y'
            
            options = []
            if start_year:
                options.extend(["--start-year", start_year])
            if end_year:
                options.extend(["--end-year", end_year])
            if mask:
                options.append("--process-mask")
            if data:
                options.append("--process-data")
            if two_d:
                options.append("--process-2d")
            if daily_clim:
                options.append("--process-daily-clim")
            if monthly_clim:
                options.append("--process-monthly-clim")
                
            if variable and options:
                process_variable(variable, options=options)
        
        else:
            print("\nInvalid option. Please try again.")
        
        input("\nPress Enter to continue...")

def direct_run():
    """Run with command line arguments"""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Interactive mode: python run_interactive.py")
        print("  Direct run:      python run_interactive.py test")
        print("                   python run_interactive.py process <variable>")
        print("                   python run_interactive.py test-year <variable> <year>")
        return
    
    command = sys.argv[1]
    
    if command == "test":
        test_grid_standardization()
    elif command == "process" and len(sys.argv) > 2:
        variable = sys.argv[2]
        process_variable(variable)
    elif command == "test-year" and len(sys.argv) > 3:
        variable = sys.argv[2]
        year = int(sys.argv[3])
        quick_test_year(variable, year)
    else:
        print(f"Unknown command: {command}")

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    if len(sys.argv) == 1:
        # Interactive mode
        try:
            run_interactive()
        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Exiting...")
    else:
        # Direct command mode
        direct_run()

