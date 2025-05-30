#!/usr/bin/env python3
"""Script to run comprehensive tests for the ECG analysis system."""

import subprocess
import sys
import os
from pathlib import Path
import argparse


def run_command(command: str, description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {command}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with exit code {e.returncode}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False


def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(description="Run ECG analysis system tests")
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    parser.add_argument("--integration", action="store_true", help="Run integration tests only")
    parser.add_argument("--coverage", action="store_true", help="Generate coverage report")
    parser.add_argument("--lint", action="store_true", help="Run linting checks")
    parser.add_argument("--type-check", action="store_true", help="Run type checking")
    parser.add_argument("--all", action="store_true", help="Run all tests and checks")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Change to project root directory
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    print(f"Running tests from: {project_root}")
    
    success_count = 0
    total_count = 0
    
    # Determine what to run
    run_unit = args.unit or args.all or not any([args.unit, args.integration, args.coverage, args.lint, args.type_check])
    run_integration = args.integration or args.all
    run_coverage = args.coverage or args.all
    run_lint = args.lint or args.all
    run_type_check = args.type_check or args.all
    
    verbose_flag = "-v" if args.verbose else ""
    
    # Install dependencies first
    total_count += 1
    if run_command("pip install -r requirements.txt", "Installing dependencies"):
        success_count += 1
    
    # Run unit tests
    if run_unit:
        total_count += 1
        if run_command(f"python -m pytest tests/unit/ {verbose_flag}", "Unit tests"):
            success_count += 1
    
    # Run integration tests
    if run_integration:
        total_count += 1
        if run_command(f"python -m pytest tests/integration/ {verbose_flag}", "Integration tests"):
            success_count += 1
    
    # Run tests with coverage
    if run_coverage:
        total_count += 1
        if run_command(f"python -m pytest tests/ --cov=src --cov-report=html --cov-report=term {verbose_flag}", 
                      "Tests with coverage"):
            success_count += 1
            print("\n📊 Coverage report generated in htmlcov/index.html")
    
    # Run linting
    if run_lint:
        total_count += 1
        if run_command("python -m flake8 src/ tests/ --max-line-length=88 --extend-ignore=E203,W503", 
                      "Code linting"):
            success_count += 1
        
        total_count += 1
        if run_command("python -m black --check src/ tests/", "Code formatting check"):
            success_count += 1
    
    # Run type checking
    if run_type_check:
        total_count += 1
        if run_command("python -m mypy src/ --ignore-missing-imports", "Type checking"):
            success_count += 1
    
    # Run data generation tests
    total_count += 1
    if run_command(f"python -m pytest tests/data/ {verbose_flag}", "Data generation tests"):
        success_count += 1
    
    # Summary
    print(f"\n{'='*60}")
    print(f"TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Passed: {success_count}/{total_count}")
    
    if success_count == total_count:
        print("🎉 All tests passed!")
        return 0
    else:
        print(f"❌ {total_count - success_count} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())