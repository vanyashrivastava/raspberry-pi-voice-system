#!/usr/bin/env python3
"""
Supabase Setup Helper Script

This script helps with the initial setup and validation of Supabase integration.
Run this after creating your Supabase project and setting environment variables.

Usage:
    python supabase_setup_helper.py
"""

import os
import sys
from pathlib import Path


def print_header(text):
    """Print a section header."""
    print(f"\n{'='*70}")
    print(f"  {text}")
    print(f"{'='*70}\n")


def print_step(step_num, text):
    """Print a step number and description."""
    print(f"[Step {step_num}] {text}")


def check_env_file():
    """Check if .env file exists and is configured."""
    print_step(1, "Checking .env configuration")
    
    env_file = Path('.env')
    
    if not env_file.exists():
        print("  ✗ .env file not found")
        print("  → Run: cp .env.example .env")
        print("  → Then edit .env with your Supabase credentials")
        return False
    
    # Check for required variables
    with open(env_file) as f:
        content = f.read()
    
    required = ['SUPABASE_URL', 'SUPABASE_KEY', 'SUPABASE_SERVICE_ROLE_KEY']
    missing = []
    
    for var in required:
        if var not in content:
            missing.append(var)
        elif content.find(f'{var}=your_') >= 0:
            missing.append(f"{var} (not configured)")
    
    if missing:
        print(f"  ✗ Missing or unconfigured variables:")
        for var in missing:
            print(f"    - {var}")
        return False
    
    print("  ✓ .env file configured correctly")
    return True


def check_dependencies():
    """Check if required Python packages are installed."""
    print_step(2, "Checking dependencies")
    
    required_packages = {
        'requests': 'requests',
        'dotenv': 'python-dotenv',
        'supabase': 'supabase',
    }
    
    missing = []
    for import_name, package_name in required_packages.items():
        try:
            __import__(import_name)
            print(f"  ✓ {package_name}")
        except ImportError:
            missing.append(package_name)
            print(f"  ✗ {package_name} not installed")
    
    if missing:
        print(f"\n  Install missing packages:")
        print(f"  → pip install {' '.join(missing)}")
        return False
    
    return True


def test_connection():
    """Test connection to Supabase."""
    print_step(3, "Testing Supabase connection")
    
    try:
        from src.supabase.db_manager import SupabaseDB
        
        db = SupabaseDB()
        if db.test_connection():
            print("  ✓ Successfully connected to Supabase")
            return True
        else:
            print("  ✗ Connection test failed")
            print("  → Check SUPABASE_URL and SUPABASE_KEY in .env")
            return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        print("  → Ensure all dependencies are installed")
        return False


def check_database_tables():
    """Check if database tables exist."""
    print_step(4, "Checking database tables")
    
    try:
        from src.supabase.db_manager import SupabaseDB
        
        db = SupabaseDB()
        
        # Try to fetch from each table (limit 1)
        tables = ['audio_transcripts', 'alerts', 'email_events', 'system_logs', 'system_metrics']
        all_exist = True
        
        for table in tables:
            try:
                response = db.session.get(
                    f"{db.supabase_url}/rest/v1/{table}?limit=1",
                    headers=db._get_headers(),
                    timeout=db.timeout
                )
                if response.status_code == 200:
                    print(f"  ✓ {table}")
                else:
                    print(f"  ✗ {table} (status: {response.status_code})")
                    all_exist = False
            except Exception as e:
                print(f"  ✗ {table} (error: {str(e)[:50]}...)")
                all_exist = False
        
        if not all_exist:
            print("\n  Tables missing? Run the migration in Supabase:")
            print("  1. Go to Supabase Dashboard → SQL Editor")
            print("  2. Create new query")
            print("  3. Copy content of supabase/migrations/001_init_schema.sql")
            print("  4. Paste and click Run")
        
        return all_exist
    except Exception as e:
        print(f"  ✗ Error checking tables: {e}")
        return False


def test_insert():
    """Test inserting sample data."""
    print_step(5, "Testing data insertion")
    
    try:
        from src.supabase.db_manager import SupabaseDB
        
        db = SupabaseDB()
        
        # Try inserting test data
        result = db.insert_alert({
            'alert_type': 'test',
            'scam_probability': 50.0,
            'source': 'setup_helper',
            'description': 'Test alert from setup helper'
        })
        
        if result.get('success'):
            print(f"  ✓ Successfully inserted test alert")
            print(f"    ID: {result.get('id')}")
            return True
        else:
            print(f"  ✗ Insert failed: {result.get('error')}")
            return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def print_next_steps():
    """Print next steps for the user."""
    print_header("Next Steps")
    
    print("✅ Setup verification complete!\n")
    
    print("To integrate Supabase into your system:\n")
    
    print("1. Update main.py:")
    print("   from src.supabase.supabase_alert_logger import SupabaseAlertLogger")
    print("   alert_logger = SupabaseAlertLogger(use_supabase=True)\n")
    
    print("2. Integrate into your components:")
    print("   - Audio stream handler → insert_transcript()")
    print("   - Email parser → insert_email_event()")
    print("   - Alert system → log_alert()\n")
    
    print("3. View data in Supabase Dashboard:")
    print("   - Go to Data Editor")
    print("   - Select tables to view data\n")
    
    print("For detailed instructions, see:")
    print("   - SUPABASE_SETUP.md (complete guide)")
    print("   - SUPABASE_QUICK_REFERENCE.md (quick commands)")
    print("   - src/supabase/integration_examples.py (code samples)\n")


def print_summary(results):
    """Print summary of all checks."""
    print_header("Setup Summary")
    
    checks = [
        ("Environment Configuration", results[0]),
        ("Dependencies", results[1]),
        ("Supabase Connection", results[2]),
        ("Database Tables", results[3]),
        ("Data Insertion", results[4]),
    ]
    
    all_passed = all(results)
    
    for check_name, passed in checks:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{check_name:<30} {status}")
    
    print(f"\n{'Overall Status':<30} {'✓ READY' if all_passed else '✗ SETUP NEEDED'}")
    
    if all_passed:
        print("\n🎉 Your Supabase integration is ready to use!")
    else:
        print("\n⚠️  Please address the failing checks above before proceeding.")
    
    return all_passed


def main():
    """Run all setup checks."""
    print_header("Supabase Setup Helper")
    
    print("This script will verify your Supabase integration setup.\n")
    
    results = []
    
    # Run checks
    results.append(check_env_file())
    results.append(check_dependencies())
    results.append(test_connection())
    results.append(check_database_tables())
    results.append(test_insert())
    
    # Print summary and next steps
    all_passed = print_summary(results)
    
    if all_passed:
        print_next_steps()
        return 0
    else:
        print("\n❌ Setup verification failed. Fix the issues above and run again.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
