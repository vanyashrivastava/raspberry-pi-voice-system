#!/usr/bin/env python3
"""
Interactive setup script for Supabase credentials.
This helps you add your Supabase credentials to .env safely.

Usage:
    python setup_supabase_credentials.py
"""

import os
from pathlib import Path

def main():
    print("\n" + "="*70)
    print("🔐 SUPABASE CREDENTIALS SETUP")
    print("="*70)
    print("\nThis will help you add your Supabase credentials to .env\n")
    
    print("To get your credentials:")
    print("1. Go to your Supabase Dashboard")
    print("2. Click Settings → API")
    print("3. Copy the values below\n")
    
    print("-"*70)
    
    # Get credentials from user
    supabase_url = input("Enter SUPABASE_URL (https://xxx.supabase.co): ").strip()
    supabase_key = input("Enter SUPABASE_KEY (anon key): ").strip()
    service_role_key = input("Enter SUPABASE_SERVICE_ROLE_KEY (service role key): ").strip()
    
    print("-"*70 + "\n")
    
    # Validate
    if not all([supabase_url, supabase_key, service_role_key]):
        print("❌ All fields are required")
        return False
    
    # Read current .env
    env_file = Path('.env')
    if env_file.exists():
        with open(env_file) as f:
            content = f.read()
    else:
        print("⚠️  .env file not found, creating from template...")
        content = ""
    
    # Update with new credentials
    lines = content.split('\n')
    updated_lines = []
    
    for line in lines:
        if line.startswith('SUPABASE_URL='):
            updated_lines.append(f'SUPABASE_URL={supabase_url}')
        elif line.startswith('SUPABASE_KEY='):
            updated_lines.append(f'SUPABASE_KEY={supabase_key}')
        elif line.startswith('SUPABASE_SERVICE_ROLE_KEY='):
            updated_lines.append(f'SUPABASE_SERVICE_ROLE_KEY={service_role_key}')
        else:
            updated_lines.append(line)
    
    # Write back
    with open(env_file, 'w') as f:
        f.write('\n'.join(updated_lines))
    
    print("✅ Credentials saved to .env")
    print("\n⚠️  IMPORTANT: Never commit .env to GitHub!")
    print("   It's automatically in .gitignore\n")
    
    print("Now test the connection:")
    print("  python test_supabase_query.py\n")
    
    return True

if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
