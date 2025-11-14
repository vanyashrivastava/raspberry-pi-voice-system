#!/usr/bin/env python3
"""
Quick script to query and display Supabase data.
Run this locally to test connection and see your data.

Usage:
    python test_supabase_query.py
"""

import os
import json
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    print("\n" + "="*70)
    print("🔌 SUPABASE DATA QUERY TEST")
    print("="*70 + "\n")
    
    # Get credentials
    supabase_url = os.getenv('SUPABASE_URL')
    supabase_key = os.getenv('SUPABASE_KEY')
    service_role_key = os.getenv('SUPABASE_SERVICE_ROLE_KEY')
    
    if not supabase_url or not supabase_key:
        print("❌ ERROR: Missing Supabase credentials")
        print("   Set SUPABASE_URL and SUPABASE_KEY in .env file")
        return False
    
    print(f"📍 Supabase URL: {supabase_url}")
    print(f"🔑 Auth Key: {supabase_key[:20]}...{supabase_key[-10:]}\n")
    
    # Setup headers - use service role key for full access
    headers = {
        'Authorization': f'Bearer {service_role_key}',
        'apikey': service_role_key,
        'Content-Type': 'application/json',
    }
    
    # Test connection
    print("Testing connection...")
    try:
        response = requests.get(
            f"{supabase_url}/rest/v1/",
            headers=headers,
            timeout=10
        )
        print(f"✅ Connection successful (status: {response.status_code})\n")
    except Exception as e:
        print(f"❌ Connection failed: {e}\n")
        return False
    
    # List of tables to query
    tables = [
        'call_transcripts',
    ]
    
    print("-"*70)
    print("📊 FETCHING DATA FROM TABLES")
    print("-"*70 + "\n")
    
    found_data = False
    
    for table in tables:
        print(f"📋 Querying table: '{table}'")
        
        try:
            # Fetch latest row
            url = f"{supabase_url}/rest/v1/{table}?limit=1&order=created_at.desc"
            response = requests.get(
                url,
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                
                if data:
                    found_data = True
                    print(f"   ✅ Success! Found {len(data)} row(s)\n")
                    print(f"   📌 DATA:")
                    print("   " + "─"*66)
                    
                    # Pretty print the row
                    formatted = json.dumps(data[0], indent=6, default=str)
                    for line in formatted.split('\n'):
                        print(f"   {line}")
                    
                    print("   " + "─"*66 + "\n")
                else:
                    print(f"   ⚠️  Table exists but is empty\n")
            
            elif response.status_code == 404:
                print(f"   ⚠️  Table not found (404)\n")
            
            else:
                print(f"   ❌ Error (status {response.status_code}): {response.text[:100]}\n")
        
        except Exception as e:
            print(f"   ❌ Error: {e}\n")
    
    print("-"*70)
    if found_data:
        print("✅ SUCCESS! Supabase is connected and working!")
        print("   Your data is being stored in the cloud! 🎉")
    else:
        print("⚠️  No data found in tables")
        print("   Make sure you've added at least one row of test data")
    print("-"*70 + "\n")
    
    return found_data

if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
