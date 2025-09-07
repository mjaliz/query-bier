#!/usr/bin/env python3
"""
Simple script to run the false positives to Query conversion.
"""

import os
import sys

# Add src to path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.convert_false_positives import main

if __name__ == "__main__":
    print("Starting false positives to Query conversion...")
    print("Make sure you have:")
    print("1. PostgreSQL running")
    print("2. Created a database (default: query_bier)")
    print("3. Set up .env file with your PostgreSQL credentials (or use defaults)")
    print()

    try:
        main()
        print("\n✅ Conversion completed successfully!")
    except Exception as e:
        print(f"\n❌ Conversion failed: {e}")
        sys.exit(1)
