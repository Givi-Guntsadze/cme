#!/usr/bin/env python
"""Debug script to identify home route crash."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import MagicMock
from app.main import home

# Create a mock request
request = MagicMock()
request.query_params = {}

try:
    result = home(request)
    print("Home route returned successfully!")
    print(f"Status: {result.status_code}")
except Exception as e:
    print(f"EXCEPTION: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
