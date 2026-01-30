#!/usr/bin/env python3
"""
Test runner for T-MIDAS.
"""

import sys
import os
sys.path.insert(0, '/opt/T-MIDAS')

if __name__ == "__main__":
    import pytest
    # Run tests
    os.chdir('/opt/T-MIDAS')
    sys.exit(pytest.main(["-v", "tmidas/tests/"]))
