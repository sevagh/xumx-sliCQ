# Pytest __init__.py file

import os
import sys
import pytest

# Standard pytest main
if __name__ == "__main__":
    sys.exit(pytest.main([os.path.dirname(__file__)]))