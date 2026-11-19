"""Request limits (doppel_api/limits.py) are off for the suite; test_limits.py turns them on where it tests them."""

import os

os.environ["DOPPEL_LIMITS"] = "off"
