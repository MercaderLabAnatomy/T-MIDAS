"""
Configuration settings for T-MIDAS.
"""

import os

# Default paths
TMIDAS_PATH = '/opt/T-MIDAS'
MODELS_PATH = os.path.join(TMIDAS_PATH, 'models')

# Environment settings
DEFAULT_ENV = 'tmidas-env'
TRACKASTRA_ENV = 'trackastra'

# Default parameters
DEFAULT_MIN_SIZE = 250.0
DEFAULT_COMPRESSION = 'zlib'
