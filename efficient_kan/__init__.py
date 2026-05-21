# Make this directory a package
import os
import sys

# Get the absolute path of the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Import the KAN class
from .kan import KAN

# Make KAN available when importing the package
__all__ = ["KAN"]