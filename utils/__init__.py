"""
Useful utils
"""
from .misc import AverageMeter

# progress bar
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "progress"))
from progress.bar import Bar as Bar