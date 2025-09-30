"""
Compare adaptive vs fixed encoding strategies.
THIS IS YOUR KEY RESULT for the paper!
"""

import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.train_tiny import train_quick
import matplotlib.pyplot as plt

def compare_encoding_strategies():
    """
    Train models with:
    1. Adaptive encoding (your approach)
    2. Fixed rate encoding
    3. Fixed temporal encoding
    4. Fixed best manual selection
    """
    
    results = {}
    
    print("="*60)
    print("ENCODING STRATEGY COMPARISON")
    print("="*60 + "\n")
    
    # 1. Adaptive (already trained)
    print("1. Loading adaptive model results...")
    # You already have this!
    
    # 2. Fixed rate encoding
    print("\n2. Training with FIXED rate encoding...")
    # Modify model to use only rate encoding
    # (Implementation below)
    
    # 3. Fixed temporal encoding
    print("\n3. Training with FIXED temporal encoding...")
    
    # 4. Plot comparison
    print("\n4. Creating comparison plots...")
    
    return results

if __name__ == '__main__':
    compare_encoding_strategies()