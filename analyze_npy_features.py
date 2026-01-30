
import numpy as np
import os

file_path = r"c:\Users\Rahman\Desktop\traindataset\dataset_bisindo\train\A\A_b01_s01_20251203_183029.npy"
try:
    data = np.load(file_path)
    print(f"Shape: {data.shape}")
    
    # Assume 126 features (2 hands * 21 * 3) or 63 (1 hand).
    # Let's check the first 126 values stats
    if data.shape[0] >= 126:
        features = data[:126]
        meta = data[126:]
        print(f"First 126 stats: min={features.min()}, max={features.max()}")
        print(f"Remaining {len(meta)} stats: {meta}")
    
    # Check if 63 makes more sense
    features63 = data[:63]
    print(f"First 63 stats: min={features63.min()}, max={features63.max()}")
    
except Exception as e:
    print(f"Error loading file: {e}")
