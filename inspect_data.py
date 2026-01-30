
import numpy as np
import os

file_path = r"c:\Users\Rahman\Desktop\traindataset\dataset_bisindo\train\A\A_b01_s01_20251203_183029.npy"
try:
    data = np.load(file_path)
    print(f"Shape: {data.shape}")
    print(f"Data: {data}")
except Exception as e:
    print(f"Error loading file: {e}")
