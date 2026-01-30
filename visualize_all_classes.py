import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from train_mlp_torch import BisindoMLP, INPUT_SIZE, HIDDEN_SIZES, NUM_CLASSES, MODEL_SAVE_PATH, BisindoDataset
import math

def visualize_all_classes():
    # Load Model
    model = BisindoMLP(INPUT_SIZE, HIDDEN_SIZES, NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.eval()
    
    # Load one sample per class from validation set
    dataset_path = r"c:\Users\Rahman\Desktop\traindataset\dataset_bisindo"
    val_dataset = BisindoDataset(dataset_path, split='val')
    
    # Organize samples by class
    class_samples = {}
    
    # Iterate through dataset to find one sample for each class
    # Since dataset is large, we can just iterate until we find all or finish
    found_classes = set()
    
    # We need to map index back to class name
    idx_to_class = {v: k for k, v in val_dataset.class_to_idx.items()}
    
    for i in range(len(val_dataset)):
        inp, label_idx = val_dataset[i]
        label = label_idx.item()
        
        if label not in found_classes:
            class_samples[label] = inp
            found_classes.add(label)
            
        if len(found_classes) == NUM_CLASSES:
            break
            
    # Sort by class index
    sorted_labels = sorted(class_samples.keys())
    
    # Plotting
    cols = 6
    rows = math.ceil(len(sorted_labels) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(20, rows * 3.5))
    axes = axes.flatten()
    
    correct_count = 0
    
    for i, label_idx in enumerate(sorted_labels):
        ax = axes[i]
        input_tensor = class_samples[label_idx]
        class_name = idx_to_class[label_idx]
        
        # Predict
        with torch.no_grad():
            output = model(input_tensor.unsqueeze(0))
            _, pred_idx = torch.max(output, 1)
            pred_class = idx_to_class[pred_idx.item()]
            
        is_correct = (pred_class == class_name)
        if is_correct:
            correct_count += 1
            color = 'black'
        else:
            color = 'red'
            
        # Reshape for plotting (Assuming 21 points * 3 coords * 2 hands = 126)
        # We only care about X, Y. Z is ignored for 2D plot.
        # Data is flattened: x1, y1, z1, x2, y2, z2...
        landmarks = input_tensor.numpy()
        
        # Reshape to (42, 3)
        landmarks_3d = landmarks.reshape(-1, 3)
        
        # Split into Hand 1 (0-20) and Hand 2 (21-41)
        # Note: If second hand is empty/zeros, it will just plot at 0,0 or hidden
        
        # Connections for a hand
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),      # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),      # Index
            (0, 9), (9, 10), (10, 11), (11, 12), # Middle
            (0, 13), (13, 14), (14, 15), (15, 16), # Ring
            (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
            (5, 9), (9, 13), (13, 17) # Palm
        ]
        
        # Plot Hand 1
        h1 = landmarks_3d[:21]
        x1, y1 = h1[:, 0], h1[:, 1]
        # Invert Y because image coords are top-down, typically normalized data might be different
        # Usually mediapipe is normalized 0-1. Standard plot y is bottom-up. 
        # So we might want to invert Y to look like an image.
        ax.scatter(x1, -y1, s=10, c='blue')
        
        for start, end in connections:
            ax.plot([x1[start], x1[end]], [-y1[start], -y1[end]], 'b-', linewidth=1)
            
        # Plot Hand 2 (if exists / values are not all zero)
        if landmarks_3d.shape[0] > 21:
            h2 = landmarks_3d[21:]
            # Check if it has valid data (simple check: not all zeros)
            if not np.allclose(h2, 0):
                x2, y2 = h2[:, 0], h2[:, 1]
                ax.scatter(x2, -y2, s=10, c='magenta')
                for start, end in connections:
                    ax.plot([x2[start], x2[end]], [-y2[start], -y2[end]], 'm-', linewidth=1)
        
        ax.set_title(f"True: {class_name}\nPred: {pred_class}", color=color, fontweight='bold')
        ax.axis('off')
        ax.set_aspect('equal')

    # Turn off unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
        
    plt.tight_layout()
    plt.suptitle(f"Sample Predictions per Class (Val Set)", fontsize=16)
    plt.subplots_adjust(top=0.95)
    
    save_path = "all_classes_predictions.png"
    plt.savefig(save_path)
    print(f"Visualization saved to {save_path}")

if __name__ == "__main__":
    visualize_all_classes()
