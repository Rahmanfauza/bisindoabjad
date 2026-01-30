import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from train_mlp_torch import BisindoMLP, INPUT_SIZE, HIDDEN_SIZES, NUM_CLASSES, MODEL_SAVE_PATH

def predict_new_data():
    # Load Model
    model = BisindoMLP(INPUT_SIZE, HIDDEN_SIZES, NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.eval()
    
    # New Data Directory
    new_data_dir = r"c:\Users\Rahman\Desktop\traindataset\prediction new data"
    files = [f for f in os.listdir(new_data_dir) if f.endswith('.npy')]
    files.sort()
    
    # Classes A-Z
    classes = [chr(i) for i in range(ord('A'), ord('Z')+1)]
    
    # Plotting
    cols = 3
    rows = (len(files) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 4))
    axes = axes.flatten()
    
    print(f"Predicting on {len(files)} files from {new_data_dir}...")
    
    for i, fname in enumerate(files):
        path = os.path.join(new_data_dir, fname)
        ax = axes[i]
        
        # Load Data
        try:
            data = np.load(path).astype(np.float32)
            features = data[:INPUT_SIZE]
            
            # Simple validation/padding
            if features.shape[0] < INPUT_SIZE:
                features = np.pad(features, (0, INPUT_SIZE - features.shape[0]))
            else:
                features = features[:INPUT_SIZE]
                
            input_tensor = torch.from_numpy(features)
        except Exception as e:
            print(f"Error loading {fname}: {e}")
            continue
            
        # Predict
        with torch.no_grad():
            output = model(input_tensor.unsqueeze(0))
            probs = torch.softmax(output, 1)
            conf, pred_idx = torch.max(probs, 1)
            pred_class = classes[pred_idx.item()]
            confidence = conf.item() * 100
            
        # Extract True Label from filename (Assumed format: "A_...")
        true_label = fname.split('_')[0] if '_' in fname else "?"
        
        is_correct = (pred_class == true_label)
        color = 'black' if is_correct else 'red'
        
        # Visualize Landmarks
        landmarks = input_tensor.numpy()
        landmarks_3d = landmarks.reshape(-1, 3)
        
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
        ax.scatter(x1, -y1, s=20, c='blue')
        for start, end in connections:
            ax.plot([x1[start], x1[end]], [-y1[start], -y1[end]], 'b-', linewidth=1)
            
        # Plot Hand 2
        if landmarks_3d.shape[0] > 21:
            h2 = landmarks_3d[21:]
            if not np.allclose(h2, 0):
                x2, y2 = h2[:, 0], h2[:, 1]
                ax.scatter(x2, -y2, s=20, c='magenta')
                for start, end in connections:
                    ax.plot([x2[start], x2[end]], [-y2[start], -y2[end]], 'm-', linewidth=1)
                    
        ax.set_title(f"File: {fname}\nTrue: {true_label} | Pred: {pred_class}\nConf: {confidence:.2f}%", 
                     color=color, fontweight='bold', fontsize=10)
        ax.axis('off')
        ax.set_aspect('equal')
        
        print(f"File: {fname} -> Pred: {pred_class} ({confidence:.2f}%) [{ 'OK' if is_correct else 'FAIL' }]")

    # Turn off unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    save_path = "new_data_predictions.png"
    plt.savefig(save_path)
    print(f"Visualization saved to {save_path}")

if __name__ == "__main__":
    predict_new_data()
