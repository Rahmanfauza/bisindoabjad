import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score

# Configuration
DATASET_PATH = r"c:\Users\Rahman\Desktop\traindataset\dataset_bisindo"
INPUT_SIZE = 126  # 21 landmarks * 3 coords * 2 hands (flattened)
HIDDEN_SIZES = [512, 256, 128]
NUM_CLASSES = 26
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 150
MODEL_SAVE_PATH = "bisindo_mlp_v1.pth"

class BisindoDataset(Dataset):
    def __init__(self, root_dir, split='train'):
        self.root_dir = os.path.join(root_dir, split)
        self.classes = sorted([d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.samples = []
        
        for cls_name in self.classes:
            cls_dir = os.path.join(self.root_dir, cls_name)
            for fname in os.listdir(cls_dir):
                if fname.endswith('.npy'):
                    self.samples.append((os.path.join(cls_dir, fname), self.class_to_idx[cls_name]))
        
        print(f"Loaded {len(self.samples)} samples for split '{split}'. Classes: {len(self.classes)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            # Load data and take first 126 features
            data = np.load(path).astype(np.float32)
            features = data[:INPUT_SIZE]
            
            # Simple validation to ensure shape
            if features.shape[0] != INPUT_SIZE:
                # Pad or truncate if necessary (though current analysis says they are 131)
                if features.shape[0] < INPUT_SIZE:
                    features = np.pad(features, (0, INPUT_SIZE - features.shape[0]))
                else:
                    features = features[:INPUT_SIZE]
                    
            return torch.from_numpy(features), torch.tensor(label, dtype=torch.long)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return torch.zeros(INPUT_SIZE), torch.tensor(label, dtype=torch.long)

class BisindoMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes):
        super(BisindoMLP, self).__init__()
        layers = []
        in_dim = input_size
        for h_dim in hidden_sizes:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Datasets
    train_dataset = BisindoDataset(DATASET_PATH, split='train')
    val_dataset = BisindoDataset(DATASET_PATH, split='val')
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = BisindoMLP(INPUT_SIZE, HIDDEN_SIZES, NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_acc = 0.0

    history = {'loss': [], 'val_acc': []}

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(train_loader)
        
        # Validation
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_acc = accuracy_score(all_labels, all_preds)
        
        history['loss'].append(epoch_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print("Model saved.")

    print(f"Training complete. Best Val Acc: {best_acc:.4f}")
    
    # Save history
    with open("training_history.json", "w") as f:
        json.dump(history, f)
    print("Training history saved to training_history.json")

if __name__ == "__main__":
    train_model()
