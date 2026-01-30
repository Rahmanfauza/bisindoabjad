import torch.nn as nn

# Configuration matching train_mlp_torch.py
INPUT_SIZE = 126  # 21 landmarks * 3 coords * 2 hands (flattened)
HIDDEN_SIZES = [512, 256, 128]
NUM_CLASSES = 26

class BisindoMLP(nn.Module):
    def __init__(self, input_size=INPUT_SIZE, hidden_sizes=HIDDEN_SIZES, num_classes=NUM_CLASSES):
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
