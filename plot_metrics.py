import json
import matplotlib.pyplot as plt

# Load history
with open("training_history.json", "r") as f:
    history = json.load(f)

loss = history['loss']
val_acc = history['val_acc']
epochs = range(1, len(loss) + 1)

# Plot
plt.figure(figsize=(12, 5))

# Loss Plot
plt.subplot(1, 2, 1)
plt.plot(epochs, loss, 'b-', label='Training Loss')
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Accuracy Plot
plt.subplot(1, 2, 2)
plt.plot(epochs, val_acc, 'g-', label='Validation Accuracy')
plt.title('Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_plot.png')
print("Plot saved to training_plot.png")
