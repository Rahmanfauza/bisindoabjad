import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import re
import glob

# Configuration
DATASET_PATH = r"c:\Users\Rahman\Desktop\traindataset\dataset_bisindo"
INPUT_SIZE = 126
MAX_TIMESTEPS = 20  # Max frames per sequence
NUM_CLASSES = 26
BATCH_SIZE = 32
EPOCHS = 100
MODEL_SAVE_PATH = "bisindo_lstm.h5"
TFLITE_SAVE_PATH = "bisindo_lstm.tflite"

def parse_filename(filepath):
    """
    Parses filename to extract info for grouping.
    Format example: A_b01_s01_20251203_183029.npy
    """
    filename = os.path.basename(filepath)
    # Regex to match: Class_Batch_Step_Date_Time.npy
    # Note: Class can be one letter. Batch is bXX. Step is sXX.
    match = re.match(r"([A-Z])_b(\d+)_s(\d+)_(\d{8})_(\d{6})\.npy", filename)
    if match:
        return {
            "class": match.group(1),
            "batch": match.group(2),
            "step": int(match.group(3)),
            "timestamp": match.group(4) + "_" + match.group(5),
            "path": filepath
        }
    return None

def load_data(root_dir, split='train'):
    split_dir = os.path.join(root_dir, split)
    classes = sorted([d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))])
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    
    sequences = {} # Key: "Class_Batch_Timestamp", Value: list of records
    
    print(f"Scanning {split} data...")
    for cls_name in classes:
        cls_dir = os.path.join(split_dir, cls_name)
        files = glob.glob(os.path.join(cls_dir, "*.npy"))
        
        for f in files:
            info = parse_filename(f)
            if info:
                key = f"{info['class']}_{info['batch']}_{info['timestamp']}"
                if key not in sequences:
                    sequences[key] = []
                sequences[key].append(info)
    
    X = []
    y = []
    
    print(f"Found {len(sequences)} sequences. Processing...")
    
    for key, frames in sequences.items():
        # Sort by step
        frames.sort(key=lambda x: x['step'])
        
        # Load data
        seq_data = []
        label = class_to_idx[frames[0]['class']]
        
        for frame in frames:
            try:
                data = np.load(frame['path']).astype(np.float32)
                # Ensure feature size
                if data.shape[0] >= INPUT_SIZE:
                    seq_data.append(data[:INPUT_SIZE])
                else:
                    # Pad if too short (rare)
                    seq_data.append(np.pad(data, (0, INPUT_SIZE - data.shape[0])))
            except Exception as e:
                print(f"Error reading {frame['path']}: {e}")
        
        if not seq_data:
            continue
            
        # Pad or Truncate sequence
        seq_len = len(seq_data)
        if seq_len > MAX_TIMESTEPS:
            seq_data = seq_data[:MAX_TIMESTEPS]
        else:
            # Pad with zeros (or masking value)
            # Create a full array of shape (MAX_TIMESTEPS, INPUT_SIZE)
            padded = np.zeros((MAX_TIMESTEPS, INPUT_SIZE), dtype=np.float32)
            padded[:seq_len, :] = np.array(seq_data)
            seq_data = padded
            
        X.append(seq_data)
        y.append(label)
        
    return np.array(X), np.array(y), classes

def build_model(num_classes):
    model = keras.Sequential([
        layers.Masking(mask_value=0.0, input_shape=(MAX_TIMESTEPS, INPUT_SIZE)),
        layers.LSTM(64, return_sequences=False),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def main():
    # 1. Load Data
    X_train, y_train, classes = load_data(DATASET_PATH, 'train')
    X_val, y_val, _ = load_data(DATASET_PATH, 'val')
    
    print(f"Training Info: X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"Val Info: X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
    
    if len(classes) != NUM_CLASSES:
        print(f"Warning: Found {len(classes)} classes, but expected {NUM_CLASSES}")
    
    # 2. Build Model
    model = build_model(len(classes))
    model.summary()
    
    # 3. Train
    callbacks = [
        keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True)
    ]
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks
    )
    
    # 4. Save and Convert to TFLite
    print("Saving model...")
    model.save(MODEL_SAVE_PATH)
    
    print("Converting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Enable Select TF ops (Required for LSTM/RNN often)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS, 
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    
    # Disable experimental lower tensor list ops if needed (sometimes helps with LSTM)
    converter._experimental_lower_tensor_list_ops = False
    
    tflite_model = converter.convert()
    
    with open(TFLITE_SAVE_PATH, 'wb') as f:
        f.write(tflite_model)
        
    print(f"TFLite model saved to {TFLITE_SAVE_PATH}")

if __name__ == "__main__":
    main()
