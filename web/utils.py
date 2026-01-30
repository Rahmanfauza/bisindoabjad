import numpy as np

def extract_landmarks(hand_landmarks, hand_type):
    """
    Extracts landmarks from a single hand in the format expected by the model.
    Matches logic in `realtime_detection.py` and `inisiasi cllass.py`.
    
    Returns: [hand_code, x1, y1, z1, ..., x21, y21, z21]
    """
    landmarks = []
    # Raw coordinates (0-1), NO re-normalization like (x-0.5)*2
    for landmark in hand_landmarks.landmark:
        landmarks.extend([landmark.x, landmark.y, landmark.z])
    
    # Add hand type as prefix (1 for right, -1 for left)
    # inisiasi cllass.py logic: 'Right' -> 1, else -> -1
    hand_code = 1 if hand_type == "Right" else -1
    landmarks.insert(0, hand_code)
    
    return landmarks

def process_landmarks(results, input_size=126):
    """
    Processes MediaPipe results into a feature vector for the model.
    Combines Left and Right hands, pads/truncates to input_size.
    """
    # Initialize empty slots (64 zeros each), matching inisiasi cllass.py
    # 64 features = 1 hand_code + 21*3 coords
    left_hand_data = [0] * 64
    right_hand_data = [0] * 64
    
    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            # Get hand label (Right/Left)
            hand_label = handedness.classification[0].label
            
            # Extract features
            feats = extract_landmarks(hand_landmarks, hand_label)
            
            # Slot into correct array
            if hand_label == "Left":
                left_hand_data = feats
            else:
                right_hand_data = feats
                
    # Combine exactly as data collector: Left + Right
    combined_data = left_hand_data + right_hand_data # Length 128
    
    # Slice to INPUT_SIZE (126) to match Training logic
    features = np.array(combined_data[:input_size], dtype=np.float32)
    
    return features

def predict_on_frame(image_np, model, device, classes, mp_hands, mp_drawing, mp_drawing_styles):
    """
    Processes a single frame for real-time detection.
    """
    prediction_text = "Waiting for hands..."
    annotated_image = image_np.copy()
    
    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        max_num_hands=2) as hands:
        
        results = hands.process(image_np)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    annotated_image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
            
            # Prepare features
            features = process_landmarks(results)
            
            # Inference
            import torch
            input_tensor = torch.from_numpy(features).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(input_tensor)
                probs = torch.softmax(output, 1)
                conf, pred_idx = torch.max(probs, 1)
                
                pred_char = classes[pred_idx.item()]
                conf_val = conf.item() * 100
                
                prediction_text = f"Pred: {pred_char} ({conf_val:.1f}%)"
    
    return annotated_image, prediction_text

