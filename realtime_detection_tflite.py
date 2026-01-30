import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import os

# Configuration
MODEL_PATH = "bisindo_mlp.tflite"
INPUT_SIZE = 126
# Classes A-Z
CLASSES = [chr(i) for i in range(ord('A'), ord('Z')+1)]

def realtime_detection_tflite():
    # 1. Load TFLite Model
    print(f"Loading TFLite model from {MODEL_PATH}...")
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file '{MODEL_PATH}' not found.")
        return

    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    input_index = input_details[0]['index']
    output_index = output_details[0]['index']

    print("Model loaded successfully.")

    # 2. Setup MediaPipe
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    # 3. Webcam Capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Starting webcam... Press 'q' to quit.")

    def extract_landmarks(hand_landmarks, hand_type):
        """
        Extracts landmarks to match training data format.
        Returns: [hand_code, x1, y1, z1, ..., x21, y21, z21]
        """
        landmarks = []
        for landmark in hand_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])
        
        # Hand code: 1 for Right, -1 for Left
        hand_code = 1 if hand_type == "Right" else -1
        landmarks.insert(0, hand_code)
        
        return landmarks

    def softmax(x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        max_num_hands=2) as hands:
        
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # Mirror the frame
            image = cv2.flip(image, 1)

            # Convert to RGB
            image.flags.writeable = False
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)

            # Prepare for drawing
            image.flags.writeable = True
            
            prediction_text = "Waiting for hands..."
            
            # Initialize empty features (64 zeros each)
            left_hand_data = [0] * 64
            right_hand_data = [0] * 64
            
            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    hand_label = handedness.classification[0].label
                    
                    feats = extract_landmarks(hand_landmarks, hand_label)
                    
                    if hand_label == "Left":
                        left_hand_data = feats
                    else:
                        right_hand_data = feats

                    # Draw landmarks
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
                
                # Combine Left + Right
                combined_data = left_hand_data + right_hand_data
                
                # Slice to 126 features
                features = np.array(combined_data[:INPUT_SIZE], dtype=np.float32)
                
                # Reshape for TFLite [1, 126]
                input_tensor = np.expand_dims(features, axis=0)
                
                # Inference
                interpreter.set_tensor(input_index, input_tensor)
                interpreter.invoke()
                output_data = interpreter.get_tensor(output_index) # Shape [1, NUM_CLASSES]
                
                # Process Output
                # Determine if we need softmax. Usually models output logits.
                # If the model ends with Softmax layer, then output_data is prob.
                # Assuming logits as per Torch training script (CrossEntropyLoss takes logits).
                probs = softmax(output_data[0]) 
                
                pred_idx = np.argmax(probs)
                conf_val = probs[pred_idx] * 100
                pred_char = CLASSES[pred_idx]
                
                prediction_text = f"Pred: {pred_char} ({conf_val:.1f}%)"

            # Display Text
            cv2.putText(image, prediction_text, (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            cv2.imshow('Bisindo Real-time Detection (TFLite)', image)
            
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    realtime_detection_tflite()
