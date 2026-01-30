import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from collections import deque

def realtime_lstm_h5():
    # 1. Load Model
    MODEL_PATH = "bisindo_lstm.h5"
    print(f"Loading Keras model from {MODEL_PATH}...")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Configuration
    INPUT_SIZE = 126
    SEQUENCE_LENGTH = 20
    CLASSES = [chr(i) for i in range(ord('A'), ord('Z')+1)]
    
    # buffer to store sequence of frames
    sequence_buffer = deque(maxlen=SEQUENCE_LENGTH)

    # 2. Setup MediaPipe
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    cap = cv2.VideoCapture(0)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Starting webcam... Press 'q' to quit.")

    def extract_landmarks(hand_landmarks, hand_type):
        landmarks = []
        for landmark in hand_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])
        # Hand code: Right=1, Left=-1
        hand_code = 1 if hand_type == "Right" else -1
        landmarks.insert(0, hand_code)
        return landmarks

    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        max_num_hands=2) as hands:
        
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                continue

            # Flip and Convert
            image = cv2.flip(image, 1)
            image.flags.writeable = False
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)
            
            image.flags.writeable = True

            # Prepare current frame features
            left_hand_data = [0] * 64
            right_hand_data = [0] * 64
            
            has_hands = False

            if results.multi_hand_landmarks and results.multi_handedness:
                has_hands = True
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    hand_label = handedness.classification[0].label
                    feats = extract_landmarks(hand_landmarks, hand_label)
                    
                    if hand_label == "Left":
                        left_hand_data = feats
                    else:
                        right_hand_data = feats

                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
            
            # Combine
            combined_data = left_hand_data + right_hand_data
            current_frame_features = np.array(combined_data[:INPUT_SIZE], dtype=np.float32)
            
            # Add to buffer
            # If no hands detected, we still add the zero-vector frame to maintain time flow
            # (Or you could choose to only add frames with hands. 
            #  Given training might have some empty frames or not, usually continous stream is best for LSTM)
            sequence_buffer.append(current_frame_features)
            
            prediction_text = "Gathering frames..."
            
            if len(sequence_buffer) == SEQUENCE_LENGTH:
                # Prepare input: (1, 20, 126)
                input_sequence = np.expand_dims(np.array(sequence_buffer), axis=0)
                
                # Predict
                preds = model.predict(input_sequence, verbose=0)
                pred_idx = np.argmax(preds)
                conf = preds[0][pred_idx] * 100
                pred_char = CLASSES[pred_idx]
                
                prediction_text = f"Pred: {pred_char} ({conf:.1f}%)"

            # Display
            cv2.putText(image, prediction_text, (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            # Draw buffer status
            cv2.putText(image, f"Buffer: {len(sequence_buffer)}/{SEQUENCE_LENGTH}", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

            cv2.imshow('Bisindo LSTM (H5)', image)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    realtime_lstm_h5()
