import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from collections import deque

def realtime_lstm_tflite():
    # 1. Load TFLite Model
    MODEL_PATH = "bisindo_lstm.tflite"
    print(f"Loading TFLite model from {MODEL_PATH}...")
    try:
        # Load the TFLite model and allocate tensors.
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()

        # Get input and output tensors.
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        print("Model loaded successfully.")
        print("Input Shape:", input_details[0]['shape'])
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Configuration
    INPUT_SIZE = 126
    SEQUENCE_LENGTH = 20
    CLASSES = [chr(i) for i in range(ord('A'), ord('Z')+1)]
    
    # buffer
    sequence_buffer = deque(maxlen=SEQUENCE_LENGTH)

    # 2. Setup MediaPipe
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    cap = cv2.VideoCapture(0)
    print("Starting webcam... Press 'q' to quit.")

    def extract_landmarks(hand_landmarks, hand_type):
        landmarks = []
        for landmark in hand_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])
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

                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
            
            combined_data = left_hand_data + right_hand_data
            current_frame_features = np.array(combined_data[:INPUT_SIZE], dtype=np.float32)
            
            sequence_buffer.append(current_frame_features)
            
            prediction_text = "Gathering frames..."
            
            if len(sequence_buffer) == SEQUENCE_LENGTH:
                # Prepare input: (1, 20, 126)
                input_data = np.expand_dims(np.array(sequence_buffer), axis=0).astype(np.float32)
                
                # Set input tensor
                interpreter.set_tensor(input_details[0]['index'], input_data)

                # Run inference
                interpreter.invoke()

                # Get output
                output_data = interpreter.get_tensor(output_details[0]['index'])
                
                pred_idx = np.argmax(output_data[0])
                conf = output_data[0][pred_idx] * 100
                pred_char = CLASSES[pred_idx]
                
                prediction_text = f"TFLite: {pred_char} ({conf:.1f}%)"

            cv2.putText(image, prediction_text, (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(image, f"Buffer: {len(sequence_buffer)}/{SEQUENCE_LENGTH}", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

            cv2.imshow('Bisindo LSTM (TFLite)', image)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    realtime_lstm_tflite()
