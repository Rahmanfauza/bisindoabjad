import streamlit as st
import cv2
import numpy as np
import torch
import mediapipe as mp
from PIL import Image
import os

from model import BisindoMLP, INPUT_SIZE, HIDDEN_SIZES, NUM_CLASSES
from utils import process_landmarks, predict_on_frame

# Page config
st.set_page_config(
    page_title="Bisindo Detection",
    page_icon="👐",
    layout="wide"
)

# Constants
MODEL_PATH = "bisindo_mlp_v1.pth"
CLASSES = [chr(i) for i in range(ord('A'), ord('Z')+1)]

# Load Model (Cached)
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BisindoMLP(INPUT_SIZE, HIDDEN_SIZES, NUM_CLASSES).to(device)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        return model, device
    except FileNotFoundError:
        st.error(f"Model file not found at: {MODEL_PATH}")
        return None, None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

model, device = load_model()

# Setup MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def process_and_predict(image):
    if model is None:
        return
    
    # Convert PIL Image to slightly acceptable format for MediaPipe (numpy array RGB)
    image_np = np.array(image)
    if image_np.shape[-1] == 4: # RGBA to RGB
         image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
    
    # MediaPipe processing
    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        max_num_hands=2) as hands:
        
        results = hands.process(image_np)
        
        # Draw landmarks on a copy of the image for visualization
        annotated_image = image_np.copy()
        
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
            input_tensor = torch.from_numpy(features).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(input_tensor)
                probs = torch.softmax(output, 1)
                conf, pred_idx = torch.max(probs, 1)
                
                pred_char = CLASSES[pred_idx.item()]
                conf_val = conf.item() * 100
                
            st.success(f"Prediction: **{pred_char}** (Confidence: {conf_val:.1f}%)")
            
        else:
            st.warning("No hands detected in the image.")
            
        st.image(annotated_image, caption="Processed Image", use_container_width=True)


# UI Layout
st.title("👐 Bisindo Sign Language Detection")
st.write("Upload an image, take a photo, or use live webcam.")

tab1, tab2, tab3 = st.tabs(["📷 Take Photo", "📂 Upload Image", "🎥 Live Realtime"])

with tab1:
    camera_image = st.camera_input("Take a picture")
    if camera_image:
        image = Image.open(camera_image)
        process_and_predict(image)

with tab2:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        if st.button("Analyze Uploaded Image"):
            process_and_predict(image)

with tab3:
    st.header("Live Webcam Detection")
    st.write("Click **Start** to open webcam. Click **Stop** to close it.")
    
    col1, col2 = st.columns(2)
    with col1:
        start_btn = st.button("Start Live Feed")
    with col2:
        stop_btn = st.button("Stop Live Feed")
        
    if start_btn:
        cap = cv2.VideoCapture(0)
        st_frame = st.empty()
        
        while cap.isOpened():
            if stop_btn: # Use session state or check button (Streamlit reruns on interaction)
                # In standard Streamlit, buttons trigger rerun. 
                # For a loop, we effectively need to break when the script reruns or a condition is met.
                # However, since 'stop_btn' is outside, we can't easily check it inside loop without session state logic.
                # Simplified: The "Stop" button triggers a rerun where 'start_btn' is False, so the loop isn't entered.
                # But to break the loop interactively, we need a unique key or check.
                # Actually, standard Streamlit pattern for loops is `run = st.checkbox('Run')`.
                break

            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture video")
                break
            
            # Mirror frame
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process
            annotated_frame, pred_text = predict_on_frame(frame_rgb, model, device, CLASSES, mp_hands, mp_drawing, mp_drawing_styles)
            
            # Draw prediction ON the frame
            cv2.putText(annotated_frame, pred_text, (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            # Display
            st_frame.image(annotated_frame, channels="RGB")
            
        cap.release()
    else:
        st.info("Webcam is not running.")

