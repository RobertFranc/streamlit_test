import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import cv2
import numpy as np
import mediapipe as mp


class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        # Example processing: convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return gray

# Callback function to process each video frame
def video_frame_callback(frame):
    # Convert the frame to a NumPy array (BGR format for OpenCV)
    img = frame.to_ndarray(format="bgr24") 
    
    # ... MediaPipe / Drawing logic here ...
    results = mp_pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    if results.pose_landmarks:
        # Draw landmarks for user feedback
        mp_drawing.draw_landmarks(img, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
        
        # Custom CNN Model Logic
        # Extract specific coordinates and pass to CNN model
        # prediction = my_cnn_model.predict(preprocessed_landmarks)
    
    # --- FRAME PROCESSING ---
    # Example: Apply Canny edge detection
    edges = cv2.Canny(img, 100, 200)
    
    cv2.circle(edges, (100, 100), 50, (255, 0, 0), -1)
    
    # Convert grayscale edges back to BGR for display
    processed_img = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    # ------------------------------

    # Return the processed frame back to the browser
    return av.VideoFrame.from_ndarray(processed_img, format="bgr24")

st.title("Real-Time Video Streaming App")
st.write("Click 'Start' to turn on your webcam and see live streaming.")

# Initialize MediaPipe outside the callback to avoid reloading models on every frame
mp_pose = mp.solutions.pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Optional: Configure STUN servers for reliable deployment on Community Cloud
# This helps in establishing peer-to-peer connections across different networks
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


# Streamlit-WebRTC component
webrtc_streamer(
    key="live-stream",
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},   
    async_transform=True,
    #video_frame_callback=video_frame_callback
)



