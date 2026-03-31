import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import cv2
import mediapipe as mp
import av



# Callback function to process each video frame
def video_frame_callback(frame):
    # Convert the frame to a NumPy array (BGR format for OpenCV)
    img = frame.to_ndarray(format="bgr24") 
    img.flags.writeable = False
    # Convert color format from BGR (cv2) to RGB (mediapipe)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # TODO: 1⃣ Flip image if using front camera or laptop integrated webcam
    # ✅🙆‍♂️
    img = cv2.flip(src=img, flipCode=1)

##################################################

##################################################
    # ... MediaPipe / Drawing logic here ...
    results = mp_pose.process(img)

    if results.pose_landmarks:
        # Draw landmarks for user feedback
        mp_drawing.draw_landmarks(img, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
        
        
        # Custom CNN Model Logic
        # Extract specific coordinates and pass to CNN model
        # prediction = my_cnn_model.predict(preprocessed_landmarks)

    # # --- FRAME PROCESSING ---
    # Convert color format back from RGB (mediapipe) to BGR (cv2)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # ------------------------------

    # Return the processed frame back to the browser
    #return av.VideoFrame.from_ndarray(processed_img, format="bgr24")
    return av.VideoFrame.from_ndarray(img, format="bgr24")

st.title("Real-Time Video Streaming App")
st.write("Click 'Start' to turn on your webcam and see live streaming.")

# Initialize MediaPipe outside the callback to avoid reloading models on every frame:
##################################################

# Legacy API: Force the "Lite" model which is more stable on Cloud:
# TODO: 2⃣ change to the current API: from mediapipe.tasks.vision import PoseLandmarkerOptions, PoseLandmarker
mp_pose = mp.solutions.pose.Pose( 
    static_image_mode=False,
    model_complexity=1, # 0 = Lite and is faster for Cloud, 1 = Full, 2 = Heavy
    smooth_landmarks=True,
    enable_segmentation=False,
    smooth_segmentation=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)



# Initialize ONCE at the global level (outside the class)
mp_drawing = mp.solutions.drawing_utils


# Optional: Configure STUN servers for reliable deployment on Community Cloud
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
    ]}
)

# Streamlit-WebRTC component
webrtc_streamer(
    key="live-stream",
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},   
    #async_transform=True,
    async_processing=True,
    video_frame_callback=video_frame_callback
)
