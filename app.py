import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import cv2
import numpy as np

st.title("Real-Time Video Streaming App")

# Optional: Configure STUN servers for reliable deployment on Community Cloud
# This helps in establishing peer-to-peer connections across different networks
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        # Example processing: convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return gray

webrtc_streamer(
    key="example",
    video_transformer_factory=VideoTransformer,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    async_transform=True
)
