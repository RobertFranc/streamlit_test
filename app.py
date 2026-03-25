import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import cv2
import numpy as np
import mediapipe as mp
import av


class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        # Example processing: convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return gray

def resize_img(image, window_name:str='', d_height:int=480, d_width:int=480, verbose:bool=False):
    """
    ### Proportionally resize an image to desired size and show it.
    """
    if not window_name:
        window_name = 'image'
    h, w = image.shape[:2]
    if h < w:
        img = cv2.resize(src=image, dsize=(d_width, math.floor(h / (w/d_width) ) ) )
    elif h < w:
        img = cv2.resize(src=image, dsize=(math.floor(w / (h / d_height) ), d_height) )
    elif h == w:
        img = cv2.resize(src=image,
                         dsize=(d_height, d_width)
                        )
    
    if verbose:
        print(d_height, d_width)
        print(f"Original image:\t{window_name}\t{image.shape}")
        print(f"Reshaped image:\t{window_name}\t{img.shape}")

    return img


# Callback function to process each video frame
def video_frame_callback(frame):
    # Take only the 5th frame:
    st.session_state.framecount += 1
    #if st.session_state.framecount % 5 != 0: return frame
    # Convert the frame to a NumPy array (BGR format for OpenCV)
    img = frame.to_ndarray(format="bgr24") 
    img.flags.writeable = False
    # Convert color format from BGR (cv2) to RGB (mediapipe)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # TODO: 1⃣ Flip image if using front camera or laptop integrated webcam
    # ✅🙆‍♂️
    img = cv2.flip(src=img, flipCode=1)
    ##################################################
    results_face = face_mesh.process(img)

    # If mediapipe detects any faces:
    if results_face.multi_face_landmarks:
        # NOTE: For Cropping Detected Face 👇:
        # Take the bounding coordinates of the detected face to use for cropping the image:
        mesh_points = np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results_face.multi_face_landmarks[0].landmark])
        xmin = min(mesh_points[:, 0])
        ymin = min(mesh_points[:, 1])
        xmax = max(mesh_points[:, 0])
        ymax = max(mesh_points[:, 1])
        point1 = xmin, ymin # Upper-Left ↖️ point of cropped image
        point2 = xmax, ymax # Lower-Right ↘️ point of cropped image
        cropped_image = img[ymin:ymax, xmin:xmax]
        # NOTE: For Cropping Detected Face 👆 :

        # NOTE: Printing label to the picture 👇:
        cv2.rectangle(img=img, pt1=point1, pt2=point2, color=(0,0,255), thickness=2, lineType=cv2.LINE_AA)
        # NOTE: Printing label to the picture 👆:

        # 👇
        # Resize the frame to for easy viewing
        #resized_frame = resize_img(window_name='img', image=cropped_image, d_height=480, d_width=480, verbose=True)
        # 👆           

        pil_image = Image.fromarray(cropped_image) # convert image frame to PIL Image object
        # 👇Create a temporary PNG file:
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_imgfile:
            temp_imgfile_path = temp_imgfile.name
            print('temp_imgfile_path:', temp_imgfile_path)
            pil_image.save(temp_imgfile_path)
        # 👆Create a temporary PNG file:


        # 👇Load cropped face image and resize for prediction
        image = load_img( # ❓whatsdis: return value❓
            path=temp_imgfile_path,
            color_mode="grayscale",
            target_size=(48,48), # 💡 resize to match the preset model
            interpolation="nearest",
            keep_aspect_ratio=False,)
        # 👆Load cropped face image and resize for prediction

        # Remove temp image file after using to free up some memory
        os.remove(temp_imgfile_path)

        # 🏭 preprocess for prediction
        input_arr = img_to_array(image)
        input_arr = np.array([input_arr])  # Convert single image to a batch. --> resulting shape: (1, 48, 48, 1) image data
        # Or try as is

        # 🔮 Prediction 
        preds = my_model.predict(input_arr)
        maxpred_index = preds.argmax()
        maxpredrate = max(preds)
        # if verbose: print('prediction:', preds, class_labels[prediction])
        predicted_emotion = class_labels[maxpred_index]
        if predicted_emotion: # Show frame label if emption is found
            predict_count += 1
            print(f'preds: {preds[0]} 💡 maxpred_index={maxpred_index} 💡 max={maxpredrate}') 
            print(f'frame #: {frame_count} ⭐ predict #: {predict_count} ⭐⭐⭐⭐⭐ emotion: {predicted_emotion}⭐⭐⭐⭐⭐')

##################################################
# ... MediaPipe / Drawing logic here ...
    pose_results = mp_pose.process(img)
    
    if pose_results.pose_landmarks:
        # Draw landmarks for user feedback
        mp_drawing.draw_landmarks(img, pose_results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
        
        # Custom CNN Model Logic
        # Extract specific coordinates and pass to CNN model
        # prediction = my_cnn_model.predict(preprocessed_landmarks)
    
    results_hands = mp_hands.process(img)
    if results_hands.hand_landmarks:
        # Draw landmarks for user feedback
        #mp_drawing.draw_landmarks(img, results_hands.hand_landmarks, mp.solutions.hand.HANDS_CONNECTIONS)
        mp_drawing.draw_landmarks(img, results_hands.left_hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(img, results_hands.right_hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
    # # --- FRAME PROCESSING ---
    # # Example: Apply Canny edge detection
    # edges = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), 100, 200)
    
    # cv2.circle(edges, (100, 100), 50, (255, 0, 0), -1)

    # Convert color format back from RGB (mediapipe) to BGR (cv2)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # # Convert grayscale edges back to BGR for display
    # processed_img = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    # # ------------------------------

    # Return the processed frame back to the browser
    #return av.VideoFrame.from_ndarray(processed_img, format="bgr24")
    return av.VideoFrame.from_ndarray(img, format="bgr24")

st.title("Real-Time Video Streaming App")
st.write("Click 'Start' to turn on your webcam and see live streaming.")
st.session_state['framecount'] = st.session_state.get('framecount', 0)
# Initialize MediaPipe outside the callback to avoid reloading models on every frame:
# For mediapipe processing one frame:
face_mesh = mp.solutions.face_mesh.FaceMesh(
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
            )

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
mp_hands = mp.solutions.hands.Hands(
    static_image_mode=False, # Whether to treat the input images as a batch of static and possibly unrelated images, or a video stream.
    max_num_hands=2,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


mp_drawing = mp.solutions.drawing_utils


# Optional: Configure STUN servers for reliable deployment on Community Cloud
# This helps in establishing peer-to-peer connections across different networks
# RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        # If possible, replace the below with own Twilio/Metered TURN credentials
        # {"urls": ["turn:relay.metered.ca:80"], "username": "openrelayproject", "credential": "openrelayproject"}
    ]}
)

# Streamlit-WebRTC component
webrtc_streamer(
    key="live-stream",
    rtc_configuration=RTC_CONFIGURATION,
    #media_stream_constraints={"video": True, "audio": False},   
    media_stream_constraints={"video": {"width": 640, "height": 480}, "audio": False},
    #async_transform=True,
    async_processing=True,
    video_frame_callback=video_frame_callback
)



# TODO: 1⃣ Flip image if using front camera or laptop integrated webcam
# TODO: 2⃣ change to the current API: from mediapipe.tasks.vision import PoseLandmarkerOptions, PoseLandmarker
