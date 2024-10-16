import streamlit as st
import requests
import numpy as np
import cv2
from mtcnn import MTCNN
from PIL import Image
import io
import tempfile

# FastAPI server URL
API_URL = "http://localhost:8000/recognize"

def detect_faces(image):
    detector = MTCNN()
    image_np = np.array(image)
    if len(image_np.shape) == 2:  # If grayscale
        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
    elif image_np.shape[2] == 4:  # If RGBA
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    faces = detector.detect_faces(image_np)
    return faces, image_np

def annotate_image(image, faces, names):
    for face, name in zip(faces, names):
        x, y, width, height = face['box']
        cv2.rectangle(image, (x, y), (x+width, y+height), (0, 255, 0), 2)
        cv2.putText(image, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return image

def recognize_face(face_image):
    is_success, buffer = cv2.imencode(".jpg", face_image)
    io_buf = io.BytesIO(buffer)
    
    files = {"file": ("image.jpg", io_buf, "image/jpeg")}
    response = requests.post(API_URL, files=files)
    
    if response.status_code == 200:
        return response.json()["celebrity"]
    else:
        return "Unknown"

def process_image(image):
    faces, image_np = detect_faces(image)
    if len(faces) > 0:
        names = []
        for face in faces:
            x, y, width, height = face['box']
            face_image = image_np[y:y+height, x:x+width]
            name = recognize_face(face_image)
            names.append(name)
        
        annotated_image = annotate_image(image_np, faces, names)
        annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        
        return annotated_image_rgb, names
    else:
        return None, []

def main():
    st.set_page_config(layout="wide", page_title="Celebrity Face Recognition")
    
    # Custom CSS for dark theme with outline buttons
    st.markdown("""
    <style>
    .stApp {
        background-color: #000000;
        color: #FFFFFF;
    }
    .stButton>button {
        background-color: transparent;
        color: #FFFFFF;
        border: 2px solid #FFFFFF;
        border-radius: 5px;
        padding: 5px 10px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #FFFFFF;
        color: #000000;
    }
    .stRadio>label, .stSelectbox>label {
        color: #FFFFFF;
    }
    .uploadedFile {
        background-color: #1E1E1E;
        border: 1px solid #333333;
        border-radius: 5px;
    }
    .css-145kmo2 {
        color: #FFFFFF;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("Celebrity Face Recognition")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input")
        input_option = st.selectbox("Select input option:", ("Image Upload", "Webcam", "Video Upload"))

        if input_option == "Image Upload":
            uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
            
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)
                
                if st.button("Detect and Recognize"):
                    annotated_image, names = process_image(image)
                    
                    if annotated_image is not None:
                        with col2:
                            st.subheader("Results")
                            st.image(annotated_image, caption="Detected and Recognized Faces", use_column_width=True)
                            
                            st.write("Recognized celebrities:")
                            for name in set(names):
                                st.write(f"- {name}")
                    else:
                        with col2:
                            st.subheader("Results")
                            st.write("No faces detected in the image.")

        elif input_option == "Webcam":
            st.write("Webcam feed:")
            webcam = st.camera_input("Take a picture")

            if webcam is not None:
                image = Image.open(webcam)
                annotated_image, names = process_image(image)
                
                if annotated_image is not None:
                    with col2:
                        st.subheader("Results")
                        st.image(annotated_image, caption="Detected and Recognized Faces", use_column_width=True)
                        
                        st.write("Recognized celebrities:")
                        for name in set(names):
                            st.write(f"- {name}")
                else:
                    with col2:
                        st.subheader("Results")
                        st.write("No faces detected in the image.")

        elif input_option == "Video Upload":
            uploaded_video = st.file_uploader("Choose a video...", type=["mp4", "mov", "avi"])
            
            if uploaded_video is not None:
                tfile = tempfile.NamedTemporaryFile(delete=False) 
                tfile.write(uploaded_video.read())
                
                video = cv2.VideoCapture(tfile.name)
                
                if st.button("Process Video"):
                    with col2:
                        st.subheader("Results")
                        stframe = st.empty()
                        while video.isOpened():
                            ret, frame = video.read()
                            if not ret:
                                break
                            
                            annotated_frame, names = process_image(frame)
                            
                            if annotated_frame is not None:
                                stframe.image(annotated_frame, channels="RGB", use_column_width=True)
                                
                                st.write("Recognized celebrities:")
                                for name in set(names):
                                    st.write(f"- {name}")
                            else:
                                st.write("No faces detected in this frame.")
                            
                    video.release()

if __name__ == "__main__":
    main()