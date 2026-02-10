import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

st.set_page_config(layout="wide", page_title="Face Mask Detection")

@st.cache_resource
def load_face_mask_model():
    return load_model("vgg16_custom_model.keras", compile=False)

model = load_face_mask_model()

@st.cache_resource
def load_face_detector():
    return cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

face_cascade = load_face_detector()

def predict_and_label_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = img[y:y+h, x:x+w]
        face = cv2.resize(face, (224, 224))
        face = face.astype("float32") / 255.0
        face = np.expand_dims(face, axis=0)

        pred = model.predict(face, verbose=0)[0]
        score = pred[0]


        if score <= 0.4:
            label = "With Mask"
            color = (0, 255, 0)
        elif score > 0.5:
            label = "Without Mask"
            color = (0, 0, 255)
        else:
            label = "Unknown"
            color = (255, 255, 0)



        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
        cv2.putText(img, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    return img

st.markdown(
    "<h1 style='text-align:center;'>😷 Face Mask Detection System</h1>",
    unsafe_allow_html=True
)

st.markdown("---")

left, right = st.columns([1, 2])

with left:
    st.markdown("### Select Mode")
    option = st.radio(
        "",
        ("Upload Image", "Live Camera"),
        label_visibility="collapsed"
    )

with right:
    if option == "Upload Image":
        st.markdown("### Upload an Image")
        img_file = st.file_uploader(
            "",
            type=["jpg", "png", "jpeg"],
            label_visibility="collapsed"
        )

        if img_file is not None:
            file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            img = predict_and_label_faces(img)
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), channels="RGB")

    if option == "Live Camera":
        st.markdown("### Live Camera Detection")

        col1, col2 = st.columns(2)
        start = col1.button("▶ Start Camera", use_container_width=True)
        stop = col2.button("⏹ Stop Camera", use_container_width=True)

        frame_box = st.empty()

        if start:
            cap = cv2.VideoCapture(0)

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame = predict_and_label_faces(frame)
                frame_box.image(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                    channels="RGB"
                )

                if stop:
                    break

            cap.release()
