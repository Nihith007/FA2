# ==========================================================================
# FA-2 : Elderly Fall Detection — Streamlit Dashboard
# Step 7 of the assignment
# Deploy this file (with fall_detection_model.h5 and class_names.txt in the
# same GitHub repo) to Streamlit Cloud: https://streamlit.io
# ==========================================================================

import os
import urllib.request
import streamlit as st
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import tensorflow as tf
from PIL import Image
import pandas as pd

st.set_page_config(page_title="Elderly Fall Detection Dashboard", layout="wide")

# ------------------------------------------------------------------
# Load model and class names (must be uploaded to the same repo)
# ------------------------------------------------------------------
@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path="fall_detection_model.tflite")
    interpreter.allocate_tensors()
    with open("class_names.txt") as f:
        class_names = [line.strip() for line in f.readlines()]
    return interpreter, class_names


interpreter, CLASS_NAMES = load_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
IMG_SIZE = (128, 128)

# Standard 33-point BlazePose skeleton connections (stable across versions)
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32),
]

POSE_MODEL_PATH = "pose_landmarker.task"
POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)


@st.cache_resource
def load_pose_detector():
    if not os.path.exists(POSE_MODEL_PATH):
        urllib.request.urlretrieve(POSE_MODEL_URL, POSE_MODEL_PATH)
    base_options = mp_python.BaseOptions(model_asset_path=POSE_MODEL_PATH)
    options = vision.PoseLandmarkerOptions(base_options=base_options)
    return vision.PoseLandmarker.create_from_options(options)


pose_detector = load_pose_detector()

# ------------------------------------------------------------------
# Session-state counters for the analytics dashboard
# ------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []  # list of predicted class names


def predict_activity(pil_image: Image.Image):
    """Run pose estimation + CNN classification on a single image."""
    image_bgr = cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)
    annotated = image_bgr.copy()

    # Pose estimation (for visualization)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    pose_result = pose_detector.detect(mp_image)

    if pose_result.pose_landmarks:
        h, w, _ = annotated.shape
        landmarks = pose_result.pose_landmarks[0]  # first detected person
        points = []
        for lm in landmarks:
            x, y = int(lm.x * w), int(lm.y * h)
            points.append((x, y))
            cv2.circle(annotated, (x, y), 4, (0, 255, 0), -1)
        for start_idx, end_idx in POSE_CONNECTIONS:
            cv2.line(annotated, points[start_idx], points[end_idx], (255, 0, 0), 2)

    # CNN activity classification (TFLite inference)
    resized = cv2.resize(image_bgr, IMG_SIZE)
    normalized = resized.astype("float32") / 255.0
    batch = np.expand_dims(normalized, axis=0)

    interpreter.set_tensor(input_details[0]["index"], batch)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]["index"])[0]
    pred_idx = int(np.argmax(preds))
    pred_class = CLASS_NAMES[pred_idx]
    confidence = float(preds[pred_idx])

    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    return pred_class, confidence, annotated_rgb


# ------------------------------------------------------------------
# Sidebar — upload controls
# ------------------------------------------------------------------
st.sidebar.title("Upload")
uploaded_file = st.sidebar.file_uploader(
    "Upload an image or video frame", type=["jpg", "jpeg", "png"]
)
uploaded_video = st.sidebar.file_uploader(
    "Or upload a short video", type=["mp4", "mov", "avi"]
)

st.title("🏥 AI-Powered Elderly Fall Detection Dashboard")
st.caption("Pose estimation + deep learning activity classification with real-time alerts")

col1, col2 = st.columns([2, 1])

# ------------------------------------------------------------------
# Handle single image upload
# ------------------------------------------------------------------
if uploaded_file is not None:
    pil_image = Image.open(uploaded_file)
    pred_class, confidence, annotated_rgb = predict_activity(pil_image)
    st.session_state.history.append(pred_class)

    with col1:
        st.image(annotated_rgb, caption="Pose estimation output", use_container_width=True)

    with col2:
        st.metric("Predicted Activity", pred_class)
        st.metric("Confidence", f"{confidence * 100:.1f}%")

        if pred_class == "Fall":
            st.error("🚨 FALL DETECTED — Emergency alert triggered! Notify caregiver immediately.")
        else:
            st.success(f"✅ Normal monitoring — activity: {pred_class}")

# ------------------------------------------------------------------
# Handle video upload — process frame by frame
# ------------------------------------------------------------------
if uploaded_video is not None:
    tmp_path = "temp_video.mp4"
    with open(tmp_path, "wb") as f:
        f.write(uploaded_video.read())

    cap = cv2.VideoCapture(tmp_path)
    frame_placeholder = col1.empty()
    alert_placeholder = col2.empty()
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % 10 != 0:  # sample every 10th frame for speed
            continue

        pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        pred_class, confidence, annotated_rgb = predict_activity(pil_frame)
        st.session_state.history.append(pred_class)

        frame_placeholder.image(annotated_rgb, caption=f"Frame {frame_count}", use_container_width=True)
        if pred_class == "Fall":
            alert_placeholder.error(f"🚨 FALL DETECTED at frame {frame_count}!")
        else:
            alert_placeholder.info(f"Activity: {pred_class} ({confidence*100:.1f}%)")

    cap.release()

# ------------------------------------------------------------------
# Monitoring analytics
# ------------------------------------------------------------------
st.divider()
st.subheader("📊 Monitoring Analytics")

if st.session_state.history:
    counts = pd.Series(st.session_state.history).value_counts()
    total = len(st.session_state.history)
    fall_count = counts.get("Fall", 0)
    normal_count = total - fall_count

    m1, m2, m3 = st.columns(3)
    m1.metric("Total Activities Detected", total)
    m2.metric("Fall Detected Count", fall_count)
    m3.metric("Normal Activity Count", normal_count)

    st.bar_chart(counts)
else:
    st.info("Upload an image or video to start monitoring.")

if st.sidebar.button("Reset dashboard"):
    st.session_state.history = []
    st.rerun()
