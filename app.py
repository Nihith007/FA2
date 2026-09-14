"""
Elderly Fall / Activity Detection — Streamlit App (FA-2)
=========================================================
Loads the CNN trained in the companion Colab notebook
(fall_detection_model.h5 + class_names.txt) and lets a caregiver
upload an image, take a photo, or upload a video to classify activity
into one of 3 classes: Falling, Lying, Normal.

Covers FA-2 Step 7 requirements:
- Upload images
- Upload videos
- Run AI predictions
- Display fall alerts (emergency notification)
- Show monitoring analytics (totals, fall count, normal count,
  confidence score, activity distribution chart)
- Pose visualization overlay (MediaPipe)

Run with:
    streamlit run app.py

Expected files in the same folder as this script:
    fall_detection_model.h5
    class_names.txt
    pose_landmarker.task   (auto-downloaded on first run if missing)
"""

import os
import time
import urllib.request
from collections import Counter

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
IMG_SIZE = (128, 128)
MODEL_PATH = "fall_detection_model.h5"
CLASS_NAMES_PATH = "class_names.txt"
POSE_MODEL_PATH = "pose_landmarker.task"
POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)
FALL_LABEL = "Falling"        # must match the class name used in class_names.txt
LYING_LABEL = "Lying"         # post-fall / person-down state
VIDEO_SAMPLE_EVERY_N_FRAMES = 15  # classify roughly ~2 frames/sec at 30fps video

# Standard 33-point BlazePose skeleton connections (stable across versions)
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32),
]

st.set_page_config(page_title="Elderly Fall Detection", page_icon="🚨", layout="centered")


# ----------------------------------------------------------------------
# Cached resource loaders
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading classification model...")
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(
            f"Model file '{MODEL_PATH}' not found. Place it next to app.py "
            "(it's produced by the training notebook)."
        )
        st.stop()
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_resource(show_spinner=False)
def load_class_names():
    if not os.path.exists(CLASS_NAMES_PATH):
        st.error(f"'{CLASS_NAMES_PATH}' not found. Place it next to app.py.")
        st.stop()
    with open(CLASS_NAMES_PATH, "r") as f:
        return [line.strip() for line in f if line.strip()]


@st.cache_resource(show_spinner="Loading pose model...")
def load_pose_detector():
    if not os.path.exists(POSE_MODEL_PATH):
        try:
            urllib.request.urlretrieve(POSE_MODEL_URL, POSE_MODEL_PATH)
        except Exception as e:
            st.warning(f"Could not download pose model automatically: {e}")
            return None
    base_options = mp_python.BaseOptions(model_asset_path=POSE_MODEL_PATH)
    options = vision.PoseLandmarkerOptions(base_options=base_options)
    return vision.PoseLandmarker.create_from_options(options)


# ----------------------------------------------------------------------
# Session state — monitoring analytics accumulate across the session
# ----------------------------------------------------------------------
def init_session_state():
    if "history" not in st.session_state:
        # each entry: {"label": str, "confidence": float, "source": str, "timestamp": float}
        st.session_state.history = []


def log_prediction(label: str, confidence: float, source: str):
    st.session_state.history.append(
        {"label": label, "confidence": confidence, "source": source, "timestamp": time.time()}
    )


def reset_session():
    st.session_state.history = []


# ----------------------------------------------------------------------
# Core logic
# ----------------------------------------------------------------------
def classify_array(model, class_names, rgb_array: np.ndarray):
    """Resize/normalize an RGB numpy image and run the CNN. Returns (label, probs dict)."""
    img = Image.fromarray(rgb_array).convert("RGB").resize(IMG_SIZE)
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)  # batch dim

    probs = model.predict(arr, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    label = class_names[pred_idx]
    prob_dict = {name: float(p) for name, p in zip(class_names, probs)}
    return label, prob_dict


def draw_pose_on_array(detector, bgr_image: np.ndarray) -> tuple[np.ndarray, bool]:
    """Runs MediaPipe pose detection on a BGR numpy image and draws the skeleton.
    Returns (annotated_bgr_image, person_detected)."""
    if detector is None:
        return bgr_image, False

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB))
    result = detector.detect(mp_image)

    annotated = bgr_image.copy()
    if not result.pose_landmarks:
        return annotated, False

    h, w, _ = annotated.shape
    landmarks = result.pose_landmarks[0]  # first detected person

    points = []
    for lm in landmarks:
        x, y = int(lm.x * w), int(lm.y * h)
        points.append((x, y))
        cv2.circle(annotated, (x, y), 4, (0, 255, 0), -1)

    for start_idx, end_idx in POSE_CONNECTIONS:
        if start_idx < len(points) and end_idx < len(points):
            cv2.line(annotated, points[start_idx], points[end_idx], (255, 0, 0), 2)

    return annotated, True


def show_fall_alert(label: str, confidence: float):
    """Explicit emergency alert banner, as required by the FA-2 brief."""
    if label == FALL_LABEL:
        st.error(
            f"🚨 **EMERGENCY ALERT — FALL IN PROGRESS** 🚨\n\n"
            f"Confidence: {confidence:.1%}. Notify caregiver / emergency contact immediately."
        )
    elif label == LYING_LABEL:
        st.warning(
            f"⚠️ **Person appears to be lying down** — possible post-fall state.\n\n"
            f"Confidence: {confidence:.1%}. Check on them if this is unexpected."
        )
    else:
        st.success(f"✅ Normal activity detected: **{label}** ({confidence:.1%} confidence)")


def render_session_analytics():
    """Monitoring analytics panel: totals, fall count, normal count, distribution chart."""
    history = st.session_state.history
    st.subheader("📊 Monitoring Analytics")

    if not history:
        st.caption("No activity recorded yet this session.")
        return

    total = len(history)
    fall_count = sum(1 for h in history if h["label"] == FALL_LABEL)
    lying_count = sum(1 for h in history if h["label"] == LYING_LABEL)
    normal_count = sum(1 for h in history if h["label"] not in (FALL_LABEL, LYING_LABEL))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total activities detected", total)
    c2.metric("Fall events", fall_count)
    c3.metric("Lying events", lying_count)
    c4.metric("Normal activity count", normal_count)

    counts = Counter(h["label"] for h in history)
    dist_df = pd.DataFrame({"Activity": list(counts.keys()), "Count": list(counts.values())})
    dist_df = dist_df.set_index("Activity")
    st.bar_chart(dist_df)

    with st.expander("View detailed prediction log"):
        log_df = pd.DataFrame(history)
        log_df["time"] = pd.to_datetime(log_df["timestamp"], unit="s").dt.strftime("%H:%M:%S")
        st.dataframe(log_df[["time", "source", "label", "confidence"]], use_container_width=True)

    if st.button("🔄 Reset session analytics"):
        reset_session()
        st.rerun()


def process_video(model, class_names, pose_detector, video_path: str, show_pose: bool):
    """Samples frames from an uploaded video, classifies each, logs results,
    and shows a live progress bar plus a summary once done."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Could not open the uploaded video.")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    progress = st.progress(0, text="Processing video...")
    frame_idx = 0
    processed = 0
    fall_frame_preview = None
    last_annotated_preview = None

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        if frame_idx % VIDEO_SAMPLE_EVERY_N_FRAMES == 0:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            label, probs = classify_array(model, class_names, rgb)
            log_prediction(label, probs[label], source="video")
            processed += 1

            if show_pose:
                annotated_bgr, _ = draw_pose_on_array(pose_detector, frame_bgr)
                last_annotated_preview = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
            else:
                last_annotated_preview = rgb

            if label == FALL_LABEL and fall_frame_preview is None:
                fall_frame_preview = last_annotated_preview

        frame_idx += 1
        progress.progress(min(frame_idx / total_frames, 1.0), text=f"Processing video... ({frame_idx}/{total_frames} frames)")

    cap.release()
    progress.empty()
    st.success(f"Video processed: {processed} frames analyzed.")

    if fall_frame_preview is not None:
        st.error("🚨 A fall was detected at least once in this video.")
        st.image(fall_frame_preview, caption="First detected fall frame", use_container_width=True)
    elif last_annotated_preview is not None:
        st.image(last_annotated_preview, caption="Last analyzed frame", use_container_width=True)


# ----------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------
init_session_state()

st.title("🚨 Elderly Fall / Activity Detection")
st.caption("Upload a photo, take a picture, or upload a video to monitor activity and detect falls.")

with st.sidebar:
    st.header("Options")
    show_pose = st.checkbox("Overlay pose skeleton", value=True)
    show_probs = st.checkbox("Show class probabilities", value=True)
    st.markdown("---")
    st.caption(
        "Model expects: fall_detection_model.h5 and class_names.txt "
        "in the app folder (from the training notebook)."
    )

model = load_model()
class_names = load_class_names()
pose_detector = load_pose_detector() if show_pose else None

tab_upload, tab_camera, tab_video = st.tabs(["📁 Upload Image", "📷 Camera", "🎥 Upload Video"])

# ---- Image upload tab ----
with tab_upload:
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"], key="img_uploader")
    if uploaded_file is not None:
        image_source = Image.open(uploaded_file)
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Input")
            st.image(image_source, use_container_width=True)

        with st.spinner("Classifying..."):
            rgb_arr = np.array(image_source.convert("RGB"))
            label, probs = classify_array(model, class_names, rgb_arr)
            log_prediction(label, probs[label], source="image")

        bgr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
        if show_pose:
            annotated_bgr, person_found = draw_pose_on_array(pose_detector, bgr)
            display_img = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        else:
            display_img = rgb_arr
            person_found = None

        with col2:
            st.subheader("Result")
            st.image(display_img, use_container_width=True)
            if show_pose and person_found is False:
                st.caption("⚠️ No person detected for pose overlay.")

        st.markdown("---")
        show_fall_alert(label, probs[label])
        if show_probs:
            st.bar_chart(probs)

# ---- Camera tab ----
with tab_camera:
    camera_file = st.camera_input("Take a photo")
    if camera_file is not None:
        image_source = Image.open(camera_file)
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Input")
            st.image(image_source, use_container_width=True)

        with st.spinner("Classifying..."):
            rgb_arr = np.array(image_source.convert("RGB"))
            label, probs = classify_array(model, class_names, rgb_arr)
            log_prediction(label, probs[label], source="camera")

        bgr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
        if show_pose:
            annotated_bgr, person_found = draw_pose_on_array(pose_detector, bgr)
            display_img = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        else:
            display_img = rgb_arr
            person_found = None

        with col2:
            st.subheader("Result")
            st.image(display_img, use_container_width=True)
            if show_pose and person_found is False:
                st.caption("⚠️ No person detected for pose overlay.")

        st.markdown("---")
        show_fall_alert(label, probs[label])
        if show_probs:
            st.bar_chart(probs)

# ---- Video upload tab ----
with tab_video:
    st.caption(
        f"Frames are sampled every {VIDEO_SAMPLE_EVERY_N_FRAMES} frames "
        "(not every single frame) to keep processing fast."
    )
    video_file = st.file_uploader("Choose a video", type=["mp4", "mov", "avi", "mkv"], key="video_uploader")
    if video_file is not None:
        st.video(video_file)
        if st.button("▶️ Run analysis on this video"):
            # Write to a temp file since OpenCV needs a filesystem path
            temp_path = os.path.join("temp_uploaded_video." + video_file.name.split(".")[-1])
            with open(temp_path, "wb") as f:
                f.write(video_file.getbuffer())
            try:
                process_video(model, class_names, pose_detector, temp_path, show_pose)
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

st.markdown("---")
render_session_analytics()
