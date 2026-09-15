"""
Elderly Fall / Activity Detection — Streamlit App (FA-2)
=========================================================
Loads the CNN trained in the companion Colab notebook
(fall_detection_model.tflite + class_names.txt) and lets a caregiver
upload an image, take a photo, or upload a video to classify activity
into one of 5 classes: Fall, Normal, Sitting, Standing, Walking.

NEW IN THIS VERSION
--------------------
- Real-time monitoring panel (latest alert + live event timeline)
- Fall / Normal / per-class activity counts
- Emergency alert banner (kept, made more visible)
- Activity distribution charts (kept)
- Editable prediction log: caregiver can confirm the TRUE activity for any
  prediction and tag Lighting / Camera angle / Occlusion for that capture.
  This turns the running history into a small, growing evaluation set.
- Confusion matrix + auto-generated "commonly confused classes" insights,
  built live from the confirmed predictions above.
- Correct fall detections / False alarms / Missed falls / Misclassified
  activities counters, computed from confirmed predictions.
- Prediction screenshot gallery (Falls / False alarms / Misclassified / All).
- Environmental-factor analysis: false-alarm & accuracy rate broken down by
  Lighting, Camera angle and Occlusion tag, so you can see which conditions
  actually hurt the model in practice.
- Model Training Performance section: upload the accuracy/loss history you
  exported from the Colab notebook (history.history) as JSON or CSV to plot
  Accuracy and Loss curves. This app never re-trains anything itself — it
  only visualizes numbers you already computed in the notebook.

Run with:
    streamlit run app.py

Expected files in the same folder as this script:
    fall_detection_model.tflite
    class_names.txt

To get an Accuracy/Loss graph, export your Keras History object from the
training notebook, e.g.:

    import json
    with open("training_history.json", "w") as f:
        json.dump(history.history, f)

...then upload training_history.json in the "Model Training Performance"
section of this app.
"""

import io
import itertools
import json
import os
import time
import urllib.request
from collections import Counter

import cv2
import matplotlib.pyplot as plt
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
MODEL_PATH = "fall_detection_model.tflite"
CLASS_NAMES_PATH = "class_names.txt"
FALL_LABEL = "Fall"        # must match the class name used in class_names.txt exactly
VIDEO_SAMPLE_EVERY_N_FRAMES = 15  # classify roughly ~2 frames/sec at 30fps video
THUMB_MAX_DIM = 220         # size of stored screenshot thumbnails

POSE_MODEL_PATH = "pose_landmarker.task"
POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)
# Standard 33-point BlazePose skeleton connections (stable across versions)
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32),
]

# Environmental tagging options — used to analyse WHY a prediction may have
# been wrong (lighting, camera angle, occlusion), and to spot classes that
# get confused because of similar body postures.
UNCONFIRMED = "Unconfirmed"
LIGHTING_OPTIONS = ["Not specified", "Good / even lighting", "Dim / low light",
                     "Overly bright / glare", "Backlit"]
ANGLE_OPTIONS = ["Not specified", "Front-facing", "Side view", "Overhead",
                  "Low angle", "Partial / oblique view"]
OCCLUSION_OPTIONS = ["Not specified", "None", "Partial (furniture / limbs)",
                      "Heavy occlusion"]

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
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter


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
        urllib.request.urlretrieve(POSE_MODEL_URL, POSE_MODEL_PATH)
    base_options = mp_python.BaseOptions(model_asset_path=POSE_MODEL_PATH)
    options = vision.PoseLandmarkerOptions(base_options=base_options)
    return vision.PoseLandmarker.create_from_options(options)


# ----------------------------------------------------------------------
# Session state — monitoring analytics accumulate across the session
# ----------------------------------------------------------------------
def init_session_state():
    if "history" not in st.session_state:
        st.session_state.history = []
    if "training_history" not in st.session_state:
        st.session_state.training_history = None
    if "last_alert" not in st.session_state:
        st.session_state.last_alert = None


def array_to_thumb_bytes(rgb_array: np.ndarray, max_dim: int = THUMB_MAX_DIM) -> bytes:
    """Shrink a frame and encode it as PNG bytes for the screenshot gallery."""
    img = Image.fromarray(rgb_array).convert("RGB")
    img.thumbnail((max_dim, max_dim))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def log_prediction(label: str, confidence: float, source: str, thumb: bytes = None):
    """Append a new prediction to the running history / evaluation log."""
    entry = {
        "label": label,               # what the model predicted
        "confidence": confidence,
        "source": source,             # image / camera / video
        "timestamp": time.time(),
        "actual_label": UNCONFIRMED,  # caregiver can confirm/correct this later
        "lighting": "Not specified",
        "camera_angle": "Not specified",
        "occlusion": "Not specified",
        "thumb": thumb,
    }
    st.session_state.history.append(entry)
    st.session_state.last_alert = entry


def reset_session():
    st.session_state.history = []
    st.session_state.last_alert = None


# ----------------------------------------------------------------------
# Core logic
# ----------------------------------------------------------------------
def classify_array(model, class_names, rgb_array: np.ndarray):
    """Resize/normalize an RGB numpy image and run the TFLite model."""
    interpreter = model
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    img = Image.fromarray(rgb_array).convert("RGB").resize(IMG_SIZE)
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)

    interpreter.set_tensor(input_details[0]['index'], arr)
    interpreter.invoke()
    probs = interpreter.get_tensor(output_details[0]['index'])[0]

    pred_idx = int(np.argmax(probs))
    if pred_idx >= len(class_names):
        st.error(
            f"Model predicts {len(probs)} classes but class_names.txt only has "
            f"{len(class_names)} entries. Re-export fall_detection_model.tflite "
            "from your 5-class training run and re-upload class_names.txt."
        )
        st.stop()
    label = class_names[pred_idx]
    prob_dict = {name: float(p) for name, p in zip(class_names, probs)}
    return label, prob_dict


def draw_pose_on_array(pose_detector, bgr_image):
    if pose_detector is None:
        return bgr_image, False

    rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = pose_detector.detect(mp_image)

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
        cv2.line(annotated, points[start_idx], points[end_idx], (255, 0, 0), 2)

    return annotated, True


def show_fall_alert(label: str, confidence: float):
    """Explicit emergency alert banner, as required by the FA-2 brief."""
    if label == FALL_LABEL:
        st.error(
            f"🚨 **EMERGENCY ALERT — FALL IN PROGRESS** 🚨\n\n"
            f"Confidence: {confidence:.1%}. Notify caregiver / emergency contact immediately."
        )
    else:
        st.success(f"✅ Activity detected: **{label}** ({confidence:.1%} confidence)")


# ----------------------------------------------------------------------
# Real-time monitoring panel
# ----------------------------------------------------------------------
def render_realtime_monitor():
    st.subheader("🔴 Real-Time Monitoring Panel")
    last = st.session_state.last_alert

    if last is None:
        st.info("Waiting for the first capture (image, camera, or video frame)...")
        return

    age_sec = time.time() - last["timestamp"]
    col1, col2, col3 = st.columns(3)
    col1.metric("Latest activity", last["label"])
    col2.metric("Confidence", f"{last['confidence']:.1%}")
    col3.metric("Seconds since last update", f"{age_sec:.0f}s")

    if last["label"] == FALL_LABEL:
        st.error("🚨 Most recent capture was classified as a **FALL**. Verify immediately.")
    else:
        st.success(f"Most recent capture: **{last['label']}** — no fall currently detected.")

    # Live timeline of fall vs non-fall events across the session
    history = st.session_state.history
    if len(history) >= 2:
        timeline_df = pd.DataFrame(
            {"is_fall": [1 if h["label"] == FALL_LABEL else 0 for h in history]}
        )
        st.caption("Fall events over the session (1 = fall detected, 0 = normal activity)")
        st.line_chart(timeline_df)


# ----------------------------------------------------------------------
# Editable prediction log (confirm true label + tag conditions)
# ----------------------------------------------------------------------
def render_editable_log(class_names):
    st.subheader("📝 Prediction Log — confirm activity & tag conditions")
    st.caption(
        "For any capture, set the TRUE activity if the model got it wrong, and "
        "optionally tag the lighting / camera angle / occlusion at capture time. "
        "This builds the data used for the confusion matrix and condition analysis below."
    )
    history = st.session_state.history
    if not history:
        st.caption("No predictions logged yet.")
        return

    log_df = pd.DataFrame(history)
    log_df["idx"] = range(len(log_df))
    log_df["time"] = pd.to_datetime(log_df["timestamp"], unit="s").dt.strftime("%H:%M:%S")
    display_cols = ["idx", "time", "source", "label", "confidence",
                     "actual_label", "lighting", "camera_angle", "occlusion"]

    edited_df = st.data_editor(
        log_df[display_cols],
        hide_index=True,
        use_container_width=True,
        key="log_editor",
        disabled=["idx", "time", "source", "label", "confidence"],
        column_config={
            "label": st.column_config.TextColumn("Predicted"),
            "confidence": st.column_config.NumberColumn("Confidence", format="%.1%%"),
            "actual_label": st.column_config.SelectboxColumn(
                "True activity", options=[UNCONFIRMED] + class_names, required=True
            ),
            "lighting": st.column_config.SelectboxColumn("Lighting", options=LIGHTING_OPTIONS),
            "camera_angle": st.column_config.SelectboxColumn("Camera angle", options=ANGLE_OPTIONS),
            "occlusion": st.column_config.SelectboxColumn("Occlusion", options=OCCLUSION_OPTIONS),
        },
    )

    # Write edits back into session state history
    for _, row in edited_df.iterrows():
        i = int(row["idx"])
        st.session_state.history[i]["actual_label"] = row["actual_label"]
        st.session_state.history[i]["lighting"] = row["lighting"]
        st.session_state.history[i]["camera_angle"] = row["camera_angle"]
        st.session_state.history[i]["occlusion"] = row["occlusion"]


# ----------------------------------------------------------------------
# Confusion matrix, correctness counters, and "similar posture" insights
# ----------------------------------------------------------------------
def get_confirmed_df(class_names):
    history = st.session_state.history
    confirmed = [h for h in history if h["actual_label"] != UNCONFIRMED]
    if not confirmed:
        return None
    df = pd.DataFrame(confirmed)
    return df


def render_confusion_matrix(class_names):
    st.subheader("📉 Confusion Matrix & Detection Accuracy")
    df = get_confirmed_df(class_names)
    if df is None:
        st.caption(
            "No confirmed predictions yet. Use the editable log above to set the "
            "TRUE activity for at least a few captures to build a confusion matrix."
        )
        return

    correct = int((df["actual_label"] == df["label"]).sum())
    total = len(df)
    incorrect = total - correct

    fall_correct = int(((df["label"] == FALL_LABEL) & (df["actual_label"] == FALL_LABEL)).sum())
    false_alarms = int(((df["label"] == FALL_LABEL) & (df["actual_label"] != FALL_LABEL)).sum())
    missed_falls = int(((df["actual_label"] == FALL_LABEL) & (df["label"] != FALL_LABEL)).sum())
    other_misclassified = incorrect - false_alarms - missed_falls

    c1, c2 = st.columns(2)
    c1.metric("Confirmed predictions", total)
    c2.metric("Overall accuracy (confirmed)", f"{(correct / total):.1%}" if total else "—")

    c3, c4, c5, c6 = st.columns(4)
    c3.metric("✅ Correct fall detections", fall_correct)
    c4.metric("🚫 False alarms", false_alarms)
    c5.metric("🙈 Missed falls", missed_falls)
    c6.metric("❓ Other misclassified", max(other_misclassified, 0))

    # Build confusion matrix over the full class list so every class shows up
    crosstab = pd.crosstab(df["actual_label"], df["label"])
    crosstab = crosstab.reindex(index=class_names, columns=class_names, fill_value=0)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(crosstab.values, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix (confirmed predictions)")
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            val = crosstab.values[i, j]
            ax.text(j, i, str(val), ha="center", va="center",
                     color="white" if val > crosstab.values.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    st.pyplot(fig)

    # Auto-generated "commonly confused classes" insight — useful for spotting
    # similar-body-posture issues (e.g. Sitting vs Fall, Standing vs Walking).
    insights = []
    for actual_cls, pred_cls in itertools.permutations(class_names, 2):
        count = crosstab.loc[actual_cls, pred_cls]
        if count > 0:
            insights.append((count, actual_cls, pred_cls))
    insights.sort(reverse=True)
    if insights:
        st.markdown("**Commonly confused classes (possible similar-posture issues):**")
        for count, actual_cls, pred_cls in insights[:5]:
            st.write(f"- **{actual_cls}** was predicted as **{pred_cls}** {count} time(s).")


# ----------------------------------------------------------------------
# Environmental-factor analysis (lighting / angle / occlusion)
# ----------------------------------------------------------------------
def render_environment_analysis(class_names):
    st.subheader("🔎 Environmental Factor Analysis")
    st.caption(
        "Breaks down accuracy and false-alarm rate by the conditions tagged in the "
        "prediction log, to see which real-world factors actually hurt the model."
    )
    df = get_confirmed_df(class_names)
    if df is None:
        st.caption("No confirmed + tagged predictions yet.")
        return

    def summarize(group_col, options):
        rows = []
        for value in options:
            if value == "Not specified":
                continue
            subset = df[df[group_col] == value]
            if subset.empty:
                continue
            acc = (subset["actual_label"] == subset["label"]).mean()
            false_alarm_rate = (
                (subset["label"] == FALL_LABEL) & (subset["actual_label"] != FALL_LABEL)
            ).mean()
            rows.append({"Condition": value, "Accuracy": acc,
                         "False alarm rate": false_alarm_rate, "Samples": len(subset)})
        return pd.DataFrame(rows)

    tab_light, tab_angle, tab_occ = st.tabs(["💡 Lighting", "📷 Camera angle", "🧱 Occlusion"])

    with tab_light:
        light_df = summarize("lighting", LIGHTING_OPTIONS)
        if light_df.empty:
            st.caption("No lighting-tagged data yet.")
        else:
            st.dataframe(light_df, hide_index=True, use_container_width=True)
            st.bar_chart(light_df.set_index("Condition")[["Accuracy", "False alarm rate"]])

    with tab_angle:
        angle_df = summarize("camera_angle", ANGLE_OPTIONS)
        if angle_df.empty:
            st.caption("No camera-angle-tagged data yet.")
        else:
            st.dataframe(angle_df, hide_index=True, use_container_width=True)
            st.bar_chart(angle_df.set_index("Condition")[["Accuracy", "False alarm rate"]])

    with tab_occ:
        occ_df = summarize("occlusion", OCCLUSION_OPTIONS)
        if occ_df.empty:
            st.caption("No occlusion-tagged data yet.")
        else:
            st.dataframe(occ_df, hide_index=True, use_container_width=True)
            st.bar_chart(occ_df.set_index("Condition")[["Accuracy", "False alarm rate"]])


# ----------------------------------------------------------------------
# Prediction screenshot gallery
# ----------------------------------------------------------------------
def render_screenshot_gallery():
    st.subheader("🖼️ Prediction Screenshots")
    history = [h for h in st.session_state.history if h.get("thumb")]
    if not history:
        st.caption("No screenshots captured yet.")
        return

    falls = [h for h in history if h["label"] == FALL_LABEL]
    false_alarms = [h for h in history
                     if h["label"] == FALL_LABEL and h["actual_label"] not in (FALL_LABEL, UNCONFIRMED)]
    misclassified = [h for h in history
                      if h["actual_label"] != UNCONFIRMED and h["actual_label"] != h["label"]]

    tab_all, tab_falls, tab_false, tab_mis = st.tabs(
        [f"All ({len(history)})", f"🚨 Falls ({len(falls)})",
         f"🚫 False alarms ({len(false_alarms)})", f"❓ Misclassified ({len(misclassified)})"]
    )

    def show_grid(entries):
        if not entries:
            st.caption("Nothing in this category yet.")
            return
        cols = st.columns(3)
        for i, h in enumerate(entries[-9:]):  # most recent 9 to keep it light
            with cols[i % 3]:
                st.image(h["thumb"], use_container_width=True)
                caption = f"{h['label']} ({h['confidence']:.0%})"
                if h["actual_label"] != UNCONFIRMED:
                    caption += f" — actual: {h['actual_label']}"
                st.caption(caption)

    with tab_all:
        show_grid(history)
    with tab_falls:
        show_grid(falls)
    with tab_false:
        show_grid(false_alarms)
    with tab_mis:
        show_grid(misclassified)


# ----------------------------------------------------------------------
# Activity distribution + full analytics panel
# ----------------------------------------------------------------------
def render_session_analytics(class_names):
    history = st.session_state.history
    st.subheader("📊 Activity Distribution")

    if not history:
        st.caption("No activity recorded yet this session.")
        return

    total = len(history)
    fall_count = sum(1 for h in history if h["label"] == FALL_LABEL)
    non_fall_count = total - fall_count

    c1, c2, c3 = st.columns(3)
    c1.metric("Total activities detected", total)
    c2.metric("Fall events", fall_count)
    c3.metric("Normal / non-fall activities", non_fall_count)

    counts = Counter(h["label"] for h in history)
    dist_df = pd.DataFrame({"Activity": list(counts.keys()), "Count": list(counts.values())})
    dist_df = dist_df.set_index("Activity")
    st.bar_chart(dist_df)

    if st.button("🔄 Reset session analytics"):
        reset_session()
        st.rerun()


# ----------------------------------------------------------------------
# Model training performance (accuracy / loss curves from the notebook)
# ----------------------------------------------------------------------
def render_training_performance():
    st.subheader("🧪 Model Training Performance")
    st.caption(
        "Upload the training history exported from the Colab notebook "
        "(JSON from `history.history`, or a CSV with accuracy/val_accuracy/loss/val_loss "
        "columns) to view the Accuracy and Loss curves used to validate the model."
    )

    uploaded = st.file_uploader(
        "Upload training_history.json or .csv", type=["json", "csv"], key="history_uploader"
    )
    if uploaded is not None:
        try:
            if uploaded.name.endswith(".json"):
                hist = json.load(uploaded)
            else:
                hist_df = pd.read_csv(uploaded)
                hist = {col: hist_df[col].tolist() for col in hist_df.columns}
            st.session_state.training_history = hist
        except Exception as e:
            st.error(f"Could not parse the uploaded file: {e}")

    hist = st.session_state.training_history
    if not hist:
        st.caption("No training history loaded yet.")
        return

    acc_keys = [k for k in hist if "acc" in k.lower()]
    loss_keys = [k for k in hist if "loss" in k.lower()]

    if acc_keys:
        st.markdown("**Accuracy Graph**")
        acc_df = pd.DataFrame({k: hist[k] for k in acc_keys})
        acc_df.index.name = "Epoch"
        st.line_chart(acc_df)

    if loss_keys:
        st.markdown("**Loss Graph**")
        loss_df = pd.DataFrame({k: hist[k] for k in loss_keys})
        loss_df.index.name = "Epoch"
        st.line_chart(loss_df)

    if not acc_keys and not loss_keys:
        st.warning(
            "Uploaded file didn't contain any keys with 'acc' or 'loss' in the name — "
            "check the export from the notebook."
        )


# ----------------------------------------------------------------------
# Video processing
# ----------------------------------------------------------------------
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

            if show_pose:
                annotated_bgr, _ = draw_pose_on_array(pose_detector, frame_bgr)
                preview_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
            else:
                preview_rgb = rgb

            thumb = array_to_thumb_bytes(preview_rgb)
            log_prediction(label, probs[label], source="video", thumb=thumb)
            processed += 1
            last_annotated_preview = preview_rgb

            if label == FALL_LABEL and fall_frame_preview is None:
                fall_frame_preview = preview_rgb

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
        "Model expects: fall_detection_model.tflite and class_names.txt "
        "in the app folder (from the training notebook)."
    )

model = load_model()
class_names = load_class_names()
pose_detector = load_pose_detector() if show_pose else None

# Real-time monitoring panel sits at the top so it's always visible
render_realtime_monitor()
st.markdown("---")

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

        bgr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
        if show_pose:
            annotated_bgr, person_found = draw_pose_on_array(pose_detector, bgr)
            display_img = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        else:
            display_img = rgb_arr
            person_found = None

        log_prediction(label, probs[label], source="image", thumb=array_to_thumb_bytes(display_img))

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

        bgr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
        if show_pose:
            annotated_bgr, person_found = draw_pose_on_array(pose_detector, bgr)
            display_img = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        else:
            display_img = rgb_arr
            person_found = None

        log_prediction(label, probs[label], source="camera", thumb=array_to_thumb_bytes(display_img))

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
            temp_path = os.path.join("temp_uploaded_video." + video_file.name.split(".")[-1])
            with open(temp_path, "wb") as f:
                f.write(video_file.getbuffer())
            try:
                process_video(model, class_names, pose_detector, temp_path, show_pose)
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

st.markdown("---")
render_session_analytics(class_names)
st.markdown("---")
render_editable_log(class_names)
st.markdown("---")
render_confusion_matrix(class_names)
st.markdown("---")
render_environment_analysis(class_names)
st.markdown("---")
render_screenshot_gallery()
st.markdown("---")
render_training_performance()
