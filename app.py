"""
Elderly Fall / Activity Detection — Streamlit App (FA-2)
=========================================================
Loads the CNN trained in the companion Colab notebook
(fall_detection_model.tflite + class_names.txt) and lets a caregiver
upload an image, take a photo, or upload a video to classify activity.

Run with:
    streamlit run app.py

Expected files in the same folder as this script:
    fall_detection_model.tflite
    class_names.txt
"""

import io
import itertools
import os
import time
import urllib.request
from collections import Counter

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import tflite_runtime.interpreter as tflite
from PIL import Image

import mediapipe as mp
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
IMG_SIZE = (128, 128)
MODEL_PATH = "fall_detection_model.tflite"
CLASS_NAMES_PATH = "class_names.txt"
# If the model isn't found next to app.py (e.g. it's too big for a normal git
# push), the app will try to download it from here. Use a direct-download
# link (e.g. a GitHub Release asset URL) for files over ~50MB.
MODEL_URL = ""
CLASS_NAMES_URL = ""
FALL_LABEL = "Fall"        # must match the class name used in class_names.txt exactly
WALKING_LABEL = "Walking"  # must match the class name used in class_names.txt exactly
VIDEO_SAMPLE_EVERY_N_FRAMES = 15  # classify roughly ~2 frames/sec at 30fps video
THUMB_MAX_DIM = 220         # size of stored screenshot thumbnails

# --- Diagnostic heuristic thresholds (tune these against your own footage) ---
HORIZONTAL_ANGLE_THRESHOLD = 55
MOTION_SCORE_THRESHOLD = 10

# Environmental tagging options
UNCONFIRMED = "Unconfirmed"
LIGHTING_OPTIONS = ["Not specified", "Good / even lighting", "Dim / low light",
                     "Overly bright / glare", "Backlit"]
ANGLE_OPTIONS = ["Not specified", "Front-facing", "Side view", "Overhead",
                  "Low angle", "Partial / oblique view"]
OCCLUSION_OPTIONS = ["Not specified", "None", "Partial (furniture / limbs)",
                      "Heavy occlusion"]

st.set_page_config(page_title="Elderly Fall Detection", page_icon="🚨", layout="centered")


def is_git_lfs_pointer(path: str) -> bool:
    """Detect the classic failure mode: the file 'exists' but is actually a
    tiny Git LFS pointer (plain text) instead of the real binary."""
    try:
        if os.path.getsize(path) > 2000:
            return False
        with open(path, "rb") as f:
            head = f.read(200)
        return head.startswith(b"version https://git-lfs.github.com/spec")
    except OSError:
        return False


def looks_like_valid_tflite(path: str) -> bool:
    """TFLite files are FlatBuffers with a 'TFL3' identifier at byte offset 4."""
    try:
        with open(path, "rb") as f:
            f.seek(4)
            ident = f.read(4)
        return ident == b"TFL3"
    except OSError:
        return False


def ensure_file(path: str, url: str, friendly_name: str, silent: bool = False) -> bool:
    """Make sure `path` exists and is a real file (not an LFS pointer).
    Tries to download it from `url` if missing/broken and a URL was given."""
    needs_download = not os.path.exists(path) or is_git_lfs_pointer(path)

    if needs_download and url:
        try:
            with st.spinner(f"Downloading {friendly_name} from {url} ..."):
                urllib.request.urlretrieve(url, path)
            needs_download = not os.path.exists(path) or is_git_lfs_pointer(path)
        except Exception as e:
            if not silent:
                st.error(f"Failed to download {friendly_name} from {url}: {e}")
            return False

    if needs_download:
        if silent:
            return False
        cwd = os.getcwd()
        try:
            nearby_files = os.listdir(cwd)
        except OSError:
            nearby_files = []
        if os.path.exists(path) and is_git_lfs_pointer(path):
            st.error(
                f"⚠️ '{path}' exists but is only a Git LFS pointer file, not the real "
                f"{friendly_name}. Fix by either: (1) running `git lfs pull` where you "
                "deployed from, (2) enabling Git LFS support on your hosting platform, or "
                f"(3) setting MODEL_URL / CLASS_NAMES_URL in app.py to a direct download link."
            )
        else:
            st.error(
                f"'{path}' not found. The app is currently running from: `{cwd}`, which "
                f"contains: {nearby_files if nearby_files else '(nothing readable)'}. "
                f"Check that {friendly_name} is committed to GitHub in this exact folder, "
                "is under GitHub's ~100MB limit (or handled via Git LFS), and isn't excluded "
                "by .gitignore."
            )
        return False

    return True


def resolve_class_label(class_names, expected_label):
    """Match `expected_label` against class_names.txt case-insensitively."""
    for name in class_names:
        if name.strip().lower() == expected_label.strip().lower():
            return name
    st.warning(
        f"⚠️ Couldn't find a class named '{expected_label}' (case-insensitive) in "
        f"class_names.txt. Found: {class_names}. Alerts/flags tied to "
        f"'{expected_label}' will not work until this is fixed."
    )
    return expected_label


# ----------------------------------------------------------------------
# Cached resource loaders
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading classification model...")
def load_model():
    corruption_hint = (
        "This usually means the binary got corrupted in Git — most commonly because "
        "there's no `.gitattributes` marking the file as binary. Fix: add a `.gitattributes` "
        "file to the repo root containing:\n```\n*.tflite -text\n```\n"
        "then re-add and re-commit the model file from a fresh export."
    )

    if not ensure_file(MODEL_PATH, MODEL_URL, "TFLite model file"):
        st.stop()

    if not looks_like_valid_tflite(MODEL_PATH):
        st.error(f"'{MODEL_PATH}' was found but doesn't look like a valid TFLite file "
                 f"(missing the TFL3 header). {corruption_hint}")
        st.stop()

    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return {"type": "tflite", "model": interpreter}


@st.cache_resource(show_spinner=False)
def load_class_names():
    if not ensure_file(CLASS_NAMES_PATH, CLASS_NAMES_URL, "class_names.txt"):
        st.stop()
    with open(CLASS_NAMES_PATH, "r") as f:
        return [line.strip() for line in f if line.strip()]


@st.cache_resource(show_spinner="Loading pose model...")
def load_pose_detector():
    return mp_pose.Pose(static_image_mode=True, model_complexity=1, min_detection_confidence=0.5)


# ----------------------------------------------------------------------
# Session state — monitoring analytics accumulate across the session
# ----------------------------------------------------------------------
def init_session_state():
    if "history" not in st.session_state:
        st.session_state.history = []
    if "last_alert" not in st.session_state:
        st.session_state.last_alert = None


def array_to_thumb_bytes(rgb_array: np.ndarray, max_dim: int = THUMB_MAX_DIM) -> bytes:
    """Shrink a frame and encode it as PNG bytes for the screenshot gallery."""
    img = Image.fromarray(rgb_array).convert("RGB")
    img.thumbnail((max_dim, max_dim))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def log_prediction(label: str, confidence: float, source: str, thumb: bytes = None,
                    pose_flag: bool = False, motion_flag: bool = False, pose_angle=None):
    entry = {
        "label": label,
        "confidence": confidence,
        "source": source,
        "timestamp": time.time(),
        "actual_label": UNCONFIRMED,
        "lighting": "Not specified",
        "camera_angle": "Not specified",
        "occlusion": "Not specified",
        "thumb": thumb,
        "pose_flag": bool(pose_flag),
        "motion_flag": bool(motion_flag),
        "pose_angle": pose_angle,
    }
    st.session_state.history.append(entry)
    st.session_state.last_alert = entry


def reset_session():
    st.session_state.history = []
    st.session_state.last_alert = None


# ----------------------------------------------------------------------
# Core logic
# ----------------------------------------------------------------------
def classify_array(model_bundle, class_names, rgb_array: np.ndarray):
    """Resize/normalize an RGB numpy image and run the TFLite classifier."""
    img = Image.fromarray(rgb_array).convert("RGB").resize(IMG_SIZE)
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)

    interpreter = model_bundle["model"]
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], arr)
    interpreter.invoke()
    probs = interpreter.get_tensor(output_details[0]['index'])[0]

    pred_idx = int(np.argmax(probs))
    if pred_idx >= len(class_names):
        st.error(
            f"Model predicts {len(probs)} classes but class_names.txt only has "
            f"{len(class_names)} entries. Re-export fall_detection_model.tflite "
            "and re-upload class_names.txt so they match."
        )
        st.stop()
    label = class_names[pred_idx]
    prob_dict = {name: float(p) for name, p in zip(class_names, probs)}
    return label, prob_dict


def draw_pose_on_array(pose_detector, bgr_image):
    """Detect + draw the pose skeleton. Also returns the raw landmarks (or
    None) so callers can run the orientation heuristic without re-running
    pose detection a second time."""
    if pose_detector is None:
        return bgr_image, False, None

    rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    results = pose_detector.process(rgb)

    annotated = bgr_image.copy()
    if not results.pose_landmarks:
        return annotated, False, None

    mp_drawing.draw_landmarks(annotated, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    return annotated, True, results.pose_landmarks.landmark


def estimate_body_orientation(landmarks):
    """Rough "standing vs lying down" signal from the shoulder->hip line.
    Uses BlazePose indices: 11/12 = left/right shoulder, 23/24 = left/right hip.
    Returns the angle in degrees from vertical (0 = upright, 90 = lying), or
    None if the needed landmarks aren't available."""
    if not landmarks or len(landmarks) < 25:
        return None
    try:
        sx = (landmarks[11].x + landmarks[12].x) / 2.0
        sy = (landmarks[11].y + landmarks[12].y) / 2.0
        hx = (landmarks[23].x + landmarks[24].x) / 2.0
        hy = (landmarks[23].y + landmarks[24].y) / 2.0
    except (IndexError, AttributeError):
        return None

    dx = abs(hx - sx)
    dy = abs(hy - sy)
    if dx == 0 and dy == 0:
        return None
    angle = np.degrees(np.arctan2(dx, dy + 1e-6))
    return float(angle)


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
    return pd.DataFrame(confirmed)


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
        for i, h in enumerate(entries[-9:]):
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
# Diagnostic flags: Fall-as-Sitting confusion & missed Walking detections
# ----------------------------------------------------------------------
def render_diagnostic_flags():
    st.subheader("🩺 Model Diagnostic Flags")
    st.caption(
        "These are runtime heuristics — not ground truth — meant to help you SEE "
        "and collect evidence for known problems, since they can only be fixed "
        "by retraining the CNN with better data, not by this app alone."
    )
    history = st.session_state.history
    if not history:
        st.caption("No activity recorded yet this session.")
        return

    pose_flagged = [h for h in history if h.get("pose_flag")]
    motion_flagged = [h for h in history if h.get("motion_flag")]

    c1, c2 = st.columns(2)
    c1.metric("⚠️ Possible Fall→Sitting confusion", len(pose_flagged))
    c2.metric("⚠️ Possible missed Walking", len(motion_flagged))

    if pose_flagged:
        with st.expander(f"Frames flagged as horizontal-but-not-Fall ({len(pose_flagged)})"):
            for h in pose_flagged[-10:]:
                cols = st.columns([1, 3])
                if h.get("thumb"):
                    cols[0].image(h["thumb"], use_container_width=True)
                angle_txt = f"{h['pose_angle']:.0f}°" if h.get("pose_angle") is not None else "n/a"
                cols[1].write(
                    f"Predicted **{h['label']}** ({h['confidence']:.0%}) — "
                    f"body angle from upright: {angle_txt}"
                )
            st.markdown(
                "**Likely cause:** training Fall images may mostly show a person already "
                "still/sprawled on the floor, while Sitting shares a similar low silhouette. "
                "**To fix at the source:** add more Fall training images across the full "
                "falling motion and multiple camera angles, and ensure Sitting examples use "
                "the same camera angles."
            )

    if motion_flagged:
        with st.expander(f"Frames flagged as motion-but-not-Walking ({len(motion_flagged)})"):
            for h in motion_flagged[-10:]:
                cols = st.columns([1, 3])
                if h.get("thumb"):
                    cols[0].image(h["thumb"], use_container_width=True)
                cols[1].write(f"Predicted **{h['label']}** ({h['confidence']:.0%})")
            st.markdown(
                "**Likely cause:** Walking training data may be limited in camera angle, "
                "speed, or lighting. **To fix at the source:** add Walking clips from the "
                "actual deployed camera position, at varied speeds/lighting."
            )

    if not pose_flagged and not motion_flagged:
        st.caption("No diagnostic flags raised yet for the current session's captures.")


# ----------------------------------------------------------------------
# Video processing
# ----------------------------------------------------------------------
def process_video(model, class_names, pose_detector, video_path: str, show_pose: bool):
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
    prev_gray = None
    pose_flag_count = 0
    motion_flag_count = 0

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        if frame_idx % VIDEO_SAMPLE_EVERY_N_FRAMES == 0:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            label, probs = classify_array(model, class_names, rgb)

            pose_angle = None
            if show_pose:
                annotated_bgr, _, landmarks = draw_pose_on_array(pose_detector, frame_bgr)
                preview_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
                pose_angle = estimate_body_orientation(landmarks)
            else:
                preview_rgb = rgb

            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            motion_score = 0.0
            if prev_gray is not None:
                motion_score = float(cv2.absdiff(gray, prev_gray).mean())
            prev_gray = gray

            pose_flag = (pose_angle is not None and pose_angle > HORIZONTAL_ANGLE_THRESHOLD
                         and label != FALL_LABEL)
            motion_flag = (motion_score > MOTION_SCORE_THRESHOLD
                           and label not in (WALKING_LABEL, FALL_LABEL))
            pose_flag_count += int(pose_flag)
            motion_flag_count += int(motion_flag)

            thumb = array_to_thumb_bytes(preview_rgb)
            log_prediction(label, probs[label], source="video", thumb=thumb,
                            pose_flag=pose_flag, motion_flag=motion_flag, pose_angle=pose_angle)
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

    if pose_flag_count:
        st.warning(
            f"⚠️ {pose_flag_count} frame(s) had a horizontal/lying body orientation but "
            f"weren't classified as Fall — possible Fall-as-Sitting confusion."
        )
    if motion_flag_count:
        st.warning(
            f"⚠️ {motion_flag_count} frame(s) showed real motion but weren't classified as "
            f"Walking — possible missed Walking detections."
        )


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
FALL_LABEL = resolve_class_label(class_names, FALL_LABEL)
WALKING_LABEL = resolve_class_label(class_names, WALKING_LABEL)
pose_detector = load_pose_detector() if show_pose else None

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
        pose_angle = None
        if show_pose:
            annotated_bgr, person_found, landmarks = draw_pose_on_array(pose_detector, bgr)
            display_img = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
            pose_angle = estimate_body_orientation(landmarks)
        else:
            display_img = rgb_arr
            person_found = None

        pose_flag = (pose_angle is not None and pose_angle > HORIZONTAL_ANGLE_THRESHOLD
                     and label != FALL_LABEL)
        log_prediction(label, probs[label], source="image",
                        thumb=array_to_thumb_bytes(display_img),
                        pose_flag=pose_flag, pose_angle=pose_angle)

        with col2:
            st.subheader("Result")
            st.image(display_img, use_container_width=True)
            if show_pose and person_found is False:
                st.caption("⚠️ No person detected for pose overlay.")

        st.markdown("---")
        show_fall_alert(label, probs[label])
        if pose_flag:
            st.warning(
                f"⚠️ Body orientation looks horizontal (≈{pose_angle:.0f}° from upright) "
                f"but the model predicted **{label}**, not Fall. This may be a Fall "
                f"being confused with Sitting — please verify."
            )
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
        pose_angle = None
        if show_pose:
            annotated_bgr, person_found, landmarks = draw_pose_on_array(pose_detector, bgr)
            display_img = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
            pose_angle = estimate_body_orientation(landmarks)
        else:
            display_img = rgb_arr
            person_found = None

        pose_flag = (pose_angle is not None and pose_angle > HORIZONTAL_ANGLE_THRESHOLD
                     and label != FALL_LABEL)
        log_prediction(label, probs[label], source="camera",
                        thumb=array_to_thumb_bytes(display_img),
                        pose_flag=pose_flag, pose_angle=pose_angle)

        with col2:
            st.subheader("Result")
            st.image(display_img, use_container_width=True)
            if show_pose and person_found is False:
                st.caption("⚠️ No person detected for pose overlay.")

        st.markdown("---")
        show_fall_alert(label, probs[label])
        if pose_flag:
            st.warning(
                f"⚠️ Body orientation looks horizontal (≈{pose_angle:.0f}° from upright) "
                f"but the model predicted **{label}**, not Fall. This may be a Fall "
                f"being confused with Sitting — please verify."
            )
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
render_diagnostic_flags()
