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
- Diagnostic flags for two specific problems observed in testing:
    1) Falls being classified as Sitting: a MediaPipe pose heuristic
       estimates whether the body is lying horizontal vs upright. If the
       body looks horizontal but the CNN did NOT say "Fall", the app raises
       a caution ("pose says lying down, model said <label> — verify") and
       logs it, so you can see how often this happens.
    2) Walking never being detected: for video, the app compares consecutive
       sampled frames for motion. If real motion is present but the CNN
       predicts a static class (not Walking, not Fall), it's flagged as a
       possible missed-Walking case.
  These are runtime heuristics to help you SEE the problem and collect
  evidence — they cannot fix the underlying CNN. The real fix is retraining
  with more/varied "Fall" examples that look like sitting-on-floor poses,
  and more "Walking" clips across different angles/speeds/lighting.
- Class-name auto-check: on startup the app verifies that "Fall" and
  "Walking" actually exist in class_names.txt (case-insensitively) and warns
  you if the casing/spelling differs from what the app expects, since a
  mismatch there would silently stop alerts from ever firing.

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
H5_MODEL_PATH = "fall_detection_model.h5"  # fallback if the TFLite conversion step wasn't run
CLASS_NAMES_PATH = "class_names.txt"
# If the model isn't found next to app.py (e.g. it's too big for a normal git
# push, or Git LFS wasn't pulled), the app will try to download it from here.
# Use a direct-download link: a GitHub "Release" asset URL
# (https://github.com/<user>/<repo>/releases/download/<tag>/<file>.tflite)
# works well for files over ~50MB; raw.githubusercontent.com works for
# smaller files committed normally (not via LFS).
MODEL_URL = ""       # e.g. "https://github.com/you/repo/releases/download/v1/fall_detection_model.tflite"
H5_MODEL_URL = ""    # same idea, for the .h5 fallback
CLASS_NAMES_URL = ""  # optional, same idea for class_names.txt
FALL_LABEL = "Fall"        # must match the class name used in class_names.txt exactly
WALKING_LABEL = "Walking"  # must match the class name used in class_names.txt exactly
VIDEO_SAMPLE_EVERY_N_FRAMES = 15  # classify roughly ~2 frames/sec at 30fps video
THUMB_MAX_DIM = 220         # size of stored screenshot thumbnails

# --- Diagnostic heuristic thresholds (tune these against your own footage) ---
# Angle (degrees) from vertical, based on shoulder-to-hip line, above which the
# body is considered "lying horizontal" — used to flag Fall-vs-Sitting confusion.
HORIZONTAL_ANGLE_THRESHOLD = 55
# Mean absolute pixel difference between consecutive sampled frames above which
# we consider "real motion" present — used to flag missed Walking detections.
MOTION_SCORE_THRESHOLD = 10

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


def is_git_lfs_pointer(path: str) -> bool:
    """Detect the classic failure mode: the file 'exists' but is actually a
    tiny Git LFS pointer (plain text) instead of the real binary, because LFS
    objects weren't pulled during clone/deploy."""
    try:
        if os.path.getsize(path) > 2000:
            return False
        with open(path, "rb") as f:
            head = f.read(200)
        return head.startswith(b"version https://git-lfs.github.com/spec")
    except OSError:
        return False


def looks_like_valid_hdf5(path: str) -> bool:
    """Check the real HDF5 magic bytes, not just file size — catches a file
    that 'exists' and isn't an LFS pointer but is still corrupted (e.g. Git
    line-ending normalization mangled a binary file that wasn't marked as
    binary via .gitattributes)."""
    try:
        with open(path, "rb") as f:
            sig = f.read(8)
        return sig == b"\x89HDF\r\n\x1a\n"
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
    Tries to download it from `url` if missing/broken and a URL was given.
    Shows a detailed diagnostic (cwd contents, LFS hint) instead of just
    'not found' when it still can't locate a usable file — unless `silent`
    is True, which is used for optional/fallback attempts where a failure
    here isn't necessarily an error (e.g. trying TFLite before falling back
    to a Keras .h5 model). Returns True if a usable file is available at
    `path` afterwards.
    """
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
                f"{friendly_name}. This happens when Git LFS objects weren't pulled during "
                "deploy/clone. Fix by either: (1) running `git lfs pull` where you deployed "
                "from, (2) enabling Git LFS support on your hosting platform, or "
                f"(3) setting MODEL_URL / CLASS_NAMES_URL in app.py to a direct download link "
                "(e.g. a GitHub Release asset) so the app fetches the real file itself."
            )
        else:
            st.error(
                f"'{path}' not found. The app is currently running from: `{cwd}`, which "
                f"contains: {nearby_files if nearby_files else '(nothing readable)'}. "
                f"If {friendly_name} is committed to GitHub but not showing up here, check: "
                "the file is actually in this exact folder (not a subfolder) relative to "
                "where app.py runs, it's under GitHub's ~100MB limit or handled via Git LFS "
                "(and LFS was pulled), and it isn't excluded by .gitignore. Alternatively, set "
                "MODEL_URL / CLASS_NAMES_URL near the top of app.py to a direct download link "
                "and the app will fetch it automatically."
            )
        return False

    return True


def resolve_class_label(class_names, expected_label):
    """Match `expected_label` against class_names.txt case-insensitively.

    Returns the exact string used in class_names.txt, or `expected_label`
    unchanged if no match was found at all (with a warning shown to the user).
    A mismatch here (e.g. file has 'fall' but the app expects 'Fall') would
    otherwise silently prevent alerts from ever firing.
    """
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
    """Returns a small bundle dict: {"type": "tflite"|"keras", "model": ...}.
    Prefers a TFLite model (smaller, faster); falls back to loading the raw
    Keras .h5 model directly if no .tflite conversion was ever committed.
    Validates real file signatures (not just presence/size) so a corrupted
    binary — the classic case being a .h5/.tflite committed without a
    .gitattributes marking it as binary, so Git's line-ending normalization
    silently rewrote bytes inside it — gets a clear diagnostic instead of a
    raw OSError from deep inside TensorFlow/h5py.
    """
    corruption_hint = (
        "This usually means the binary got corrupted in Git — most commonly because "
        "there's no `.gitattributes` marking the file as binary, so Git's line-ending "
        "normalization rewrote bytes inside it. Fix: add a `.gitattributes` file to the "
        "repo root containing:\n"
        "```\n*.h5 -text\n*.tflite -text\n```\n"
        "then re-add and re-commit the model file(s) from a fresh export (you may need "
        "`git rm --cached <file>` first, since the corrupted version is already in git "
        "history) and push again."
    )

    if ensure_file(MODEL_PATH, MODEL_URL, "TFLite model file", silent=True):
        if not looks_like_valid_tflite(MODEL_PATH):
            st.warning(f"'{MODEL_PATH}' was found but doesn't look like a valid TFLite file "
                       f"(missing the TFL3 header). {corruption_hint}\n\nFalling back to "
                       f"'{H5_MODEL_PATH}' for now.")
        else:
            try:
                interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
                interpreter.allocate_tensors()
                return {"type": "tflite", "model": interpreter}
            except Exception as e:
                st.warning(f"Found '{MODEL_PATH}' but couldn't load it ({e}). "
                           f"Falling back to '{H5_MODEL_PATH}'.")

    if ensure_file(H5_MODEL_PATH, H5_MODEL_URL, "Keras (.h5) model file"):
        if not looks_like_valid_hdf5(H5_MODEL_PATH):
            st.error(f"'{H5_MODEL_PATH}' was found but isn't a valid HDF5 file (wrong file "
                     f"signature). {corruption_hint}")
            st.stop()
        st.info(
            f"No usable '{MODEL_PATH}' found — loading '{H5_MODEL_PATH}' instead. This works "
            "fine, but a valid TFLite conversion loads faster in Streamlit (optional speed-up)."
        )
        try:
            keras_model = tf.keras.models.load_model(H5_MODEL_PATH)
            return {"type": "keras", "model": keras_model}
        except Exception as e:
            st.error(f"Found a valid-looking '{H5_MODEL_PATH}' but Keras still couldn't load "
                     f"it: {e}")
            st.stop()

    st.stop()


@st.cache_resource(show_spinner=False)
def load_class_names():
    if not ensure_file(CLASS_NAMES_PATH, CLASS_NAMES_URL, "class_names.txt"):
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
    """Append a new prediction to the running history / evaluation log.

    pose_flag / motion_flag are runtime diagnostic hints (not ground truth):
    pose_flag  = pose orientation looked horizontal/lying but label wasn't Fall
    motion_flag = real motion was detected between frames but label wasn't Walking/Fall
    """
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
    """Resize/normalize an RGB numpy image and run the classifier, whether
    it's a TFLite interpreter or a raw Keras model."""
    img = Image.fromarray(rgb_array).convert("RGB").resize(IMG_SIZE)
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)

    if model_bundle["type"] == "tflite":
        interpreter = model_bundle["model"]
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        interpreter.set_tensor(input_details[0]['index'], arr)
        interpreter.invoke()
        probs = interpreter.get_tensor(output_details[0]['index'])[0]
    else:  # "keras"
        keras_model = model_bundle["model"]
        probs = keras_model.predict(arr, verbose=0)[0]

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
    """Detect + draw the pose skeleton. Also returns the raw normalized
    landmarks (or None) so callers can run the orientation heuristic without
    re-running pose detection a second time."""
    if pose_detector is None:
        return bgr_image, False, None

    rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = pose_detector.detect(mp_image)

    annotated = bgr_image.copy()
    if not result.pose_landmarks:
        return annotated, False, None

    h, w, _ = annotated.shape
    landmarks = result.pose_landmarks[0]  # first detected person
    points = []
    for lm in landmarks:
        x, y = int(lm.x * w), int(lm.y * h)
        points.append((x, y))
        cv2.circle(annotated, (x, y), 4, (0, 255, 0), -1)
    for start_idx, end_idx in POSE_CONNECTIONS:
        cv2.line(annotated, points[start_idx], points[end_idx], (255, 0, 0), 2)

    return annotated, True, landmarks


def estimate_body_orientation(landmarks):
    """Rough "standing vs lying down" signal from the shoulder->hip line.

    Uses BlazePose indices: 11/12 = left/right shoulder, 23/24 = left/right hip.
    Returns the angle in degrees between that line and the vertical axis
    (0 = perfectly upright, 90 = perfectly horizontal/lying), or None if the
    needed landmarks aren't available.
    """
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
# Diagnostic flags: Fall-as-Sitting confusion & missed Walking detections
# ----------------------------------------------------------------------
def render_diagnostic_flags():
    st.subheader("🩺 Model Diagnostic Flags")
    st.caption(
        "These are runtime heuristics — not ground truth — meant to help you SEE "
        "and collect evidence for two known problems, since neither can be fixed "
        "by this app alone (both trace back to the CNN's training data)."
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
                "**Likely cause:** the CNN was probably trained on Fall images that mostly "
                "look like a person already still/sprawled on the floor, while Sitting images "
                "share a similar low, folded silhouette from this camera angle. "
                "**To fix at the source:** add more Fall training images captured at the "
                "moment of/just after falling (not only the resting position), across "
                "multiple camera angles, and make sure Sitting examples include the same "
                "camera angles so the two classes aren't separable by camera position alone."
            )

    if motion_flagged:
        with st.expander(f"Frames flagged as motion-but-not-Walking ({len(motion_flagged)})"):
            for h in motion_flagged[-10:]:
                cols = st.columns([1, 3])
                if h.get("thumb"):
                    cols[0].image(h["thumb"], use_container_width=True)
                cols[1].write(f"Predicted **{h['label']}** ({h['confidence']:.0%})")
            st.markdown(
                "**Likely cause:** the Walking class in training was probably limited in "
                "camera angle, walking speed, or lighting compared to real usage. "
                "**To fix at the source:** add Walking clips from the actual camera "
                "position(s) this app will run with, at varied speeds and lighting, and "
                "check `class_names.txt` / the model output layer to confirm Walking is "
                "actually one of the trained classes (see the class-name check at startup)."
            )

    if not pose_flagged and not motion_flagged:
        st.caption("No diagnostic flags raised yet for the current session's captures.")


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
    prev_gray = None  # for motion detection between sampled frames
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

            # Motion check: compare this sampled frame to the previous one
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
