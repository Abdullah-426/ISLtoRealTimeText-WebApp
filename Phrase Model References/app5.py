#!/usr/bin/env python3
"""
app5.py — ISL v5 Real-time Tester (Time-Resampled Segments + Strong TTA)

What’s new vs app4:
- No motion gate after first start; as soon as cooldown ends we start capturing.
- Time-based capture (collect at native webcam rate), then RESAMPLE to exactly 48 frames.
- Strong TTA: temporal shifts (±2), time-warps (0.9x, 1.0x, 1.1x), optional hand-focus (downweight face/pose).
- Same feature layout as collector; optional deltas; quality + confidence rules.

Keys:
  ESC quit | SPACE add space | B/Backspace delete | C clear | S save transcript
  P pause | +/- zoom
"""

import os
import time
import json
import argparse
from collections import deque

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import regularizers

# --------- Feature layout (MATCHES COLLECTOR) ----------
POSE_LM = 33     # (x,y,z,visibility)
FACE_LM = 468    # (x,y,z)
HAND_LM = 21     # (x,y,z)

POSE_DIM = POSE_LM * 4        # 132
FACE_DIM = FACE_LM * 3        # 1404
L_HAND_DIM = HAND_LM * 3      # 63
R_HAND_DIM = HAND_LM * 3      # 63
FRAME_DIM = POSE_DIM + FACE_DIM + L_HAND_DIM + R_HAND_DIM  # 1662

# --------- UI / Camera ----------
DISPLAY_SCALE = 0.95
DISPLAY_MIN, DISPLAY_MAX = 0.50, 1.30
CAP_WIDTH = 1280
CAP_HEIGHT = 720

# --------- MediaPipe ----------
try:
    import mediapipe as mp
    mp_holistic = mp.solutions.holistic
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles
    USE_MP = True
except Exception:
    print("[ERROR] mediapipe not installed. Run: pip install mediapipe")
    USE_MP = False


# --------- Utils ----------
def load_labels(labels_path: str):
    with open(labels_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, dict) and "classes" in obj:
        return obj["classes"]
    if isinstance(obj, dict) and "label2idx" in obj:
        return [c for c, _ in sorted(obj["label2idx"].items(), key=lambda kv: kv[1])]
    if isinstance(obj, list):
        return obj
    raise ValueError(f"Unrecognized labels format: {labels_path}")


def save_transcript(text):
    os.makedirs("transcripts", exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join("transcripts", f"isl_transcript_{stamp}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


def draw_bar(panel, x, y, w, h, frac, color=(0, 200, 0)):
    cv2.rectangle(panel, (x, y), (x + w, y + h), (70, 70, 70), 2)
    ww = int(w * max(0.0, min(1.0, frac)))
    cv2.rectangle(panel, (x, y), (x + ww, y + h), color, -1)


def draw_top3(panel, x, y, top3):
    for k, (lbl, c) in enumerate(top3):
        yy = y + 24 * k
        bar_w = int(300 * float(c))
        cv2.rectangle(panel, (x, yy), (x + 300, yy + 18), (45, 45, 45), 1)
        cv2.rectangle(panel, (x, yy), (x + bar_w, yy + 18), (0, 180, 0), -1)
        cv2.putText(panel, f"{lbl}: {float(c)*100:.1f}%", (x + 310, yy + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (225, 240, 225), 1, cv2.LINE_AA)


def draw_status(panel, txt, x=20, y=32, color=(0, 255, 255)):
    cv2.putText(panel, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.82, color, 2, cv2.LINE_AA)


# --------- Feature extraction (MATCH COLLECTOR) ----------
def extract_keypoints(results):
    """Return (vector1662, hands_present_bool)."""
    if results.pose_landmarks:
        pose = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark],
                        dtype=np.float32).flatten()
    else:
        pose = np.zeros((POSE_DIM,), dtype=np.float32)

    if results.face_landmarks:
        face = np.array([[lm.x, lm.y, lm.z] for lm in results.face_landmarks.landmark],
                        dtype=np.float32).flatten()
    else:
        face = np.zeros((FACE_DIM,), dtype=np.float32)

    if results.left_hand_landmarks:
        lh = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark],
                      dtype=np.float32).flatten()
    else:
        lh = np.zeros((L_HAND_DIM,), dtype=np.float32)

    if results.right_hand_landmarks:
        rh = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark],
                      dtype=np.float32).flatten()
    else:
        rh = np.zeros((R_HAND_DIM,), dtype=np.float32)

    v = np.concatenate([pose, face, lh, rh], axis=0)
    hands_present = (results.left_hand_landmarks is not None) or (
        results.right_hand_landmarks is not None)
    return v, hands_present


# --------- Deltas ----------
def add_deltas_seq(x: np.ndarray) -> np.ndarray:
    dx = np.concatenate([x[:1], x[1:] - x[:-1]], axis=0)
    return np.concatenate([x, dx], axis=-1)


# --------- Temporal interpolation (vectorized) ----------
def interp_time_sequence(X, t, t_new):
    """
    X: (N, D) features at times t (shape N,), return (M, D) at t_new (shape M,)
    Linear interpolation with clamped edges.
    """
    N, D = X.shape
    t = np.asarray(t, dtype=np.float64)
    t_new = np.asarray(t_new, dtype=np.float64)
    # Normalize to index space [0, N-1] to reuse fast vector interpolation
    if N == 1:
        return np.repeat(X, repeats=len(t_new), axis=0)
    # Map times to fractional indices
    s = (t_new - t[0]) / max(1e-9, (t[-1] - t[0])) * (N - 1)
    s = np.clip(s, 0.0, N - 1.0)
    i0 = np.floor(s).astype(np.int64)
    i1 = np.clip(i0 + 1, 0, N - 1)
    w = (s - i0).astype(np.float32)[:, None]  # (M,1)
    Y = (1.0 - w) * X[i0, :] + w * X[i1, :]
    return Y.astype(np.float32)


def resample_to_T(frames, times, out_T=48):
    """frames: list of (D,), times: list of timestamps -> (T,D) evenly spaced in time."""
    X = np.stack(frames, axis=0).astype(np.float32)  # (N,D)
    t = np.asarray(times, dtype=np.float64)
    if len(t) == 0:
        return np.zeros((out_T, X.shape[1]), dtype=np.float32)
    if len(t) == 1:
        return np.repeat(X, repeats=out_T, axis=0)
    t_new = np.linspace(t[0], t[-1], out_T, dtype=np.float64)
    return interp_time_sequence(X, t, t_new)


# --------- Attention layer ----------
class TemporalAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, units=128, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.proj = tf.keras.layers.Dense(units, activation="tanh")
        self.score = tf.keras.layers.Dense(1, activation=None)
        self.softmax = tf.keras.layers.Softmax(axis=1)

    def call(self, x, mask=None):
        s = self.proj(x)
        s = self.score(s)
        if mask is not None:
            mask_f = tf.cast(mask[:, :, None], dtype=s.dtype)
            s = s + (1.0 - mask_f) * tf.constant(-1e9, dtype=s.dtype)
        w = self.softmax(s)
        return tf.reduce_sum(x * w, axis=1)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"units": self.units})
        return cfg


# --------- v5 models ----------
def build_lstm_model(num_classes, seq_len, feat_dim,
                     lstm_w1=224, lstm_w2=128, dropout=0.45, l2_reg=1e-4):
    reg = regularizers.l2(l2_reg)
    inp = tf.keras.Input(shape=(seq_len, feat_dim), name="seq")
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(lstm_w1, return_sequences=True,
                             kernel_regularizer=reg, recurrent_regularizer=reg))(inp)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Dropout(0.30)(x)
    y = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(lstm_w2, return_sequences=True,
                             kernel_regularizer=reg, recurrent_regularizer=reg))(x)
    y = tf.keras.layers.LayerNormalization()(y)
    if int(x.shape[-1]) != int(y.shape[-1]):
        x = tf.keras.layers.Dense(
            int(y.shape[-1]), activation=None, kernel_regularizer=reg)(x)
    x = tf.keras.layers.Add()([x, y])
    x = TemporalAttentionLayer(units=128, name="temporal_attention")(x)
    x = tf.keras.layers.Dense(256, activation=None, kernel_regularizer=reg)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(128, activation=None, kernel_regularizer=reg)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    out = tf.keras.layers.Dense(
        num_classes, activation="softmax", dtype="float32")(x)
    return tf.keras.Model(inp, out, name="ISL_BiLSTM_Attn_v5")


def TCNBlock(filters, kernel_size=5, dilation_base=2, n_stacks=2, dropout=0.25, l2_reg=1e-4):
    reg = regularizers.l2(l2_reg)

    def f(x):
        for s in range(n_stacks):
            dil = dilation_base ** s
            y = tf.keras.layers.Conv1D(filters, kernel_size, padding="causal",
                                       dilation_rate=dil, kernel_regularizer=reg)(x)
            y = tf.keras.layers.BatchNormalization()(y)
            y = tf.keras.layers.Activation("relu")(y)
            y = tf.keras.layers.Dropout(dropout)(y)
            if int(x.shape[-1]) != int(y.shape[-1]):
                x = tf.keras.layers.Conv1D(
                    filters, 1, padding="same", kernel_regularizer=reg)(x)
            x = tf.keras.layers.Add()([x, y])
        return x
    return f


def build_tcn_model(num_classes, seq_len, feat_dim, dropout=0.45, l2_reg=1e-4):
    inp = tf.keras.Input(shape=(seq_len, feat_dim), name="seq")
    x = tf.keras.layers.Dense(256, activation="relu")(inp)
    x = tf.keras.layers.Dropout(0.25)(x)
    x = TCNBlock(256, kernel_size=5, dilation_base=2,
                 n_stacks=3, dropout=0.25, l2_reg=l2_reg)(x)
    x = TCNBlock(256, kernel_size=3, dilation_base=2,
                 n_stacks=2, dropout=0.25, l2_reg=l2_reg)(x)
    x = TemporalAttentionLayer(units=128, name="temporal_attention")(x)
    x = tf.keras.layers.Dense(256, activation=None)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(128, activation=None)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    out = tf.keras.layers.Dense(
        num_classes, activation="softmax", dtype="float32")(x)
    return tf.keras.Model(inp, out, name="ISL_TCN_v5")


# --------- TTA helpers ----------
def shift_clip(x, delta):
    """x:(T,D) -> shifted with edge padding."""
    if delta == 0:
        return x
    T = x.shape[0]
    if delta > 0:
        pad = np.repeat(x[:1], delta, axis=0)
        return np.concatenate([pad, x[:-delta]], axis=0)
    else:
        d = -delta
        pad = np.repeat(x[-1:], d, axis=0)
        return np.concatenate([x[d:], pad], axis=0)


def time_warp(x, speed=1.0):
    """
    Simple time-warp: create positions s = linspace(0, T-1, T)/speed, clamp to [0, T-1],
    then interpolate along time for all features.
    """
    T, D = x.shape
    if T == 1 or abs(speed - 1.0) < 1e-6:
        return x
    pos = np.linspace(0, T-1, T, dtype=np.float64) / max(1e-6, speed)
    pos = np.clip(pos, 0.0, T-1.0)
    i0 = np.floor(pos).astype(np.int64)
    i1 = np.clip(i0 + 1, 0, T-1)
    w = (pos - i0).astype(np.float32)[:, None]
    return ((1.0 - w) * x[i0, :] + w * x[i1, :]).astype(np.float32)


def hand_focus_variant(x, face_pose_scale=0.75):
    """Downweight face+pose dims to let hands drive more (keeps distribution closer than zeroing)."""
    x2 = x.copy()
    # Scale pose and face blocks
    x2[:, :POSE_DIM] *= face_pose_scale
    x2[:, POSE_DIM:POSE_DIM+FACE_DIM] *= face_pose_scale
    return x2


def build_tta_set(x_in, do_shift=True, do_warp=True, do_hand_focus=True,
                  shift_vals=(-2, 0, +2), warp_speeds=(0.9, 1.0, 1.1), face_pose_scale=0.75):
    variants = []
    # base or hand-focus base
    bases = [x_in]
    if do_hand_focus:
        bases.append(hand_focus_variant(x_in, face_pose_scale=face_pose_scale))
    for b in bases:
        tmp = [b]
        if do_warp:
            tmp = [time_warp(b, s) for s in warp_speeds]
        if do_shift:
            tmp2 = []
            for t in tmp:
                for dv in shift_vals:
                    tmp2.append(shift_clip(t, dv))
            variants.extend(tmp2)
        else:
            variants.extend(tmp)
    return variants  # list of (T,D)


def pool_probs(P, pool="max"):
    """P: (N, C) -> (C,)"""
    return np.max(P, axis=0) if pool == "max" else np.mean(P, axis=0)


# --------- Quality + Commit ----------
def segment_quality(hand_present_flags, min_hand_frames):
    return int(np.sum(hand_present_flags)) >= int(min_hand_frames)


def should_commit(probs, conf_hi, conf_lo, margin):
    order = np.argsort(-probs)
    p1 = float(probs[order[0]])
    p2 = float(probs[order[1]] if len(order) > 1 else 0.0)
    if p1 >= conf_hi:
        return True
    if p1 >= conf_lo and (p1 - p2) >= margin:
        return True
    return False


# --------- Loader ----------
def load_models(mode, labels_path, add_deltas, lstm_weights, tcn_weights,
                lstm_w1=224, lstm_w2=128, lstm_dropout=0.45, lstm_l2=1e-4,
                tcn_dropout=0.45, tcn_l2=1e-4, seq_len=48):
    classes = load_labels(labels_path)
    num_classes = len(classes)
    feat_dim = FRAME_DIM * (2 if add_deltas else 1)

    lstm_model = None
    tcn_model = None

    if mode in ("lstm", "ensemble"):
        if not lstm_weights:
            raise SystemExit(
                "[ERROR] --lstm_weights is required for mode lstm/ensemble")
        lstm_model = build_lstm_model(num_classes, seq_len, feat_dim,
                                      lstm_w1=lstm_w1, lstm_w2=lstm_w2,
                                      dropout=lstm_dropout, l2_reg=lstm_l2)
        lstm_model.load_weights(lstm_weights)
        print(f"[OK] Loaded LSTM weights: {lstm_weights}")

    if mode in ("tcn", "ensemble"):
        if not tcn_weights:
            raise SystemExit(
                "[ERROR] --tcn_weights is required for mode tcn/ensemble")
        tcn_model = build_tcn_model(num_classes, seq_len, feat_dim,
                                    dropout=tcn_dropout, l2_reg=tcn_l2)
        tcn_model.load_weights(tcn_weights)
        print(f"[OK] Loaded TCN weights: {tcn_weights}")

    return classes, lstm_model, tcn_model


# --------- Main ----------
def main():
    ap = argparse.ArgumentParser(
        description="ISL v5 segmented tester (time-resampled + strong TTA)")
    ap.add_argument("--mode", type=str,
                    choices=["lstm", "tcn", "ensemble"], default="tcn")
    ap.add_argument("--labels", type=str, required=True,
                    help="labels.json path")
    ap.add_argument("--lstm_weights", type=str, default=None)
    ap.add_argument("--tcn_weights", type=str, default=None)
    ap.add_argument("--add_deltas", action="store_true",
                    help="Use if trained with --add_deltas")

    ap.add_argument("--frames", type=int, default=48,
                    help="Frames per segment (seq_len)")
    ap.add_argument("--target_fps", type=float, default=20.0,
                    help="Desired timeline FPS used for capture duration")
    ap.add_argument("--segment_cooldown", type=float, default=1.5,
                    help="Cooldown seconds between segments")

    # Confidence rules
    ap.add_argument("--conf", type=float, default=0.72, help="High threshold")
    ap.add_argument("--conf_lo", type=float, default=0.55,
                    help="Low threshold with margin")
    ap.add_argument("--margin", type=float,
                    default=0.20, help="p1 - p2 margin")

    # Ensemble weights
    ap.add_argument("--ens_w_tcn", type=float, default=0.8)
    ap.add_argument("--ens_w_lstm", type=float, default=0.2)

    # TTA knobs
    ap.add_argument("--tta_pool", type=str,
                    choices=["mean", "max"], default="max")
    # disable ±2 shifts
    ap.add_argument("--no_shift", action="store_true")
    # disable 0.9/1.1 warps
    ap.add_argument("--no_warp", action="store_true")
    # disable hand-focus variant
    ap.add_argument("--no_hand_focus", action="store_true")

    # Quality
    ap.add_argument("--min_hand_frames", type=int, default=10,
                    help="Min frames with any hand visible")

    # Start flag (two hands once)
    ap.add_argument("--start_hold_frames", type=int, default=8,
                    help="Frames with both hands to arm the system first time")

    # Camera/display
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--width", type=int, default=CAP_WIDTH)
    ap.add_argument("--height", type=int, default=CAP_HEIGHT)
    ap.add_argument("--scale", type=float, default=DISPLAY_SCALE)

    # Holistic thresholds
    ap.add_argument("--det_conf", type=float, default=0.50)
    ap.add_argument("--trk_conf", type=float, default=0.50)
    ap.add_argument("--model_complexity", type=int, default=1)

    # Debug dump
    ap.add_argument("--dump_dir", type=str, default=None)

    args = ap.parse_args()

    print(
        f"[INFO] TensorFlow {tf.__version__} | GPUs: {tf.config.list_physical_devices('GPU')}")
    if not USE_MP:
        return

    # Load models
    classes, lstm_model, tcn_model = load_models(
        mode=args.mode,
        labels_path=args.labels,
        add_deltas=args.add_deltas,
        lstm_weights=args.lstm_weights,
        tcn_weights=args.tcn_weights,
        lstm_w1=224, lstm_w2=128, lstm_dropout=0.45, lstm_l2=1e-4,
        tcn_dropout=0.45, tcn_l2=1e-4,
        seq_len=args.frames
    )
    num_classes = len(classes)
    print(f"[INFO] #classes={num_classes}")

    # Camera
    cap = cv2.VideoCapture(args.cam)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    cv2.namedWindow("ISL v5 (Resampled)", cv2.WINDOW_NORMAL)
    display_scale = float(np.clip(args.scale, DISPLAY_MIN, DISPLAY_MAX))

    # States
    STATE_WAIT_START = 0
    STATE_CAPTURE = 1
    STATE_PREDICT = 2
    STATE_COOLDOWN = 3

    state = STATE_WAIT_START
    paused = False

    # Capture duration (seconds) for one segment timeline
    seg_duration = float(args.frames) / max(1e-6, args.target_fps)

    # Buffers for one segment (raw, time-stamped)
    raw_frames = []
    raw_times = []
    hand_flags = []

    # Start gating
    both_hands_count = 0
    armed_once = False

    # COOLDOWN
    cooldown_t0 = None

    # FPS overlay
    fps_hist = deque(maxlen=30)
    last_t = time.time()

    # Dump dir
    if args.dump_dir:
        os.makedirs(args.dump_dir, exist_ok=True)

    with mp_holistic.Holistic(
        model_complexity=args.model_complexity,
        refine_face_landmarks=False,  # EXACT 468 face points
        min_detection_confidence=args.det_conf,
        min_tracking_confidence=args.trk_conf
    ) as holistic:

        print("[INFO] Controls: ESC quit | Space space | B backspace | C clear | +/- zoom | S save | P pause")

        typed_text = ""
        seg_t0 = None

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            # Landmarks
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            res = holistic.process(rgb)
            rgb.flags.writeable = True

            # Draw safe
            panel = frame.copy()
            try:
                if res.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        panel, res.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                        mp_styles.get_default_pose_landmarks_style()
                    )
                if res.face_landmarks:
                    mp_drawing.draw_landmarks(
                        panel, res.face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style()
                    )
                    mp_drawing.draw_landmarks(
                        panel, res.face_landmarks, mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_styles.get_default_face_mesh_contours_style()
                    )
                if res.left_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        panel, res.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style()
                    )
                if res.right_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        panel, res.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style()
                    )
            except Exception:
                pass

            # HUD anchors
            header_x, header_y = 20, 32
            bar_x, bar_y, bar_w, bar_h = 20, 60, 520, 18

            # Current features
            vec, hands_present = extract_keypoints(res)
            now = time.time()

            if paused:
                draw_status(panel, "PAUSED - press P to resume",
                            header_x, header_y, (0, 200, 255))
            else:
                if state == STATE_WAIT_START:
                    lh_on = res.left_hand_landmarks is not None
                    rh_on = res.right_hand_landmarks is not None
                    both_hands_count = both_hands_count + \
                        1 if (lh_on and rh_on) else 0

                    draw_status(panel, "SHOW BOTH HANDS to start",
                                header_x, header_y, (0, 255, 255))
                    draw_bar(panel, bar_x, bar_y, bar_w, bar_h,
                             frac=min(1.0, both_hands_count /
                                      max(1, args.start_hold_frames)),
                             color=(0, 200, 0))

                    if both_hands_count >= args.start_hold_frames:
                        both_hands_count = 0
                        armed_once = True
                        raw_frames.clear()
                        raw_times.clear()
                        hand_flags.clear()
                        seg_t0 = None
                        state = STATE_CAPTURE

                elif state == STATE_CAPTURE:
                    if seg_t0 is None:
                        seg_t0 = now
                        raw_frames.clear()
                        raw_times.clear()
                        hand_flags.clear()

                    elapsed = now - seg_t0
                    draw_status(panel, f"CAPTURE: {elapsed:.1f}s / {seg_duration:.1f}s  frames={len(raw_frames)}",
                                header_x, header_y, (0, 255, 255))
                    draw_bar(panel, bar_x, bar_y, bar_w, bar_h,
                             frac=min(1.0, elapsed / seg_duration), color=(0, 200, 0))

                    # Collect every frame (best temporal fidelity)
                    raw_frames.append(vec)
                    raw_times.append(now)
                    hand_flags.append(hands_present)

                    # When duration reached (or enough frames), predict
                    if elapsed >= seg_duration and len(raw_frames) >= int(0.6 * args.frames):
                        state = STATE_PREDICT

                elif state == STATE_PREDICT:
                    draw_status(panel, "PREDICTING...",
                                header_x, header_y, (0, 255, 0))

                    # Time-resample to EXACT T frames
                    x = resample_to_T(raw_frames, raw_times,
                                      out_T=args.frames)  # (T,1662)

                    # quality: enough hand frames in raw capture
                    ok_hands = segment_quality(
                        np.array(hand_flags, bool), args.min_hand_frames)

                    # Deltas if needed
                    x_in = add_deltas_seq(x) if args.add_deltas else x

                    # TTA set
                    variants = build_tta_set(
                        x_in,
                        do_shift=(not args.no_shift),
                        do_warp=(not args.no_warp),
                        do_hand_focus=(not args.no_hand_focus),
                        shift_vals=(-2, 0, +2),
                        warp_speeds=(0.9, 1.0, 1.1),
                        face_pose_scale=0.75
                    )
                    X = np.stack(variants, axis=0)  # (N,T,D)

                    # Predict
                    probs_acc, wsum = None, 0.0

                    def predict_with(mdl, w):
                        nonlocal probs_acc, wsum
                        if mdl is None or w <= 0:
                            return
                        p = mdl.predict(X, verbose=0)  # (N,C)
                        p = np.max(p, axis=0) if args.tta_pool == "max" else np.mean(
                            p, axis=0)
                        probs = w * p
                        probs_acc = probs if probs_acc is None else (
                            probs_acc + probs)
                        wsum += w

                    if args.mode == "ensemble":
                        predict_with(tcn_model, args.ens_w_tcn)
                        predict_with(lstm_model, args.ens_w_lstm)
                    elif args.mode == "tcn":
                        predict_with(tcn_model, 1.0)
                    else:
                        predict_with(lstm_model, 1.0)

                    if wsum <= 0:
                        probs = np.zeros((len(classes),), dtype=np.float32)
                    else:
                        probs = probs_acc / wsum

                    order = np.argsort(-probs)
                    top1, top2, top3 = int(order[0]), int(
                        order[1]), int(order[2])
                    last_top3 = [(classes[top1], float(probs[top1])),
                                 (classes[top2], float(probs[top2])),
                                 (classes[top3], float(probs[top3]))]
                    top_label = classes[top1]

                    commit_ok = ok_hands and should_commit(
                        probs, conf_hi=args.conf, conf_lo=args.conf_lo, margin=args.margin)

                    # Dump debug
                    if args.dump_dir:
                        ts = time.strftime("%Y%m%d_%H%M%S")
                        base = os.path.join(args.dump_dir, f"{ts}_{top_label}")
                        np.save(base + "_x.npy", x)
                        if args.add_deltas:
                            np.save(base + "_xd.npy", x_in)
                        meta = {
                            "top3": [(lbl, float(p)) for lbl, p in last_top3],
                            "raw_frames": len(raw_frames),
                            "raw_duration": float(raw_times[-1] - raw_times[0]) if len(raw_times) >= 2 else 0.0,
                            "hand_frames": int(np.sum(hand_flags)),
                            "ok_hands": bool(ok_hands)
                        }
                        with open(base + "_meta.json", "w", encoding="utf-8") as f:
                            json.dump(meta, f, indent=2)

                    if commit_ok:
                        typed_text += top_label
                        if not typed_text.endswith(" "):
                            typed_text += " "

                    # reset for next segment
                    cooldown_t0 = now
                    state = STATE_COOLDOWN
                    raw_frames.clear()
                    raw_times.clear()
                    hand_flags.clear()
                    seg_t0 = None

                elif state == STATE_COOLDOWN:
                    elapsed = now - cooldown_t0 if cooldown_t0 else 0.0
                    remain = max(0.0, args.segment_cooldown - elapsed)
                    draw_status(
                        panel, f"COOLDOWN: {remain:.1f}s (auto-start next)", header_x, header_y, (180, 255, 180))
                    frac = (args.segment_cooldown - remain) / \
                        max(1e-6, args.segment_cooldown)
                    draw_bar(panel, bar_x, bar_y, bar_w,
                             bar_h, frac, color=(0, 180, 255))

                    # show last top3 to guide user
                    try:
                        draw_top3(panel, bar_x + bar_w +
                                  24, bar_y - 2, last_top3)
                    except Exception:
                        pass

                    if remain <= 0.0:
                        # auto start immediately (no motion gate)
                        state = STATE_CAPTURE
                        seg_t0 = None
                        if not armed_once:
                            state = STATE_WAIT_START

            # Typed text bar
            cv2.rectangle(panel, (20, h - 90),
                          (w - 20, h - 30), (30, 30, 30), -1)
            cv2.putText(panel, f"Typed: {typed_text}", (30, h - 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

            # FPS
            t = time.time()
            fps = 1.0 / max(1e-6, (t - last_t))
            last_t = t
            fps_hist.append(fps)
            fps_avg = sum(fps_hist) / len(fps_hist) if fps_hist else 0.0
            cv2.putText(panel, f"FPS: {fps_avg:.1f}", (w - 160, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)

            # Show
            disp = cv2.resize(panel, None, fx=display_scale, fy=display_scale,
                              interpolation=cv2.INTER_AREA) if abs(display_scale - 1.0) > 1e-3 else panel
            cv2.imshow("ISL v5 (Resampled)", disp)

            # Keys
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            elif key == 32:
                if not typed_text.endswith(" "):
                    typed_text += " "
            elif key in (8, ord('b'), ord('B')):
                typed_text = typed_text[:-1] if typed_text else typed_text
            elif key in (ord('c'), ord('C')):
                typed_text = ""
            elif key in (ord('+'), ord('=')):
                display_scale = min(DISPLAY_MAX, display_scale + 0.05)
            elif key in (ord('-'), ord('_')):
                display_scale = max(DISPLAY_MIN, display_scale - 0.05)
            elif key in (ord('s'), ord('S')):
                path = save_transcript(typed_text)
                print(f"[INFO] Transcript saved: {path}")
            elif key in (ord('p'), ord('P')):
                paused = not paused

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # keep CPU threads modest on Windows
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
    main()


"""
TCN (baseline, strongest live) (BEST)
python app5.py --mode tcn `
  --tcn_weights "models/isl_v5_tcn_deltas/best.weights.h5" `
  --labels "models/isl_v5_tcn_deltas/labels.json" `
  --add_deltas `
  --frames 48 --target_fps 20 `
  --segment_cooldown 1.5 `
  --conf 0.72 --conf_lo 0.55 --margin 0.20 `
  --tta_pool max `
  --dump_dir "debug_app5_tcn"

LSTM
python app5.py --mode lstm `
  --lstm_weights "models/isl_v5_lstm_mild_aw_deltas/best.weights.h5" `
  --labels "models/isl_v5_tcn_deltas/labels.json" `
  --add_deltas `
  --frames 48 --target_fps 20 `
  --segment_cooldown 1.5 `
  --conf 0.72 --conf_lo 0.55 --margin 0.20 `
  --tta_pool max `
  --dump_dir "debug_app5_lstm"

ENSEMBLE (TCN-weighted)
python app5.py --mode ensemble `
  --lstm_weights "models/isl_v5_lstm_mild_aw_deltas/best.weights.h5" `
  --tcn_weights  "models/isl_v5_tcn_deltas/best.weights.h5" `
  --labels "models/isl_v5_tcn_deltas/labels.json" `
  --add_deltas `
  --frames 48 --target_fps 20 `
  --segment_cooldown 1.5 `
  --ens_w_tcn 0.8 --ens_w_lstm 0.2 `
  --conf 0.72 --conf_lo 0.55 --margin 0.20 `
  --tta_pool max `
  --dump_dir "debug_app5_ens"

"""


"""

python app5.py --mode lstm `
  --lstm_weights "models/isl_v5_lstm_mild_aw_deltas/best.weights.h5" `
  --labels "models/isl_v5_tcn_deltas/labels.json" `
  --add_deltas `
  --frames 48 --target_fps 20 `
  --segment_cooldown 1.5 `
  --conf 0.74 --conf_lo 0.55 --margin 0.18 `
  --tta_pool mean `
  --det_conf 0.60 --trk_conf 0.60 --model_complexity 2 `
  --dump_dir "debug_app5_lstm_bal"


python app5.py --mode ensemble `
  --lstm_weights "models/isl_v5_lstm_mild_aw_deltas/best.weights.h5" `
  --tcn_weights  "models/isl_v5_tcn_deltas/best.weights.h5" `
  --labels "models/isl_v5_tcn_deltas/labels.json" `
  --add_deltas `
  --ens_w_tcn 0.8 --ens_w_lstm 0.2 `
  --frames 48 --target_fps 20 `
  --segment_cooldown 1.5 `
  --conf 0.75 --conf_lo 0.55 --margin 0.20 `
  --tta_pool mean `
  --det_conf 0.60 --trk_conf 0.60 --model_complexity 2 `
  --dump_dir "debug_app5_ens_bal"



"""
