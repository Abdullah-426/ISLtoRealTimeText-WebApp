from custom_layers import wcs_fn, pres_fn, lhand_fn, rhand_fn
from custom_layers import wcs_fn, pres_fn
from custom_layers import TemporalAttentionLayer
from tensorflow import keras
import tensorflow as tf
import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from collections import deque
import time

# Make repo root importable for custom_layers.py
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


# Old phrase models (v3)
LSTM_DIR = ROOT / 'models/isl_phrases_v3_lstm'
TCN_DIR = ROOT / 'models/isl_phrases_v3_tcn'

# New phrase models (v5)
V5_LSTM_DIR = ROOT / 'models/isl_v5_lstm_mild_aw_deltas'
V5_TCN_DIR = ROOT / 'models/isl_v5_tcn_deltas'

# Letters model
LETTERS_DIR = ROOT / 'models/isl_wcs_raw_aug_light_v2'

app = FastAPI(title='ISL Phrase Infer', version='0.2.0')
app.add_middleware(CORSMiddleware, allow_origins=[
                   '*'], allow_credentials=True, allow_methods=['*'], allow_headers=['*'])

# --------- Feature layout (MATCHES COLLECTOR) ----------
POSE_LM = 33     # (x,y,z,visibility)
FACE_LM = 468    # (x,y,z)
HAND_LM = 21     # (x,y,z)

POSE_DIM = POSE_LM * 4        # 132
FACE_DIM = FACE_LM * 3        # 1404
L_HAND_DIM = HAND_LM * 3      # 63
R_HAND_DIM = HAND_LM * 3      # 63
FRAME_DIM = POSE_DIM + FACE_DIM + L_HAND_DIM + R_HAND_DIM  # 1662

# --------- TTA Configuration ----------
DEFAULT_FRAMES = 48
DEFAULT_TARGET_FPS = 20.0
DEFAULT_SEGMENT_COOLDOWN = 1.5
DEFAULT_CONF_HI = 0.72
DEFAULT_CONF_LO = 0.55
DEFAULT_MARGIN = 0.20
DEFAULT_MIN_HAND_FRAMES = 10
DEFAULT_START_HOLD_FRAMES = 8

# Load models
custom = {
    'TemporalAttentionLayer': TemporalAttentionLayer,
    'wcs_fn': wcs_fn,
    'pres_fn': pres_fn,
    'lhand_fn': lhand_fn,
    'rhand_fn': rhand_fn
}

# Load old phrase models (v3) - optional
lstm = None
tcn = None
try:
    lstm = keras.models.load_model(
        str(LSTM_DIR/'best.keras'), compile=False, custom_objects=custom, safe_mode=False)
    tcn = keras.models.load_model(
        str(TCN_DIR/'best.keras'),  compile=False, custom_objects=custom, safe_mode=False)
    print("SUCCESS: Loaded old phrase models (v3)")
except Exception as e:
    print(f"WARNING: Old phrase models (v3) not found: {e}")
    print("   Continuing with v5 models only...")

# Load new phrase models (v5)
v5_lstm = keras.models.load_model(
    str(V5_LSTM_DIR/'final_model.keras'), compile=False, custom_objects=custom, safe_mode=False)
v5_tcn = keras.models.load_model(
    str(V5_TCN_DIR/'final_model.keras'), compile=False, custom_objects=custom, safe_mode=False)
print("SUCCESS: Loaded v5 phrase models")
# Load letters model with comprehensive error handling
letters = None
model_loading_errors = []

# Method 1: Try with all custom objects
try:
    letters = keras.models.load_model(
        str(LETTERS_DIR/'best.keras'), compile=False, custom_objects=custom, safe_mode=False)
    print("SUCCESS: Letters model loaded with custom objects")
except Exception as e:
    model_loading_errors.append(f"Method 1 (custom objects): {e}")

# Method 2: Try with minimal custom objects (only the essential ones)
if letters is None:
    try:
        minimal_custom = {
            'wcs_fn': wcs_fn,
            'pres_fn': pres_fn
        }
        letters = keras.models.load_model(
            str(LETTERS_DIR/'best.keras'), compile=False, custom_objects=minimal_custom, safe_mode=False)
        print("SUCCESS: Letters model loaded with minimal custom objects")
    except Exception as e:
        model_loading_errors.append(f"Method 2 (minimal custom): {e}")

# Method 3: Try without custom objects
if letters is None:
    try:
        letters = keras.models.load_model(
            str(LETTERS_DIR/'best.keras'), compile=False, safe_mode=False)
        print("SUCCESS: Letters model loaded without custom objects")
    except Exception as e:
        model_loading_errors.append(f"Method 3 (no custom): {e}")

# Method 4: Try with legacy format
if letters is None:
    try:
        letters = keras.models.load_model(
            str(LETTERS_DIR/'model.h5'), compile=False, custom_objects=custom, safe_mode=False)
        print("SUCCESS: Letters model loaded from H5 format")
    except Exception as e:
        model_loading_errors.append(f"Method 4 (H5 format): {e}")

# If all methods failed, raise error with details
if letters is None:
    error_msg = "Failed to load letters model with all methods:\n" + \
        "\n".join(model_loading_errors)
    print(f"ERROR: {error_msg}")
    raise RuntimeError(error_msg)

# Load labels
labels = []
try:
    with open(LSTM_DIR/'labels.json', 'r', encoding='utf-8') as f:
        labels_obj = json.load(f)
    labels = labels_obj['classes'] if isinstance(
        labels_obj, dict) and 'classes' in labels_obj else labels_obj
    print("SUCCESS: Loaded old phrase labels")
except Exception as e:
    print(f"WARNING: Old phrase labels not found: {e}")
    labels = []

# Load v5 labels
with open(V5_LSTM_DIR/'labels.json', 'r', encoding='utf-8') as f:
    v5_labels_obj = json.load(f)
v5_labels = v5_labels_obj['classes'] if isinstance(
    v5_labels_obj, dict) and 'classes' in v5_labels_obj else v5_labels_obj
print("SUCCESS: Loaded v5 phrase labels")

# Load letters labels
with open(LETTERS_DIR/'labels.json', 'r', encoding='utf-8') as f:
    letters_labels_obj = json.load(f)
letters_labels = letters_labels_obj['classes'] if isinstance(
    letters_labels_obj, dict) and 'classes' in letters_labels_obj else letters_labels_obj

# Get model shapes (handle missing old models)
T_lstm, D_lstm = 48, 1662  # Default values
T_tcn, D_tcn = 48, 1662    # Default values
if lstm is not None:
    _, T_lstm, D_lstm = lstm.input_shape
if tcn is not None:
    _, T_tcn, D_tcn = tcn.input_shape

# --------- TTA Helper Functions ----------


def add_deltas_seq(x: np.ndarray) -> np.ndarray:
    """Add temporal deltas to sequence."""
    dx = np.concatenate([x[:1], x[1:] - x[:-1]], axis=0)
    return np.concatenate([x, dx], axis=-1)


def interp_time_sequence(X, t, t_new):
    """Linear interpolation with clamped edges."""
    N, D = X.shape
    t = np.asarray(t, dtype=np.float64)
    t_new = np.asarray(t_new, dtype=np.float64)
    if N == 1:
        return np.repeat(X, repeats=len(t_new), axis=0)
    s = (t_new - t[0]) / max(1e-9, (t[-1] - t[0])) * (N - 1)
    s = np.clip(s, 0.0, N - 1.0)
    i0 = np.floor(s).astype(np.int64)
    i1 = np.clip(i0 + 1, 0, N - 1)
    w = (s - i0).astype(np.float32)[:, None]
    Y = (1.0 - w) * X[i0, :] + w * X[i1, :]
    return Y.astype(np.float32)


def resample_to_T(frames, times, out_T=48):
    """Resample frames to exactly T frames evenly spaced in time."""
    X = np.stack(frames, axis=0).astype(np.float32)
    t = np.asarray(times, dtype=np.float64)
    if len(t) == 0:
        return np.zeros((out_T, X.shape[1]), dtype=np.float32)
    if len(t) == 1:
        return np.repeat(X, repeats=out_T, axis=0)
    t_new = np.linspace(t[0], t[-1], out_T, dtype=np.float64)
    return interp_time_sequence(X, t, t_new)


def shift_clip(x, delta):
    """Shift sequence with edge padding."""
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
    """Simple time-warp by interpolating at different speeds."""
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
    """Downweight face+pose dims to let hands drive more."""
    x2 = x.copy()
    x2[:, :POSE_DIM] *= face_pose_scale
    x2[:, POSE_DIM:POSE_DIM+FACE_DIM] *= face_pose_scale
    return x2


def build_tta_set(x_in, do_shift=True, do_warp=True, do_hand_focus=True,
                  shift_vals=(-2, 0, +2), warp_speeds=(0.9, 1.0, 1.1), face_pose_scale=0.75):
    """Build TTA variants for robust inference."""
    variants = []
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
    return variants


def pool_probs(P, pool="max"):
    """Pool probabilities across TTA variants."""
    return np.max(P, axis=0) if pool == "max" else np.mean(P, axis=0)


def segment_quality(hand_present_flags, min_hand_frames):
    """Check if segment has enough hand frames."""
    return int(np.sum(hand_present_flags)) >= int(min_hand_frames)


def should_commit(probs, conf_hi, conf_lo, margin):
    """Decide whether to commit prediction based on confidence rules."""
    order = np.argsort(-probs)
    p1 = float(probs[order[0]])
    p2 = float(probs[order[1]] if len(order) > 1 else 0.0)
    if p1 >= conf_hi:
        return True
    if p1 >= conf_lo and (p1 - p2) >= margin:
        return True
    return False


class InferReq(BaseModel):
    x: List[List[float]]


class LettersReq(BaseModel):
    x: List[float]  # Single 126-dim vector


class EnsReq(BaseModel):
    xl: List[List[float]]
    xt: List[List[float]]
    alpha: float = 0.5
    Tl: float = 1.0
    Tt: float = 1.0


class PhraseV5Req(BaseModel):
    frames: List[List[float]]  # Raw frames (T, 1662)
    times: List[float]  # Timestamps for each frame
    hand_flags: List[bool]  # Whether hands were present in each frame
    frames_per_segment: int = DEFAULT_FRAMES
    target_fps: float = DEFAULT_TARGET_FPS
    add_deltas: bool = True
    tta_pool: str = "max"  # "max" or "mean"
    do_shift: bool = True
    do_warp: bool = True
    do_hand_focus: bool = True
    conf_hi: float = DEFAULT_CONF_HI
    conf_lo: float = DEFAULT_CONF_LO
    margin: float = DEFAULT_MARGIN
    min_hand_frames: int = DEFAULT_MIN_HAND_FRAMES


class PhraseV5EnsembleReq(BaseModel):
    frames: List[List[float]]  # Raw frames (T, 1662)
    times: List[float]  # Timestamps for each frame
    hand_flags: List[bool]  # Whether hands were present in each frame
    frames_per_segment: int = DEFAULT_FRAMES
    target_fps: float = DEFAULT_TARGET_FPS
    add_deltas: bool = True
    tta_pool: str = "max"  # "max" or "mean"
    do_shift: bool = True
    do_warp: bool = True
    do_hand_focus: bool = True
    conf_hi: float = DEFAULT_CONF_HI
    conf_lo: float = DEFAULT_CONF_LO
    margin: float = DEFAULT_MARGIN
    min_hand_frames: int = DEFAULT_MIN_HAND_FRAMES
    ens_w_tcn: float = 0.8
    ens_w_lstm: float = 0.2


def adapt_sequence_dim(seq: np.ndarray, D: int) -> np.ndarray:
    T, d0 = seq.shape
    out = np.zeros((T, D), dtype=np.float32)
    m = min(d0, D)
    out[:, :m] = seq[:, :m]
    return out


def softmax_T(p: np.ndarray, T: float) -> np.ndarray:
    p = np.clip(p, 1e-8, 1.0)
    logit = np.log(p)
    q = np.exp(logit/float(T))
    q = q / np.sum(q, axis=1, keepdims=True)
    return q


@app.get('/')
def health():
    return {'ok': True}


@app.get('/infer/meta')
def meta():
    return {
        'labels': labels,
        'v5_labels': v5_labels,
        'letters_labels': letters_labels,
        'T_lstm': int(T_lstm),
        'D_lstm': int(D_lstm),
        'T_tcn': int(T_tcn),
        'D_tcn': int(D_tcn),
        'v5_frames_per_segment': DEFAULT_FRAMES,
        'v5_target_fps': DEFAULT_TARGET_FPS,
        'v5_segment_cooldown': DEFAULT_SEGMENT_COOLDOWN,
        'v5_conf_hi': DEFAULT_CONF_HI,
        'v5_conf_lo': DEFAULT_CONF_LO,
        'v5_margin': DEFAULT_MARGIN,
        'v5_min_hand_frames': DEFAULT_MIN_HAND_FRAMES,
        'v5_start_hold_frames': DEFAULT_START_HOLD_FRAMES,
        'frame_dim': FRAME_DIM
    }


@app.post('/infer/lstm')
def infer_lstm(req: InferReq):
    if lstm is None:
        raise HTTPException(status_code=503, detail="LSTM model not available")
    x = np.array(req.x, dtype=np.float32)
    x = adapt_sequence_dim(x, int(D_lstm))[None, ...]
    y = lstm.predict(x, verbose=0)
    probs = y[0].tolist()
    return {'probs': probs}


@app.post('/infer/tcn')
def infer_tcn(req: InferReq):
    if tcn is None:
        raise HTTPException(status_code=503, detail="TCN model not available")
    x = np.array(req.x, dtype=np.float32)
    x = adapt_sequence_dim(x, int(D_tcn))[None, ...]
    y = tcn.predict(x, verbose=0)
    probs = y[0].tolist()
    return {'probs': probs}


@app.post('/infer/letters')
def infer_letters(req: LettersReq):
    try:
        x = np.array(req.x, dtype=np.float32)
        if len(x) != 126:
            raise ValueError(f"Expected 126-dim vector, got {len(x)}")

        print(f"Backend received vec126:")
        print(f"  Length: {len(x)}")
        print(f"  Range: [{np.min(x):.4f}, {np.max(x):.4f}]")
        print(f"  First 10: {x[:10].tolist()}")
        print(f"  Last 10: {x[-10:].tolist()}")
        print(f"  Non-zero count: {np.count_nonzero(x)}")

        x = x.reshape(1, -1)  # Reshape to (1, 126) for the model
        print(f"Letters input shape: {x.shape}")

        y = letters.predict(x, verbose=0)
        probs = y[0]

        print(f"Letters output shape: {probs.shape}")
        print(f"Letters max prob: {np.max(probs):.4f}")
        print(f"Letters top 3 indices: {np.argsort(-probs)[:3]}")

        # Get top prediction
        top_idx = int(np.argmax(probs))
        top_label = letters_labels[top_idx]
        top_conf = float(probs[top_idx])

        print(f"Predicted: {top_label} (confidence: {top_conf:.4f})")

        return {
            'top_label': top_label,
            'top_conf': top_conf,
            'probs': probs.tolist(),
            'labels': letters_labels
        }
    except Exception as e:
        print(f"Letters inference error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/infer/ensemble')
def infer_ensemble(req: EnsReq):
    if lstm is None or tcn is None:
        raise HTTPException(
            status_code=503, detail="Ensemble models not available")
    xl = np.array(req.xl, dtype=np.float32)
    xt = np.array(req.xt, dtype=np.float32)
    xl = adapt_sequence_dim(xl, int(D_lstm))[None, ...]
    xt = adapt_sequence_dim(xt, int(D_tcn))[None, ...]
    y1 = lstm.predict(xl, verbose=0)
    y2 = tcn.predict(xt, verbose=0)
    p1 = softmax_T(y1, req.Tl)[0]
    p2 = softmax_T(y2, req.Tt)[0]
    p = req.alpha*p1 + (1-req.alpha)*p2
    return {'probs': p.tolist()}


# --------- V5 Phrase Model Endpoints ----------

@app.post('/infer/phrase-v5/tcn')
def infer_phrase_v5_tcn(req: PhraseV5Req):
    """Infer using v5 TCN model with TTA and quality checks."""
    try:
        # Convert input data
        frames = np.array(req.frames, dtype=np.float32)
        times = np.array(req.times, dtype=np.float64)
        hand_flags = np.array(req.hand_flags, dtype=bool)

        # Time-resample to exact T frames
        x = resample_to_T(frames, times, out_T=req.frames_per_segment)

        # Quality check: enough hand frames
        ok_hands = segment_quality(hand_flags, req.min_hand_frames)

        # Add deltas if needed
        x_in = add_deltas_seq(x) if req.add_deltas else x

        # Build TTA set
        variants = build_tta_set(
            x_in,
            do_shift=req.do_shift,
            do_warp=req.do_warp,
            do_hand_focus=req.do_hand_focus,
            shift_vals=(-2, 0, +2),
            warp_speeds=(0.9, 1.0, 1.1),
            face_pose_scale=0.75
        )
        X = np.stack(variants, axis=0)  # (N, T, D)

        # Predict
        probs = v5_tcn.predict(X, verbose=0)  # (N, C)
        probs = pool_probs(probs, pool=req.tta_pool)

        # Get top predictions
        order = np.argsort(-probs)
        top1_idx = int(order[0])
        top2_idx = int(order[1]) if len(order) > 1 else top1_idx
        top3_idx = int(order[2]) if len(order) > 2 else top2_idx

        top_label = v5_labels[top1_idx]
        top_conf = float(probs[top1_idx])

        # Commit decision
        commit_ok = ok_hands and should_commit(
            probs, conf_hi=req.conf_hi, conf_lo=req.conf_lo, margin=req.margin)

        return {
            'top_label': top_label,
            'top_conf': top_conf,
            'top3': [
                (v5_labels[top1_idx], float(probs[top1_idx])),
                (v5_labels[top2_idx], float(probs[top2_idx])),
                (v5_labels[top3_idx], float(probs[top3_idx]))
            ],
            'probs': probs.tolist(),
            'labels': v5_labels,
            'commit_ok': commit_ok,
            'ok_hands': ok_hands,
            'hand_frames': int(np.sum(hand_flags)),
            'total_frames': len(frames)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/infer/phrase-v5/lstm')
def infer_phrase_v5_lstm(req: PhraseV5Req):
    """Infer using v5 LSTM model with TTA and quality checks."""
    try:
        # Convert input data
        frames = np.array(req.frames, dtype=np.float32)
        times = np.array(req.times, dtype=np.float64)
        hand_flags = np.array(req.hand_flags, dtype=bool)

        # Time-resample to exact T frames
        x = resample_to_T(frames, times, out_T=req.frames_per_segment)

        # Quality check: enough hand frames
        ok_hands = segment_quality(hand_flags, req.min_hand_frames)

        # Add deltas if needed
        x_in = add_deltas_seq(x) if req.add_deltas else x

        # Build TTA set
        variants = build_tta_set(
            x_in,
            do_shift=req.do_shift,
            do_warp=req.do_warp,
            do_hand_focus=req.do_hand_focus,
            shift_vals=(-2, 0, +2),
            warp_speeds=(0.9, 1.0, 1.1),
            face_pose_scale=0.75
        )
        X = np.stack(variants, axis=0)  # (N, T, D)

        # Predict
        probs = v5_lstm.predict(X, verbose=0)  # (N, C)
        probs = pool_probs(probs, pool=req.tta_pool)

        # Get top predictions
        order = np.argsort(-probs)
        top1_idx = int(order[0])
        top2_idx = int(order[1]) if len(order) > 1 else top1_idx
        top3_idx = int(order[2]) if len(order) > 2 else top2_idx

        top_label = v5_labels[top1_idx]
        top_conf = float(probs[top1_idx])

        # Commit decision
        commit_ok = ok_hands and should_commit(
            probs, conf_hi=req.conf_hi, conf_lo=req.conf_lo, margin=req.margin)

        return {
            'top_label': top_label,
            'top_conf': top_conf,
            'top3': [
                (v5_labels[top1_idx], float(probs[top1_idx])),
                (v5_labels[top2_idx], float(probs[top2_idx])),
                (v5_labels[top3_idx], float(probs[top3_idx]))
            ],
            'probs': probs.tolist(),
            'labels': v5_labels,
            'commit_ok': commit_ok,
            'ok_hands': ok_hands,
            'hand_frames': int(np.sum(hand_flags)),
            'total_frames': len(frames)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/infer/phrase-v5/ensemble')
def infer_phrase_v5_ensemble(req: PhraseV5EnsembleReq):
    """Infer using v5 ensemble (TCN + LSTM) with TTA and quality checks."""
    try:
        # Convert input data
        frames = np.array(req.frames, dtype=np.float32)
        times = np.array(req.times, dtype=np.float64)
        hand_flags = np.array(req.hand_flags, dtype=bool)

        # Time-resample to exact T frames
        x = resample_to_T(frames, times, out_T=req.frames_per_segment)

        # Quality check: enough hand frames
        ok_hands = segment_quality(hand_flags, req.min_hand_frames)

        # Add deltas if needed
        x_in = add_deltas_seq(x) if req.add_deltas else x

        # Build TTA set
        variants = build_tta_set(
            x_in,
            do_shift=req.do_shift,
            do_warp=req.do_warp,
            do_hand_focus=req.do_hand_focus,
            shift_vals=(-2, 0, +2),
            warp_speeds=(0.9, 1.0, 1.1),
            face_pose_scale=0.75
        )
        X = np.stack(variants, axis=0)  # (N, T, D)

        # Predict with both models
        probs_tcn = v5_tcn.predict(X, verbose=0)  # (N, C)
        probs_lstm = v5_lstm.predict(X, verbose=0)  # (N, C)

        # Pool probabilities
        probs_tcn_pooled = pool_probs(probs_tcn, pool=req.tta_pool)
        probs_lstm_pooled = pool_probs(probs_lstm, pool=req.tta_pool)

        # Ensemble combination
        probs = req.ens_w_tcn * probs_tcn_pooled + req.ens_w_lstm * probs_lstm_pooled

        # Get top predictions
        order = np.argsort(-probs)
        top1_idx = int(order[0])
        top2_idx = int(order[1]) if len(order) > 1 else top1_idx
        top3_idx = int(order[2]) if len(order) > 2 else top2_idx

        top_label = v5_labels[top1_idx]
        top_conf = float(probs[top1_idx])

        # Commit decision
        commit_ok = ok_hands and should_commit(
            probs, conf_hi=req.conf_hi, conf_lo=req.conf_lo, margin=req.margin)

        return {
            'top_label': top_label,
            'top_conf': top_conf,
            'top3': [
                (v5_labels[top1_idx], float(probs[top1_idx])),
                (v5_labels[top2_idx], float(probs[top2_idx])),
                (v5_labels[top3_idx], float(probs[top3_idx]))
            ],
            'probs': probs.tolist(),
            'probs_tcn': probs_tcn_pooled.tolist(),
            'probs_lstm': probs_lstm_pooled.tolist(),
            'labels': v5_labels,
            'commit_ok': commit_ok,
            'ok_hands': ok_hands,
            'hand_frames': int(np.sum(hand_flags)),
            'total_frames': len(frames)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
