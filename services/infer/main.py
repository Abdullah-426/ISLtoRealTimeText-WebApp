from custom_layers import wcs_fn, pres_fn, lhand_fn, rhand_fn
from custom_layers import wcs_fn, pres_fn
from custom_layers import TemporalAttentionLayer
from tensorflow import keras
import tensorflow as tf
import os
import sys
import json
from pathlib import Path
from typing import List, Dict
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Make repo root importable for custom_layers.py
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


LSTM_DIR = ROOT / 'models/isl_phrases_v3_lstm'
TCN_DIR = ROOT / 'models/isl_phrases_v3_tcn'
LETTERS_DIR = ROOT / 'models/isl_wcs_raw_aug_light_v2'

app = FastAPI(title='ISL Phrase Infer', version='0.2.0')
app.add_middleware(CORSMiddleware, allow_origins=[
                   '*'], allow_credentials=True, allow_methods=['*'], allow_headers=['*'])

# Load models
custom = {
    'TemporalAttentionLayer': TemporalAttentionLayer,
    'wcs_fn': wcs_fn,
    'pres_fn': pres_fn,
    'lhand_fn': lhand_fn,
    'rhand_fn': rhand_fn
}
lstm = keras.models.load_model(
    str(LSTM_DIR/'best.keras'), compile=False, custom_objects=custom, safe_mode=False)
tcn = keras.models.load_model(
    str(TCN_DIR/'best.keras'),  compile=False, custom_objects=custom, safe_mode=False)
letters = keras.models.load_model(
    str(LETTERS_DIR/'best.keras'), compile=False, custom_objects=custom, safe_mode=False)

with open(LSTM_DIR/'labels.json', 'r', encoding='utf-8') as f:
    labels_obj = json.load(f)
labels = labels_obj['classes'] if isinstance(
    labels_obj, dict) and 'classes' in labels_obj else labels_obj

# Load letters labels
with open(LETTERS_DIR/'labels.json', 'r', encoding='utf-8') as f:
    letters_labels_obj = json.load(f)
letters_labels = letters_labels_obj['classes'] if isinstance(
    letters_labels_obj, dict) and 'classes' in letters_labels_obj else letters_labels_obj

_, T_lstm, D_lstm = lstm.input_shape
_, T_tcn,  D_tcn = tcn.input_shape


class InferReq(BaseModel):
    x: List[List[float]]


class LettersReq(BaseModel):
    x: List[float]


class EnsReq(BaseModel):
    xl: List[List[float]]
    xt: List[List[float]]
    alpha: float = 0.5
    Tl: float = 1.0
    Tt: float = 1.0


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
        'letters_labels': letters_labels,
        'T_lstm': int(T_lstm),
        'D_lstm': int(D_lstm),
        'T_tcn': int(T_tcn),
        'D_tcn': int(D_tcn)
    }


@app.post('/infer/lstm')
def infer_lstm(req: InferReq):
    x = np.array(req.x, dtype=np.float32)
    x = adapt_sequence_dim(x, int(D_lstm))[None, ...]
    y = lstm.predict(x, verbose=0)
    probs = y[0].tolist()
    return {'probs': probs}


@app.post('/infer/tcn')
def infer_tcn(req: InferReq):
    x = np.array(req.x, dtype=np.float32)
    x = adapt_sequence_dim(x, int(D_tcn))[None, ...]
    y = tcn.predict(x, verbose=0)
    probs = y[0].tolist()
    return {'probs': probs}


@app.post('/infer/letters')
def infer_letters(req: LettersReq):
    x = np.array(req.x, dtype=np.float32)
    x = x.reshape(1, -1)  # Reshape to (1, 126) for the model
    print(f"Letters input shape: {x.shape}")
    print(f"Letters input range: [{np.min(x):.4f}, {np.max(x):.4f}]")
    print(f"Letters input mean: {np.mean(x):.4f}")
    y = letters.predict(x, verbose=0)
    probs = y[0]
    print(f"Letters output shape: {probs.shape}")
    print(f"Letters max prob: {np.max(probs):.4f}")
    print(f"Letters top 3 indices: {np.argsort(-probs)[:3]}")
    return {'probs': probs.tolist()}


@app.post('/infer/ensemble')
def infer_ensemble(req: EnsReq):
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
