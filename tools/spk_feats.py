"""MFCC 特徵擷取（紅 numpy/scipy），用於說話者相似度比對。"""
import numpy as np
from scipy.io import wavfile
from scipy.fftpack import dct


def _mel_fb(n_filters, n_fft, sr, fmin=0.0, fmax=None):
    fmax = fmax or sr / 2
    m = lambda f: 2595 * np.log10(1 + f / 700)
    mi = lambda x: 700 * (10 ** (x / 2595) - 1)
    pts = mi(np.linspace(m(fmin), m(fmax), n_filters + 2))
    bins = np.floor((n_fft + 1) * pts / sr).astype(int)
    fb = np.zeros((n_filters, n_fft // 2 + 1))
    for i in range(1, n_filters + 1):
        l, c, r = bins[i - 1], bins[i], bins[i + 1]
        if c == l: c = l + 1
        if r <= c: r = c + 1
        for k in range(l, c):
            fb[i - 1, k] = (k - l) / max(c - l, 1)
        for k in range(c, min(r, fb.shape[1])):
            fb[i - 1, k] = (r - k) / max(r - c, 1)
    return fb


_FB_CACHE = {}


def load_wav(path):
    sr, x = wavfile.read(path)
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = x.astype(np.float32) / 32768.0
    return sr, x


def vad_mask(x, sr, win=0.025, hop=0.010, thr_db=30.0, min_dur=0.25):
    """能量 VAD：回傳每個 frame 是否為語音。"""
    n = int(win * sr); hp = int(hop * sr)
    if len(x) < n:
        return np.zeros(1, dtype=bool)
    nf = 1 + (len(x) - n) // hp
    idx = np.arange(nf)[:, None] * hp + np.arange(n)[None, :]
    frames = x[idx]
    rms = np.sqrt((frames ** 2).mean(axis=1)) + 1e-10
    db = 20 * np.log10(rms)
    # 動態門檻：相對於最大能量
    keep = db > (db.max() - thr_db)
    # 型態學閉合：填補 <0.2s 空隙
    gap = int(0.2 / hop)
    out = keep.copy()
    for _ in range(3):
        # dilate
        d = np.zeros_like(out); d[:-1] |= out[1:]; d[1:] |= out[:-1]
        out |= d
    # 過短的連續段丟棄
    min_f = int(min_dur / hop)
    segs, start = [], 0
    for i in range(1, len(out) + 1):
        if i == len(out) or out[i] != out[start]:
            segs.append((start, i)); start = i
    for s, e in segs:
        if out[s] and (e - s) < min_f:
            out[s:e] = False
    return out


def mfcc(path, n_mfcc=20, n_filt=26, win=0.025, hop=0.010, n_fft=512):
    """回來 (frames, n_mfcc) 的 MFCC（含 delta），已做 per-utterance CMVN。"""
    sr, x = load_wav(path)
    n = int(win * sr); hp = int(hop * sr)
    if len(x) < n:
        x = np.pad(x, (0, n - len(x)))
    # pre-emphasis
    x = np.append(x[0], x[1:] - 0.97 * x[:-1])
    nf = 1 + (len(x) - n) // hp
    idx = np.arange(nf)[:, None] * hp + np.arange(n)[None, :]
    frames = x[idx] * np.hamming(n)
    mag = np.abs(np.fft.rfft(frames, n_fft))
    key = (n_filt, n_fft, sr)
    if key not in _FB_CACHE:
        _FB_CACHE[key] = _mel_fb(n_filt, n_fft, sr)
    fb = _FB_CACHE[key]
    energy = mag @ fb.T
    logmel = np.log(energy + 1e-10)
    c = dct(logmel, type=2, axis=1, norm='ortho')[:, :n_mfcc]
    # VAD
    mask = vad_mask(x, sr, win, hop)
    mask = mask[:c.shape[0]]
    if mask.sum() < 3:
        mask = np.ones(c.shape[0], dtype=bool)
    c = c[mask]
    # CMVN（消除通道/錄音環境差異，保留說話者動態）
    c = (c - c.mean(0)) / (c.std(0) + 1e-8)
    # delta
    d = np.gradient(c, axis=0)
    return np.hstack([c, d])


def utt_vector(path):
    """utterance -> 固定長度說話者向量（MFCC mean+std）。"""
    f = mfcc(path)
    return np.concatenate([f.mean(0), f.std(0)])
