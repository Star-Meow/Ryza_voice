"""ECAPA-TDNN 說話者嵌入（手動載入權重，繞過 Windows symlink 限制）。

Pipeline: wav -> Fbank(80mel) -> sentence-norm -> ECAPA_TDNN -> global-norm -> 192d
"""
import os
import numpy as np
import torch
from hyperpyyaml import load_hyperpyyaml

_DIR = os.path.dirname(os.path.abspath(__file__))
_MODEL_DIR = os.path.join(_DIR, 'spk_model')
DEVICE = 'cuda'  # GPU-only：本機 RTX 3060，禁止 CPU 回退
if torch.cuda.is_available():
    # TF32 matmul 在本機 GPU/driver 組合下會造成 CUDA 記憶體損壞
    # （跑約 20~30 個檔後出現 illegal instruction / CUBLAS_STATUS_INTERNAL_ERROR），
    # 關閉後 3200 檔連續測試穩定。
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.enabled = False
_m = None
_GLOB_MEAN = None


def get_model():
    global _m
    if _m is not None:
        return _m
    with open(os.path.join(_MODEL_DIR, 'hyperparams.yaml')) as f:
        h = load_hyperpyyaml(f, {})
    p = os.path.join(_MODEL_DIR, 'embedding_model.ckpt')
    h['embedding_model'].load_state_dict(
        torch.load(p, map_location='cpu', weights_only=False), strict=True)
    # global norm：此版 InputNormalization 的 buffer 延遲註冊，無法直接 load_state_dict；
    # std_norm=False 下只需減 glob_mean，手動套用。
    mvn = torch.load(os.path.join(_MODEL_DIR, 'mean_var_norm_emb.ckpt'),
                     map_location='cpu', weights_only=False)
    _m = h
    global _GLOB_MEAN
    _GLOB_MEAN = mvn['glob_mean'].numpy()
    for name in ['compute_features', 'mean_var_norm', 'embedding_model']:
        h[name].to(DEVICE).eval()
    return _m


@torch.no_grad()
def embed(wav_path):
    """wav -> (192,) L2-normalized embedding。"""
    h = get_model()
    import wave
    with wave.open(wav_path, 'rb') as w:
        nch = w.getnchannels()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    if nch > 1:
        x = x.reshape(-1, nch).mean(1)
    x = torch.from_numpy(x).unsqueeze(0).to(DEVICE)  # (1, T)
    feats = h['compute_features'](x)              # (1, T, 80)
    feats = h['mean_var_norm'](feats, torch.tensor([feats.shape[1]], device=DEVICE))
    emb = h['embedding_model'](feats)             # (1, T, 192)
    e = emb.squeeze(0).mean(dim=0).cpu().numpy() - _GLOB_MEAN
    return e / (np.linalg.norm(e) + 1e-9)


if __name__ == '__main__':
    pos = ['wav/00025.wav', 'wav/00030.wav', 'wav/00042.wav']
    neg = [
        'battle_voice/014_SE_BSP_079_KLAU_SHINEARROW01_a.wav',
        'battle_voice/014_SE_BSP_210_TAO_SLASH01_a.wav',
        'battle_voice/014_SE_BSP_095_LENT_SLASH01_HIT_a.wav',
        'battle_voice/014_SE_BSP_220_PAT_THRUST_RUSH_a.wav',
        'battle_voice/014_SE_BSP_043_ANCIENT_ABSORB.wav',
        'wav/00500.wav', 'wav/09000.wav',
    ]
    P = np.mean([embed(f) for f in pos], axis=0)
    P /= np.linalg.norm(P)
    print('file,cos_to_pos')
    for f in pos + neg:
        print(f'{os.path.basename(f)},{float(P @ embed(f)):+.4f}')
