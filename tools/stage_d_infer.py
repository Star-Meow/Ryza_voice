# -*- coding: utf-8 -*-
"""階段 D 樣本產生：(a) eval10 原台詞 10 句 ＋ (b) 全新台詞 10 句。

以 TTS_infer_pack.TTS 的 Python 介面呼叫，不經 api_v2（沿用階段 C 的 SM-INF 方法）。

與階段 C 的差異：
- 推論權重改用**正式訓練**產物（S1=ryza_v4_lora-e8.ckpt、S2=ryza_v4_lora_e8_s*_l32.pth）
- fast_langdetect 已於階段 D-1 放置正式模型檔，不再需要行程內改指向
- jieba_fast 仍無法安裝（無 wheel、需 MSVC），沿用 infer_smoke 的行程內別名，見報告 §4

工作目錄必須是 <GPT>（TTS_Config 用相對路徑）。
"""
import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys
import traceback

GPT = r"H:\git\GPT-SoVITS"
RYZA = r"H:\git\Ryza_voice"
RAW = os.path.join(RYZA, "reports", "logs", "raw")
TTS_INFER_YAML = os.path.join("GPT_SoVITS", "configs", "tts_infer.yaml")

REF_WAV = os.path.join(RYZA, "data", "ref", "ref.wav")
# data\ref\ref.txt 的內容與 ref.wav（=wav/03536.wav）不同步，此處直接用 03536 原文。
PROMPT_TEXT = "よかった。また手伝えることがあったらいつでも言ってね。"

EVAL_LIST = os.path.join(RYZA, "data", "finetune_eval10.list")
OUT_DIR = os.path.join("logs", "ryza_v4_lora", "infer")

# (b) 全新台詞 10 句：遊戲語料庫中不存在，涵蓋平敘／疑問／興奮／悲傷。
# 已以 grep data\split_manifest.csv 逐句驗證 0 命中（見報告 §5-3）。
NEW_LINES = [
    ("new_01", "平敘", "朝の光が部屋をゆっくり照らしていく。"),
    ("new_02", "平敘", "彼女は窓の外をしばらく眺めていた。"),
    ("new_03", "平敘", "兄は子供の頃から畑で花を育てることを好んでいた。"),
    ("new_04", "疑問", "この方法なら初めての人でも解決できるのでしょうか。"),
    ("new_05", "疑問", "森の奥にあるあの小屋には、行ったことがありますか。"),
    ("new_06", "興奮", "やっと答えが見えてきた！　うれしい！"),
    ("new_07", "興奮", "大成功だ！　これはすごい。"),
    ("new_08", "悲傷", "言いたいことは多いのに言葉にならない。"),
    ("new_09", "悲傷", "雨の音が聞こえる。"),
    ("new_10", "平敘", "でも、明日が来るのが楽しみだ。"),
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_clean(path):
    return subprocess.call(["git", "-C", GPT, "diff", "--quiet", "HEAD", "--", path.replace("\\", "/")],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0


def pick_newest(pattern_dir, prefix):
    """挑出 <dir>/<prefix>* 中「編號最大」的檔案。

    純字串排序會出錯：'e9' > 'e16'，十個以上 epoch 時會挑到 e9 而非 e16。
    檔名格式：ryza_v4_lora-e16.ckpt / ryza_v4_lora_e16_s1904_l32.pth
    """
    import re
    d = os.path.join(GPT, pattern_dir)
    cands = [f for f in os.listdir(d) if f.startswith(prefix)]
    if not cands:
        raise FileNotFoundError("%s 下找不到 %s*" % (d, prefix))
    keyed = []
    for f in cands:
        m = re.search(r"(\d+)(?=[._]|$)", f[len(prefix):])
        keyed.append((int(m.group(1)) if m else -1, f))
    best = max(k for k, _ in keyed)
    chosen = sorted(f for k, f in keyed if k == best)[-1]
    return os.path.join(pattern_dir, chosen).replace("\\", "/")


def load_eval10():
    rows = []
    with open(EVAL_LIST, encoding="utf-8") as f:
        for line in f.read().strip("\n").split("\n"):
            if not line.strip():
                continue
            parts = line.split("|")
            rows.append((os.path.splitext(os.path.basename(parts[0]))[0], parts[3]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--s1-prefix", default="ryza_v4_lora-e", help="GPT_weights_v4 中的權重檔名前綴")
    ap.add_argument("--s2-prefix", default="ryza_v4_lora_e8", help="SoVITS_weights_v4 中的權重檔名前綴")
    ap.add_argument("--summary-name", default="stage_d_infer_summary.json")
    args = ap.parse_args()

    os.chdir(GPT)
    sys.path.insert(0, GPT)
    sys.path.insert(0, os.path.join(GPT, "GPT_SoVITS"))
    sys.path.insert(0, os.path.join(RYZA, "tools"))

    import numpy as np
    import soundfile as sf
    import torch

    # jieba_fast 仍無法在此環境安裝（無 wheel、需 MSVC），沿用階段 C 已驗證的行程內別名。
    from infer_smoke import install_jieba_fast_alias

    shim = install_jieba_fast_alias()
    print("[SHIM] %s" % shim, flush=True)

    from TTS_infer_pack.TTS import TTS, TTS_Config

    t2s = pick_newest("GPT_weights_v4", args.s1_prefix)
    vits = pick_newest("SoVITS_weights_v4", args.s2_prefix)
    print("[WEIGHTS] S1 = %s" % t2s, flush=True)
    print("[WEIGHTS] S2 = %s" % vits, flush=True)

    os.makedirs(os.path.join(GPT, args.out_dir), exist_ok=True)
    os.makedirs(RAW, exist_ok=True)

    yaml_abs = os.path.join(GPT, TTS_INFER_YAML)
    before, clean_at_start = sha256(yaml_abs), git_clean(TTS_INFER_YAML)
    print("[YAML] 起跑前 SHA256=%s 與 HEAD 一致=%s" % (before, clean_at_start), flush=True)

    cfg = TTS_Config({"custom": {
        "bert_base_path": "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large",
        "cnhuhbert_base_path": "GPT_SoVITS/pretrained_models/chinese-hubert-base",
        "device": "cuda", "is_half": True, "version": "v4",
        "t2s_weights_path": t2s, "vits_weights_path": vits,
    }})
    print("[ACTUAL] t2s=%s vits=%s version=%s device=%s is_half=%s"
          % (cfg.t2s_weights_path, cfg.vits_weights_path, cfg.version, cfg.device, cfg.is_half), flush=True)
    if cfg.t2s_weights_path != t2s or cfg.vits_weights_path != vits:
        print("!! custom 未生效，停止", flush=True)
        sys.exit(3)

    tts = TTS(cfg)
    print("[LOADED] t2s=%s" % tts.configs.t2s_weights_path, flush=True)
    print("[LOADED] vits=%s  use_vocoder=%s" % (tts.configs.vits_weights_path, tts.configs.use_vocoder), flush=True)

    torch.cuda.reset_peak_memory_stats()
    jobs = [("a_eval10", "eval10_%s.wav" % k, t) for k, t in load_eval10()]
    jobs += [("b_new", "%s.wav" % k, t) for k, _emo, t in NEW_LINES]
    mood = dict((k, e) for k, e, _t in NEW_LINES)

    results = []
    for group, fname, text in jobs:
        sr, audio = next(tts.run({
            "text": text, "text_lang": "ja",
            "ref_audio_path": REF_WAV, "prompt_text": PROMPT_TEXT, "prompt_lang": "ja",
            "top_k": 15, "top_p": 1, "temperature": 1, "text_split_method": "cut1",
            "batch_size": 1, "batch_threshold": 0.75, "split_bucket": True,
            "speed_factor": 1.0, "seed": 1234, "parallel_infer": False,
            "repetition_penalty": 1.35, "sample_steps": 32, "super_sampling": False,
        }))
        out_path = os.path.join(args.out_dir, fname)
        sf.write(out_path, audio, sr)
        data, file_sr = sf.read(out_path, dtype="float32")
        r = {"group": group, "file": fname, "path": out_path, "text": text,
             "mood": mood.get(os.path.splitext(fname)[0]),
             "sr": int(file_sr), "samples": int(data.size),
             "duration_sec": round(data.size / file_sr, 3),
             "rms": float(np.sqrt(np.mean(np.square(data)))),
             "peak_abs": float(np.abs(data).max()),
             "has_nan": bool(np.isnan(data).any()), "has_inf": bool(np.isinf(data).any())}
        results.append(r)
        print("[OUT] %-6s %-18s sr=%d dur=%6.3fs rms=%.6f nan=%s"
              % (r["group"], r["file"], r["sr"], r["duration_sec"], r["rms"], r["has_nan"]), flush=True)

    vram_alloc = torch.cuda.max_memory_allocated() / (1024 * 1024)
    vram_reserved = torch.cuda.max_memory_reserved() / (1024 * 1024)

    if before != sha256(yaml_abs):
        print("!! tts_infer.yaml 被改動，還原單檔", flush=True)
        subprocess.check_call(["git", "-C", GPT, "checkout", "--", TTS_INFER_YAML])
    final_sha, clean_final = sha256(yaml_abs), git_clean(TTS_INFER_YAML)
    print("[YAML] 最終 SHA256=%s 與 HEAD 一致=%s" % (final_sha, clean_final), flush=True)

    summary = {"time": datetime.datetime.now().isoformat(timespec="seconds"),
               "weights": {"s1": t2s, "s2": vits},
               "prompt_text": PROMPT_TEXT, "ref_wav": REF_WAV,
               "jieba_fast_shim": shim,
               "tts_infer_yaml": {"clean_at_start": clean_at_start, "sha256_before": before,
                                  "sha256_final": final_sha, "clean_vs_head_final": clean_final},
               "vram_peak_alloc_mib": round(vram_alloc, 1),
               "vram_peak_reserved_mib": round(vram_reserved, 1),
               "n_samples": len(results), "results": results}
    path = os.path.join(RAW, args.summary_name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("\nwrote %s  (%d samples)" % (path, len(results)))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)