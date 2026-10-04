# -*- coding: utf-8 -*-
"""階段 C 推論冒煙（SM-INF）。以 Python 介面呼叫 TTS_infer_pack.TTS，不經 api_v2。

依計畫 §11.4：
  - E1 底模 baseline（S1=s1v3.ckpt、S2=s2Gv4.pth）→ 隔離推論管線本身的問題；E1 失敗即停
  - E2 冒煙權重（S1=GPT_weights_v4 產物、S2=SoVITS_weights_v4 的 LoRA）
  - 必須印出實際生效的權重路徑（TTS_Config 的 custom 陷阱，見 TTS.py:318）
  - tts_infer.yaml 前後 SHA256 比對，被改動則以版本控制還原單檔並再次比對

工作目錄必須是 <GPT>（TTS_Config 用相對路徑 "GPT_SoVITS/configs/"）。
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
PY = os.path.join(GPT, ".venv", "Scripts", "python.exe")
RAW = os.path.join(RYZA, "reports", "logs", "raw")
TTS_INFER_YAML = os.path.join("GPT_SoVITS", "configs", "tts_infer.yaml")

REF_WAV = os.path.join(RYZA, "data", "ref", "ref.wav")
# §2.1：prompt text 用 03536 原文。data\ref\ref.txt 內容已與 ref.wav 不同步，不採用。
PROMPT_TEXT = "よかった。また手伝えることがあったらいつでも言ってね。"

BASE_S1 = "GPT_SoVITS/pretrained_models/s1v3.ckpt"
BASE_S2 = "GPT_SoVITS/pretrained_models/gsv-v4-pretrained/s2Gv4.pth"
SMOKE_S1 = "GPT_weights_v4/ryza_v4_lora_smoke-e1.ckpt"
SMOKE_S2 = "SoVITS_weights_v4/ryza_v4_lora_smoke_e2_s52_l32.pth"

EVAL_LIST = os.path.join(RYZA, "data", "finetune_eval10.list")
OUT_DIR = os.path.join("logs", "ryza_v4_lora_smoke", "infer")


def install_jieba_fast_alias():
    """jieba_fast 缺失的處理（計畫 §12.3 誤以為「僅影響中文」）。

    text/chinese.py:19-23 與 text/tone_sandhi.py:17 在**模組層級** import jieba_fast，
    而 TextPreprocessor.py:13 又直接 `from text import chinese`，因此缺 jieba_fast 會讓
    整個 TTS_infer_pack 無法 import——連日文路徑也一樣，與語言無關。

    §4.2 禁止安裝套件，故不 pip install。venv 內已裝 jieba 0.42.1，而 jieba_fast 只是
    jieba 的分支（API 相同：cut / cut_for_search / posseg / setLogLevel），因此在**本行程內**
    做 sys.modules 別名。不修改任何官方檔案，也不碰磁碟上的套件。

    日文推論走 cleaner.py 的 language_module_map["ja"] = "japanese"，全程不會呼叫 jieba。
    """
    import importlib.util

    if importlib.util.find_spec("jieba_fast") is not None:
        return "jieba_fast 已存在，未做別名"
    import jieba
    import jieba.posseg

    sys.modules["jieba_fast"] = jieba
    sys.modules["jieba_fast.posseg"] = jieba.posseg
    return "jieba_fast 缺失，已在行程內別名到已安裝的 jieba %s（未安裝任何套件）" % getattr(jieba, "__version__", "?")


def point_fast_langdetect_at_bundled_model():
    """fast_langdetect 大模型路徑的處理（§4.2 禁止連網下載）。

    text/LangSegmenter/langsegmenter.py:11 把 fast_langdetect 的 cache_dir 指到
    GPT_SoVITS/pretrained_models/fast_langdetect/，該目錄不存在；fast_langdetect 只會在
    使用「套件預設 cache 目錄」時自動建立，自訂目錄則直接 FileNotFoundError，於是它
    轉而想從 fbaipublicfiles 下載 lid.176.bin——本專案禁止連網。

    但套件本身已內附 resources/lid.176.ftz（938 KB）。LangDetectConfig 支援
    custom_model_path，故把偵測器指向內附模型即可完全離線。

    必須在 `from text.LangSegmenter import LangSegmenter` **之後**才執行，
    因為 langsegmenter.py:11 在 import 時會把 _default_detector 重新指回壞路徑。

    影響面：純日文短句（如「そのことなんだけど…」）走 fast_langdetect 的短路分支，
    本來就不需要模型；較長的句子才會真正觸發偵測。
    """
    from pathlib import Path

    import fast_langdetect

    bundled = Path(fast_langdetect.infer.__file__).parent / "resources" / "lid.176.ftz"
    if not bundled.exists():
        raise FileNotFoundError("套件內附的 lid.176.ftz 不存在：%s" % bundled)
    fast_langdetect.infer._default_detector = fast_langdetect.infer.LangDetector(
        fast_langdetect.infer.LangDetectConfig(custom_model_path=str(bundled))
    )
    return "fast_langdetect 已改指向套件內附模型 %s（未下載、未複製任何檔案）" % bundled


def git_matches_head(git_path):
    """以版本控制自己的比較（尊重 core.autocrlf）判斷工作樹檔案是否等同 HEAD。

    不能直接比 SHA256：core.autocrlf=true 時，HEAD 裡存的是 LF，工作樹是 CRLF，
    位元組本就不同但內容相同（git status 乾淨）。回傳 (is_identical, head_blob_sha256)。
    """
    p = git_path.replace("\\", "/")
    rc = subprocess.call(["git", "-C", GPT, "diff", "--quiet", "HEAD", "--", p],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        blob = subprocess.check_output(["git", "-C", GPT, "show", "HEAD:%s" % p],
                                       stderr=subprocess.DEVNULL)
        head_sha = hashlib.sha256(blob).hexdigest()
    except Exception:
        head_sha = None
    return (rc == 0), head_sha


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pick_texts(n=3):
    """從 finetune_eval10.list 取長短各異的 n 行文字。"""
    rows = []
    with open(EVAL_LIST, encoding="utf-8") as f:
        for line in f.read().strip("\n").split("\n"):
            if not line.strip():
                continue
            parts = line.split("|")
            rows.append((os.path.basename(parts[0]), parts[3] if len(parts) > 3 else ""))
    rows.sort(key=lambda r: len(r[1]))
    idx = [0, len(rows) // 2, len(rows) - 1][:n]
    return [rows[i] for i in idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["base", "smoke"])
    args = ap.parse_args()

    t2s = BASE_S1 if args.mode == "base" else SMOKE_S1
    vits = BASE_S2 if args.mode == "base" else SMOKE_S2

    os.chdir(GPT)
    sys.path.insert(0, GPT)
    sys.path.insert(0, os.path.join(GPT, "GPT_SoVITS"))

    import numpy as np
    import soundfile as sf
    import torch

    shim = install_jieba_fast_alias()
    print("[SHIM] %s" % shim, flush=True)

    from TTS_infer_pack.TTS import TTS, TTS_Config

    # 必須在 LangSegmenter 被 import 之後（langsegmenter.py:11 會覆寫 _default_detector）
    shim2 = point_fast_langdetect_at_bundled_model()
    print("[SHIM] %s" % shim2, flush=True)

    os.makedirs(os.path.join(GPT, OUT_DIR), exist_ok=True)
    os.makedirs(RAW, exist_ok=True)

    yaml_abs = os.path.join(GPT, TTS_INFER_YAML)
    before = sha256(yaml_abs)
    clean_at_start, head_sha = git_matches_head(TTS_INFER_YAML)
    print("[SM-INF-4] 起跑前 on-disk SHA256 = %s" % before, flush=True)
    print("[SM-INF-4] 起跑前與 HEAD 一致（git diff --quiet）= %s" % clean_at_start, flush=True)
    print("[SM-INF-4] HEAD blob SHA256（LF）= %s" % head_sha, flush=True)
    if not clean_at_start:
        # 上一次推論若中途 crash，會停在 save_configs() 改寫後的狀態，證據會被污染。
        print("!! 警告：起跑前檔案已與 HEAD 不一致（先前推論中途中斷未還原）", flush=True)

    # ⚠️ custom 陷阱：TTS.py:318 取 configs_["custom"]，頂層傳入的鍵會被靜默忽略
    cfg = TTS_Config({
        "custom": {
            "bert_base_path": "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large",
            "cnhuhbert_base_path": "GPT_SoVITS/pretrained_models/chinese-hubert-base",
            "device": "cuda",
            "is_half": True,
            "version": "v4",
            "t2s_weights_path": t2s,
            "vits_weights_path": vits,
        }
    })

    print("\n===== SM-INF-2 custom 陷阱檢查 =====", flush=True)
    print("[EXPECTED] t2s_weights_path = %s" % t2s, flush=True)
    print("[EXPECTED] vits_weights_path = %s" % vits, flush=True)
    print("[ACTUAL  ] t2s_weights_path = %s" % cfg.t2s_weights_path, flush=True)
    print("[ACTUAL  ] vits_weights_path = %s" % cfg.vits_weights_path, flush=True)
    print("[ACTUAL  ] version=%s device=%s is_half=%s" % (cfg.version, cfg.device, cfg.is_half), flush=True)
    custom_ok = (cfg.t2s_weights_path == t2s and cfg.vits_weights_path == vits)
    print("[SM-INF-2] custom 生效 = %s" % custom_ok, flush=True)
    if not custom_ok:
        print("!! 自訂權重路徑未生效，停止（否則實際跑的不是預期模型）", flush=True)
        sys.exit(3)

    tts = TTS(cfg)
    print("\n===== TTS 載入完成，開始推論 =====", flush=True)
    print("[LOADED] t2s_weights_path = %s" % tts.configs.t2s_weights_path, flush=True)
    print("[LOADED] vits_weights_path = %s" % tts.configs.vits_weights_path, flush=True)
    print("[LOADED] model_version=%s use_vocoder=%s" % (tts.configs.version, tts.configs.use_vocoder), flush=True)

    torch.cuda.reset_peak_memory_stats()
    results = []
    for name, text in pick_texts(3):
        # TTS.run 是 generator，逐段 yield (sr, audio)；api_v2.py:441 用 next() 取第一段
        sr, audio = next(tts.run({
            "text": text,
            "text_lang": "ja",
            "ref_audio_path": REF_WAV,
            "prompt_text": PROMPT_TEXT,
            "prompt_lang": "ja",
            "top_k": 15,
            "top_p": 1,
            "temperature": 1,
            "text_split_method": "cut1",
            "batch_size": 1,
            "batch_threshold": 0.75,
            "split_bucket": True,
            "speed_factor": 1.0,
            "seed": 1234,
            "parallel_infer": False,
            "repetition_penalty": 1.35,
            "sample_steps": 32,
            "super_sampling": False,
        }))
        out_path = os.path.join(OUT_DIR, "%s_%s.wav" % (args.mode, os.path.splitext(name)[0]))
        sf.write(out_path, audio, sr)
        data, file_sr = sf.read(out_path, dtype="float32")
        results.append({
            "src": name,
            "text": text,
            "text_len": len(text),
            "output": out_path,
            "returned_sr": int(sr),
            "file_sr": int(file_sr),
            "samples": int(data.size),
            "duration_sec": round(data.size / file_sr, 3),
            "has_nan": bool(np.isnan(data).any()),
            "has_inf": bool(np.isinf(data).any()),
            "rms": float(np.sqrt(np.mean(np.square(data)))),
            "peak_abs": float(np.abs(data).max()),
        })
        r = results[-1]
        print("[OUT] %s sr=%d dur=%.3fs rms=%.6f nan=%s" %
              (r["output"], r["file_sr"], r["duration_sec"], r["rms"], r["has_nan"]), flush=True)

    vram_alloc = torch.cuda.max_memory_allocated() / (1024 * 1024)
    vram_reserved = torch.cuda.max_memory_reserved() / (1024 * 1024)

    # ---- SM-INF-4：tts_infer.yaml 前後 SHA256 ----
    after = sha256(yaml_abs)
    yaml_info = {"path": TTS_INFER_YAML,
                 "clean_at_start": clean_at_start,
                 "sha256_head_blob": head_sha,
                 "sha256_before": before,
                 "sha256_after": after,
                 "changed_by_this_run": before != after}
    if before != after:
        print("\n!! tts_infer.yaml 被改動，以版本控制還原單檔", flush=True)
        subprocess.check_call(["git", "-C", GPT, "checkout", "--", TTS_INFER_YAML])
    final_sha = sha256(yaml_abs)
    clean_final, _ = git_matches_head(TTS_INFER_YAML)
    yaml_info["sha256_final"] = final_sha
    yaml_info["restored_to_before"] = final_sha == before
    yaml_info["clean_vs_head_final"] = clean_final
    print("\n[SM-INF-4] 本輪是否改動 = %s" % yaml_info["changed_by_this_run"], flush=True)
    print("[SM-INF-4] 最終 on-disk SHA256 = %s（與起跑前相同 = %s）" % (final_sha, yaml_info["restored_to_before"]), flush=True)
    print("[SM-INF-4] 最終與 HEAD 一致（git diff --quiet）= %s" % clean_final, flush=True)

    summary = {
        "mode": args.mode,
        "time": datetime.datetime.now().isoformat(timespec="seconds"),
        "jieba_fast_shim": shim,
        "fast_langdetect_shim": shim2,
        "expected_weights": {"t2s": t2s, "vits": vits},
        "actual_weights": {"t2s": tts.configs.t2s_weights_path, "vits": tts.configs.vits_weights_path},
        "custom_trap_ok": custom_ok,
        "prompt_text": PROMPT_TEXT,
        "ref_wav": REF_WAV,
        "results": results,
        "vram_peak_alloc_mib": round(vram_alloc, 1),
        "vram_peak_reserved_mib": round(vram_reserved, 1),
        "tts_infer_yaml": yaml_info,
    }
    path = os.path.join(RAW, "infer_%s_summary.json" % args.mode)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("\nwrote %s" % path)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)