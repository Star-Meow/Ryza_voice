# -*- coding: utf-8 -*-
"""階段 D 評估：聲線相似度（ECAPA-TDNN）＋ 可懂度（Whisper CER）。

兩項指標都沿用專案既有實作，不重新發明：
- 相似度：tools/spk_model.py 的 ECAPA-TDNN pipeline；A_high 中心點的算法取自
  tools/refine_labels.py:18-28（L2-normalized embeddings 的均值再正規化）。
- CER：tools/cer_report.py 的 core_text() + levenshtein()，與 Part 1 的 1.0% 基線同源。

執行環境是 <RYZA>\.venv（torch 2.14.0+cu126、faster-whisper 1.2.1）。
需設定 HF_HUB_OFFLINE=1 讓 whisper 直接用本機快取、不連網。
"""
import glob
import json
import os
import sys

import numpy as np

RYZA = r"H:\git\Ryza_voice"
GPT = r"H:\git\GPT-SoVITS"
sys.path.insert(0, os.path.join(RYZA, "tools"))
sys.path.insert(0, RYZA)

from cer_report import core_text, levenshtein  # noqa: E402  專案既有 CER 實作

INFER_DIR = os.path.join(GPT, "logs", "ryza_v4_lora", "infer")  # 僅供參考，實際路徑由 summary 指定
THRESHOLD_RATIO = 0.90   # 生成樣本平均 >= 真實測試集平均 * 0.90
CER_LIMIT = 0.10         # 平均 CER <= 10%


def a_high_centroid():
    """A_high 中心點：92 筆 ryza_main/A_high/*.wav 的 ECAPA embedding 正規化均值。"""
    E = np.load(os.path.join(RYZA, "tools", "embeddings.npy"))
    X = E[1:] / np.linalg.norm(E[1:], axis=1, keepdims=True)   # row i-1 -> wav/{i:05d}.wav
    ids = sorted(int(os.path.basename(p).split(".")[0])
                 for p in glob.glob(os.path.join(RYZA, "ryza_main", "A_high", "*.wav")))
    ac = X[[i - 1 for i in ids]].mean(0)
    ac /= np.linalg.norm(ac)
    return X, ac, ids


def ids_from_list(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f.read().strip("\n").split("\n"):
            if line.strip():
                out.append(int(os.path.splitext(os.path.basename(line.split("|")[0]))[0]))
    return out


def cer(ref, hyp):
    r, h = core_text(ref), core_text(hyp)
    if not r:
        return None, 0, ""
    return levenshtein(r, h) / len(r), len(r), h


def main():
    import spk_model
    from faster_whisper import WhisperModel

    X, ac, a_ids = a_high_centroid()
    print("A_high 中心點：%d 筆，shape=%s，|c|=%.4f" % (len(a_ids), ac.shape, np.linalg.norm(ac)), flush=True)

    # ---- 基線：真實音訊（直接用 embeddings.npy，不需 GPU）----
    eval10_ids = ids_from_list(os.path.join(RYZA, "data", "finetune_eval10.list"))
    test24_ids = ids_from_list(os.path.join(RYZA, "data", "finetune_test.list"))
    base10 = [float(X[i - 1] @ ac) for i in eval10_ids]
    base24 = [float(X[i - 1] @ ac) for i in test24_ids]
    m10, m24 = float(np.mean(base10)), float(np.mean(base24))
    print("真實 eval10 平均 cosine = %.4f  ->  門檻 %.4f" % (m10, m10 * THRESHOLD_RATIO), flush=True)
    print("真實 test24 平均 cosine = %.4f  ->  門檻 %.4f" % (m24, m24 * THRESHOLD_RATIO), flush=True)

    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="stage_d_infer_summary.json")
    ap.add_argument("--out", default="stage_d_eval.json")
    cli = ap.parse_args()
    meta = json.load(open(os.path.join(RYZA, "reports", "logs", "raw", cli.summary), encoding="utf-8"))
    print("\n=== Whisper 載入（local_files_only，不連網）===", flush=True)
    wm = WhisperModel("large-v3", device="cuda", compute_type="float16", local_files_only=True)

    rows = []
    print("\n%-18s %-6s %8s %8s %7s" % ("file", "group", "dur_s", "cos", "CER"))
    for r in meta["results"]:
        p = os.path.join(GPT, r["path"].replace("/", os.sep))
        cos = float(spk_model.embed(p) @ ac)
        segs, _ = wm.transcribe(p, language="ja", task="transcribe", vad_filter=True, beam_size=5)
        hyp = "".join(s.text for s in segs).strip()
        c, nref, coret = cer(r["text"], hyp)
        rows.append({"file": r["file"], "group": r["group"], "mood": r["mood"],
                     "text": r["text"], "whisper": hyp, "cer": c, "ref_chars": nref,
                     "cos": cos, "dur": r["duration_sec"], "rms": r["rms"]})
        print("%-18s %-6s %8.2f %8.4f %6.2f%%" % (r["file"], r["group"], r["duration_sec"], cos, (c or 0) * 100), flush=True)

    def agg(group):
        g = [r for r in rows if r["group"] == group]
        return {"n": len(g), "mean_cos": float(np.mean([r["cos"] for r in g])),
                "mean_cer": float(np.mean([r["cer"] for r in g if r["cer"] is not None])),
                "total_ref_chars": sum(r["ref_chars"] for r in g),
                "total_edits": sum(round(r["cer"] * r["ref_chars"]) for r in g if r["cer"] is not None)}

    ga, gb = agg("a_eval10"), agg("b_new")
    both = {"n": len(rows),
            "mean_cos": float(np.mean([r["cos"] for r in rows])),
            "mean_cer": float(np.mean([r["cer"] for r in rows if r["cer"] is not None])),
            "total_ref_chars": sum(r["ref_chars"] for r in rows),
            "total_edits": sum(round(r["cer"] * r["ref_chars"]) for r in rows if r["cer"] is not None)}
    both["aggregate_cer"] = both["total_edits"] / both["total_ref_chars"]

    thr10 = m10 * THRESHOLD_RATIO
    out = {
        "a_high_centroid": {"n_files": len(a_ids), "shape": list(ac.shape)},
        "baseline": {"eval10": {"ids": eval10_ids, "per_file": base10, "mean": m10},
                     "test24": {"ids": test24_ids, "per_file": base24, "mean": m24}},
        "threshold": {"ratio": THRESHOLD_RATIO, "cos_from_eval10": thr10, "cos_from_test24": m24 * THRESHOLD_RATIO},
        "samples": rows,
        "aggregate": {"a_eval10": ga, "b_new": gb, "all20": both},
        "verdict": {
            "cos_mean_all20": both["mean_cos"],
            "cos_pass": bool(both["mean_cos"] >= thr10),
            "cos_ratio_vs_eval10": both["mean_cos"] / m10,
            "cer_mean_all20": both["mean_cer"],
            "cer_aggregate_all20": both["aggregate_cer"],
            "cer_pass": bool(both["mean_cer"] <= CER_LIMIT),
            "cer_aggregate_pass": bool(both["aggregate_cer"] <= CER_LIMIT),
        },
    }
    p = os.path.join(RYZA, "reports", "logs", "raw", cli.out)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    v = out["verdict"]
    print("\n=== 判定 ===")
    print("聲線相似度：生成 20 句平均 %.4f / 門檻 %.4f（真實 eval10 %.4f × %.0f%%）-> %s"
          % (v["cos_mean_all20"], thr10, m10, THRESHOLD_RATIO * 100, "達標" if v["cos_pass"] else "未達標"))
    print("             達成率 %.1f%%（真實 eval10 視為 100%%）" % (v["cos_ratio_vs_eval10"] * 100))
    print("可懂度 CER：逐句平均 %.2f%% / 整體加權 %.2f%% / 上限 %.0f%% -> %s"
          % (v["cer_mean_all20"] * 100, v["cer_aggregate_all20"] * 100, CER_LIMIT * 100,
             "達標" if v["cer_pass"] else "未達標"))
    print("  (a) eval10 平均 CER %.2f%% | (b) 全新台詞平均 CER %.2f%%"
          % (ga["mean_cer"] * 100, gb["mean_cer"] * 100))
    print("\nwrote %s" % p)


if __name__ == "__main__":
    main()