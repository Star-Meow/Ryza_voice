# -*- coding: utf-8 -*-
"""階段 A 驗證：PA-1～PA-13，對 train 與 eval10 兩個 exp 各跑一次。

只讀取產出與 list，不改任何檔案。UNK 來源比照 T1：以 text.japanese.g2p 取
替換前的原始音素，判定每個 UNK 是來自 # 或其他符號。
"""
import argparse
import glob
import json
import os
import random

GPT = r"H:\git\GPT-SoVITS"
RYZA = r"H:\git\Ryza_voice"
LOGS = os.path.join(GPT, "logs")

EXPS = {
    "train": {
        "dir": os.path.join(LOGS, "ryza_v4_lora"),
        "list": os.path.join(RYZA, "data", "finetune_train.list"),
        "expect": 467,
        "hubert_sample_min": 20,
        "must_include": ["07935.wav", "03166.wav"],
    },
    "eval10": {
        "dir": os.path.join(LOGS, "ryza_v4_lora_eval10"),
        "list": os.path.join(RYZA, "data", "finetune_eval10.list"),
        "expect": 10,
        "hubert_sample_min": 10,
        "must_include": [],
    },
}


def load_list_keys(path):
    keys = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f.read().strip("\n").split("\n"):
            if not line.strip():
                continue
            wav = line.split("|")[0]
            keys.append(os.path.basename(wav))
    return keys


def check(exp_key):
    import soundfile as sf
    import torch

    from text import japanese, symbols2
    from text.cleaner import clean_text

    symbols = set(symbols2.symbols)
    e = EXPS[exp_key]
    d = e["dir"]
    n = e["expect"]
    res = {}

    # PA-1 2-name2text.txt 行數與欄位數
    n2t = os.path.join(d, "2-name2text.txt")
    rows = [l for l in open(n2t, "r", encoding="utf-8").read().strip("\n").split("\n") if l.strip()]
    bad_fields = [i + 1 for i, r in enumerate(rows) if len(r.split("\t")) != 4]
    res["PA-1"] = {"pass": len(rows) == n and not bad_fields, "rows": len(rows), "expect": n, "bad_field_rows": bad_fields}

    # PA-2 4-cnhubert *.pt 數量
    huberts = sorted(glob.glob(os.path.join(d, "4-cnhubert", "*.pt")))
    res["PA-2"] = {"pass": len(huberts) == n, "count": len(huberts), "expect": n}

    # PA-3 5-wav32k 檔案數與取樣率
    wavs = sorted(glob.glob(os.path.join(d, "5-wav32k", "*")))
    random.seed(0)
    sample = wavs if len(wavs) <= 12 else random.sample(wavs, 12)
    srs = set(sf.info(w).samplerate for w in sample)
    res["PA-3"] = {"pass": len(wavs) == n and srs == {32000}, "count": len(wavs), "expect": n, "sample_rates": sorted(srs)}

    # PA-4 6-name2semantic.tsv 行數（含表頭）
    sem = os.path.join(d, "6-name2semantic.tsv")
    sem_rows = [l for l in open(sem, "r", encoding="utf-8").read().strip("\n").split("\n") if l.strip()]
    header_ok = sem_rows[0] == "item_name\tsemantic_audio" if sem_rows else False
    cr = open(sem, "rb").read().count(b"\r")
    res["PA-4"] = {"pass": len(sem_rows) == n + 1 and header_ok and cr == 0, "rows": len(sem_rows), "expect": n + 1, "header": sem_rows[0] if sem_rows else None, "cr_bytes": cr}

    # PA-5 key 集合與 list 檔名集合相等
    list_keys = set(load_list_keys(e["list"]))
    h_keys = {os.path.basename(p)[:-3] for p in huberts}
    w_keys = {os.path.basename(w) for w in wavs}
    s_keys = {r.split("\t")[0] for r in sem_rows[1:]}
    t_keys = {r.split("\t")[0] for r in rows}
    all_eq = list_keys == h_keys == w_keys == s_keys == t_keys
    res["PA-5"] = {"pass": all_eq, "list": len(list_keys), "name2text": len(t_keys), "hubert": len(h_keys), "wav32k": len(w_keys), "semantic": len(s_keys),
                   "only_in_list": sorted(list_keys - h_keys)[:5], "missing_from_list": sorted(h_keys - list_keys)[:5]}

    # PA-6 hubert 抽樣 NaN/Inf 與形狀
    checks = list(e["must_include"])
    pool = [p for p in huberts if os.path.basename(p)[:-3] not in checks]
    random.seed(1)
    checks += [os.path.basename(p)[:-3] for p in random.sample(pool, max(0, e["hubert_sample_min"] - len(checks)))]
    if exp_key == "eval10":
        checks = [os.path.basename(p)[:-3] for p in huberts]
    nan_bad, shape_bad, shapes = [], [], {}
    for k in checks:
        p = os.path.join(d, "4-cnhubert", k + ".pt")
        t = torch.load(p, map_location="cpu", weights_only=True)
        if not torch.isfinite(t).all():
            nan_bad.append(k)
        if t.dim() != 3 or t.shape[1] != 768:
            shape_bad.append((k, tuple(t.shape)))
        shapes[k] = tuple(t.shape)
    res["PA-6"] = {"pass": not nan_bad and not shape_bad, "checked": len(checks), "nan_inf": nan_bad, "bad_shape": shape_bad, "sample_shapes": {k: list(v) for k, v in list(shapes.items())[:5]}}

    # PA-7 5-wav32k 總大小
    total = sum(os.path.getsize(w) for w in wavs)
    res["PA-7"] = {"total_mb": round(total / 1024 / 1024, 1)}

    # PA-8 3-bert 存在且為空
    bert_dir = os.path.join(d, "3-bert")
    exists = os.path.isdir(bert_dir)
    entries = os.listdir(bert_dir) if exists else None
    res["PA-8"] = {"pass": exists and not entries, "exists": exists, "entries": entries}

    # PA-9 UNK 檢查（T1 方法）。門檻只計「非 # 來源」的 UNK（計畫 §9.4）
    unk_symbols = {}
    affected = set()
    # 預載 list 文字（strip 換行），避免逐行重複讀檔且混入行尾字元
    list_text = {}
    with open(e["list"], "r", encoding="utf-8") as lf:
        for line in lf.read().strip("\n").split("\n"):
            if not line.strip():
                continue
            parts = line.split("|")
            list_text[os.path.basename(parts[0])] = parts[3]
    for r in rows:
        f = r.split("\t")
        key, phones = f[0], f[1].split()
        if "UNK" not in phones:
            continue
        proc = list_text.get(key, "").replace("%", "-").replace("￥", ",")
        raw = japanese.g2p(japanese.text_normalize(proc))
        for i, ph in enumerate(phones):
            if ph != "UNK" or i >= len(raw):
                continue
            src = raw[i]
            unk_symbols[src] = unk_symbols.get(src, 0) + 1
            if src != "#":
                affected.add(key)
    non_hash = {k: v for k, v in unk_symbols.items() if k != "#"}
    res["PA-9"] = {
        "pass": sum(non_hash.values()) <= 20 and len(affected) <= 10,
        "unk_sources": unk_symbols,
        "non_hash_symbols": non_hash,
        "non_hash_total": sum(non_hash.values()),
        "non_hash_affected_lines": sorted(affected),
        "threshold_non_hash_total": 20,
        "threshold_affected_lines": 10,
        "note": "門檻只計非 # 來源；# 為官方 G2P 韻律邊界，不計入（§9.4）",
    }

    # PA-10 抽 5 行比對 name2text 音素與 T1 的 G2P 輸出
    random.seed(2)
    sample_rows = random.sample(rows, 5)
    mismatches = []
    for r in sample_rows:
        f = r.split("\t")
        proc = list_text.get(f[0], "").replace("%", "-").replace("￥", ",")
        phones, _, _ = clean_text(proc, "ja", "v4")
        if phones != f[1].split():
            mismatches.append(f[0])
    res["PA-10"] = {"pass": not mismatches, "checked": [r.split("\t")[0] for r in sample_rows], "mismatches": mismatches}

    # PA-11 第 3 步版本推斷與載入警告（由 log 檔摘錄）
    log = os.path.join(RYZA, "reports", "logs", "raw", f"{exp_key}_step3.out")
    log_txt = open(log, encoding="utf-8", errors="replace").read() if os.path.exists(log) else ""
    size = os.path.getsize(os.path.join(GPT, "GPT_SoVITS", "pretrained_models", "gsv-v4-pretrained", "s2Gv4.pth"))
    inferred = "v1" if size < 82978 * 1024 else ("v2" if size < 100 * 1024 * 1024 else ("v1" if size < 103520 * 1024 else ("v2" if size < 700 * 1024 * 1024 else "v3")))
    res["PA-11"] = {"s2G_size_bytes": size, "inferred_version": inferred, "load_msg": log_txt.strip()[:120]}

    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exps", default="train,eval10")
    args = ap.parse_args()
    out = {}
    for k in args.exps.split(","):
        out[k] = check(k.strip())
    print(json.dumps(out, ensure_ascii=False, indent=2))
    path = os.path.join(RYZA, "reports", "logs", "raw", "stage_a_pa.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("written:", path)


if __name__ == "__main__":
    main()
