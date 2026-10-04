# -*- coding: utf-8 -*-
"""T1：UNK 來源分析（唯讀，只輸出統計到 stdout，不寫入 logs\\）。

呼叫方式完全比照 1-get-text.py:92：
    clean_text(text.replace("%", "-").replace("￥", ","), lan, version)
語言 ja、version v4。clean_text 內部會把不在 symbols2 的音素替換成 "UNK"，
因此另呼叫 text.japanese.g2p 取「替換前」的原始音素，以還原每個 UNK 的來源符號。
"""
import argparse
import os
from collections import Counter

RYZA = r"H:\git\Ryza_voice"


def load_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [ln for ln in f.read().strip("\n").split("\n") if ln.strip()]


def analyze(list_path, label):
    from text import japanese, symbols2
    from text.cleaner import clean_text

    symbols = set(symbols2.symbols)
    lang_map = {"ja": "ja", "JA": "ja", "JP": "ja", "jp": "ja"}
    lines = load_lines(list_path)

    total = 0
    unk_symbol_counter = Counter()       # 被 UNK 的原始符號 → 次數
    unk_source_kind = Counter()          # "prosody_insert"（不在文字中）/"in_text"（文字中的符號）
    per_line_unk = []                    # (wav_name, unk_count, phone_count, ratio)
    short_lines = []                     # phone_count < 3
    empty_lines = []                     # phone_count == 0
    pct_lines = 0                        # 含 % 或 ￥ 的行（訓練前處理會替換，推論端不會）
    mismatch = []                        # clean_text 與本地重算不一致的行
    bad_format = []

    for idx, line in enumerate(lines, 1):
        parts = line.split("|")
        if len(parts) != 4:
            bad_format.append((idx, line[:60]))
            continue
        wav, spk, lang, text = parts
        lang = lang_map.get(lang, lang)
        if "%" in text or "￥" in text:
            pct_lines += 1
        proc = text.replace("%", "-").replace("￥", ",")
        if lang != "ja":
            # 本資料集預期全為 ja；非 ja 記錄下來
            bad_format.append((idx, f"lang={lang}: {line[:50]}"))
            continue
        phones, word2ph, norm_text = clean_text(proc, "ja", "v4")
        # 重算原始音素以還原 UNK 來源
        raw = japanese.g2p(japanese.text_normalize(proc))
        expected = ["UNK" if ph not in symbols else ph for ph in raw]
        if phones != expected or len(phones) != len(raw):
            mismatch.append((idx, wav))
        total += 1
        n_ph = len(phones)
        n_unk = sum(1 for p in phones if p == "UNK")
        per_line_unk.append((os.path.basename(wav), n_unk, n_ph, (n_unk / n_ph if n_ph else 0.0)))
        if n_ph == 0:
            empty_lines.append((idx, os.path.basename(wav), text[:40]))
        elif n_ph < 3:
            short_lines.append((idx, os.path.basename(wav), n_ph, text[:40]))
        for i, p in enumerate(phones):
            if p == "UNK" and i < len(raw):
                src = raw[i]
                unk_symbol_counter[src] += 1
                if src in proc:
                    unk_source_kind["in_text"] += 1
                else:
                    unk_source_kind["prosody_insert"] += 1

    lines_with_unk = [x for x in per_line_unk if x[1] > 0]
    ratios = sorted(x[3] for x in per_line_unk)
    unks = sorted(x[1] for x in per_line_unk)
    ph_counts = sorted(x[2] for x in per_line_unk)

    def median(v):
        n = len(v)
        return v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2

    print(f"================ {label} ================")
    print(f"list: {list_path}")
    print(f"總行數(解析成功): {total}; 格式異常或非 ja: {len(bad_format)}")
    if bad_format:
        for b in bad_format[:10]:
            print(f"  異常: {b}")
    print(f"含 % 或 ￥ 的行數（訓練端會替換、推論端不替換）: {pct_lines}")
    print(f"音素數分布: min={ph_counts[0] if ph_counts else 0}, median={median(ph_counts) if ph_counts else 0}, max={ph_counts[-1] if ph_counts else 0}")
    print(f"含 UNK 的行數: {len(lines_with_unk)} / {total}")
    print(f"每行 UNK 數分布: min={unks[0] if unks else 0}, median={median(unks) if unks else 0}, max={unks[-1] if unks else 0}")
    print(f"每行 UNK 比例分布: min={ratios[0]:.4f}, median={median(ratios):.4f}, max={ratios[-1]:.4f}" if ratios else "無比例資料")
    print(f"UNK 來源符號統計（被替換的原始音素）: {dict(unk_symbol_counter)}")
    print(f"UN K 來源類別: {dict(unk_source_kind)}")
    print(f"clean_text 與本地重算不一致的行: {len(mismatch)} {mismatch[:5]}")
    if empty_lines:
        print(f"空輸出行（0 音素）: {len(empty_lines)}")
        for e in empty_lines[:20]:
            print(f"  {e}")
    else:
        print("空輸出行（0 音素）: 0")
    if short_lines:
        print(f"異常短行（<3 音素）: {len(short_lines)}")
        for s in short_lines[:20]:
            print(f"  line={s[0]} {s[1]} phones={s[2]} text={s[3]!r}")
    else:
        print("異常短行（<3 音素）: 0")
    # UNK 最多的前 10 行明細
    top = sorted(per_line_unk, key=lambda x: -x[1])[:10]
    print("UNK 數最多的前 10 行:")
    for t in top:
        print(f"  {t[0]} unk={t[1]} phones={t[2]} ratio={t[3]:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=os.path.join(RYZA, "data", "finetune_train.list"))
    ap.add_argument("--eval10", default=os.path.join(RYZA, "data", "finetune_eval10.list"))
    args = ap.parse_args()
    analyze(args.train, "訓練集 finetune_train.list")
    print()
    analyze(args.eval10, "eval10 finetune_eval10.list")


if __name__ == "__main__":
    main()
