#!/usr/bin/env python3
"""由 ASR 結果產生 GPT-SoVITS 訓練清單。

讀 pos_label.py（SVM_DEC 編號，依 svm_dec 高→低）與
asr_screening.json（ASR 文本與過濾紀錄），只取 json 中 PASS 且
文字非空者，依編號順序輸出訓練清單：

    wav/00028.wav|ryza|ja|そっか、わかった。...

清單格式為 path|speaker|lang|文字，供 GPT-SoVITS finetune 使用。

僅使用 Python 標準庫；路徑與文字一律由參數與 json 供給，不寫死。

用法：
    python tools/transcribe.py \
        --pos-label tools/pos_label.py \
        --asr-json tools/asr_screening.json \
        --output data/ryza_train.list
"""
import argparse
import importlib.util
import json
import os
import sys


def load_svm_dec(path):
    """從 .py 檔載入 SVM_DEC 編號清單（正樣本編號，對應 wav/NNNNN.wav）。"""
    spec = importlib.util.spec_from_file_location('pos_label', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, 'SVM_DEC'):
        sys.exit(f'錯誤：{path} 缺少 SVM_DEC')
    return mod.SVM_DEC


def path_num(path):
    """wav/NNNNN.wav → 整數編號。"""
    stem = os.path.splitext(os.path.basename(path))[0]
    if stem.startswith('v'):
        stem = stem[1:]
    return int(stem)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pos-label', default='tools/pos_label.py',
                    help='正樣本編號清單（SVM_DEC）')
    ap.add_argument('--asr-json', default='tools/asr_screening.json',
                    help='ASR 結果 JSON（path / status / text）')
    ap.add_argument('--output', default='data/ryza_train.list',
                    help='輸出清單路徑')
    ap.add_argument('--audio-root', default='wav',
                    help='音檔來源目錄（用於重組路徑）')
    ap.add_argument('--label', default='ryza', help='清單第二欄（角色標籤）')
    ap.add_argument('--lang', default='ja', help='清單第三欄（語言）')
    args = ap.parse_args()

    ids = load_svm_dec(args.pos_label)
    with open(args.asr_json, encoding='utf-8') as f:
        recs = json.load(f)

    # json 以 path 為鍵值所在，編號 → 記錄
    by_num = {}
    for r in recs:
        try:
            by_num[path_num(r['path'])] = r
        except (KeyError, ValueError):
            continue

    out_dir = os.path.dirname(os.path.abspath(args.output)) or '.'
    os.makedirs(out_dir, exist_ok=True)
    n_written = 0
    n_skip = 0
    with open(args.output, 'w', encoding='utf-8', newline='\n') as f:
        for n in ids:
            r = by_num.get(n)
            if r is None:
                print(f'警告：編號 {n} 不在 {args.asr_json}，略過',
                      file=sys.stderr)
                n_skip += 1
                continue
            if r.get('status') != 'PASS' or not (r.get('text') or '').strip():
                n_skip += 1
                continue
            text = r['text'].replace('\n', ' ').strip()
            f.write(f'{args.audio_root}/{n:05d}.wav|{args.label}|{args.lang}|'
                    f'{text}\n')
            n_written += 1

    print(f'{args.output}：{n_written} 行（略過 {n_skip}）')


if __name__ == '__main__':
    main()
