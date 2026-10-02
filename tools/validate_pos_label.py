#!/usr/bin/env python3
"""獨立驗證 pos_label.py / asr_screening.json / 訓練清單三處一致性。

只讀 input（來源目錄與各產出檔），不讀任何產生器的內部狀態。

檢查項目：
1. pos_label SVM_DEC 與來源目錄（top500，剝 v 前綴）推導的期望值一致
2. 編號唯一、全為整數、落在主庫範圍 1..9569
3. 每個編號對應 wav/NNNNN.wav 存在
4. 不含 neg_labels.py 的 WRONG 與 DUPLICATE 編號
5. 三處編號集合一致：SVM_DEC = json PASS = 訓練清單編號
"""
import argparse
import importlib.util
import json
import os
import sys


def load_module(path):
    spec = importlib.util.spec_from_file_location('m', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def file_id(filename):
    """檔名 → 整數編號（v 前綴剝除）。"""
    stem = os.path.splitext(filename)[0]
    if stem.startswith('v'):
        stem = stem[1:]
    return int(stem)


def expected_from_dir(source_dir):
    """從 top500 目錄獨立推導期望編號（與 build_pos_label.py 各自實作）。"""
    ids, seen = [], set()
    for name in sorted(os.listdir(source_dir)):
        if not name.lower().endswith('.wav'):
            continue
        stem = name.split('_', 1)[1] if '_' in name else name
        n = file_id(stem)
        if n not in seen:
            seen.add(n)
            ids.append(n)
    return ids


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--source-dir', default='ryza_main/top500')
    ap.add_argument('--pos-label', default='tools/pos_label.py')
    ap.add_argument('--neg-label', default='tools/neg_labels.py')
    ap.add_argument('--asr-json', default='tools/asr_screening.json')
    ap.add_argument('--train-list', default='data/ryza_train.list')
    ap.add_argument('--audio-root', default='wav')
    args = ap.parse_args()

    errors = []

    def fail(msg):
        errors.append(msg)
        print(f'FAIL: {msg}', file=sys.stderr)

    # --- 載入 ---
    try:
        svm_dec = load_module(args.pos_label).SVM_DEC
    except Exception as ex:
        print(f'FAIL: 無法載入 {args.pos_label}：{ex}', file=sys.stderr)
        sys.exit(1)

    neg = load_module(args.neg_label)
    banned = set(getattr(neg, 'WRONG', [])) | set(getattr(neg, 'DUPLICATE', []))

    with open(args.asr_json, encoding='utf-8') as f:
        recs = json.load(f)

    # --- 1. 與來源目錄推導一致（須扣除 neg_labels 禁用編號）---
    expected = [n for n in expected_from_dir(args.source_dir)
                if n not in banned]
    if svm_dec != expected:
        only_svm = sorted(set(svm_dec) - set(expected))
        only_exp = sorted(set(expected) - set(svm_dec))
        if only_svm:
            fail(f'在 SVM_DEC 但不該在：{only_svm[:5]}')
        if only_exp:
            fail(f'該在 SVM_DEC 但缺失：{only_exp[:5]}')

    # --- 2. 唯一 / 整數 / 範圍 ---
    if len(svm_dec) != len(set(svm_dec)):
        fail(f'有重複：{len(svm_dec)} 項但唯一值僅 {len(set(svm_dec))}')
    for n in svm_dec:
        if not isinstance(n, int) or isinstance(n, bool):
            fail(f'非整數：{n!r}')
            break
        if not (1 <= n <= 9569):
            fail(f'超出主庫範圍：{n}')
            break

    # --- 3. 對應音檔存在 ---
    missing = [n for n in svm_dec
               if not os.path.exists(os.path.join(args.audio_root,
                                                  f'{n:05d}.wav'))]
    if missing:
        fail(f'{args.audio_root}/ 找不到 {len(missing)} 個編號：'
             f'{missing[:5]}')

    # --- 4. 不含負樣本 ---
    hit = sorted(set(svm_dec) & banned)
    if hit:
        fail(f'含 neg_labels 禁用編號：{hit}')

    # --- 5. 三處集合一致 ---
    json_pass = {file_id(os.path.basename(r['path'])) for r in recs
                 if r.get('status') == 'PASS'}
    if set(svm_dec) != json_pass:
        only_svm = sorted(set(svm_dec) - json_pass)
        only_json = sorted(json_pass - set(svm_dec))
        if only_svm:
            fail(f'在 SVM_DEC 但 json 非 PASS：{only_svm[:5]}')
        if only_json:
            fail(f'json PASS 但不在 SVM_DEC：{only_json[:5]}')

    if os.path.exists(args.train_list):
        list_ids = []
        with open(args.train_list, encoding='utf-8') as f:
            for line in f:
                line = line.rstrip('\n')
                if not line.strip():
                    continue
                p = line.split('|')[0]
                list_ids.append(file_id(os.path.basename(p)))
        if sorted(list_ids) != sorted(svm_dec):
            fail(f'訓練清單與 SVM_DEC 不一致：'
                 f'list={len(list_ids)} svm={len(svm_dec)}')
        if len(list_ids) != len(set(list_ids)):
            fail('訓練清單有重複編號')
    else:
        print(f'警告：{args.train_list} 不存在，略過清單檢查', file=sys.stderr)

    if errors:
        sys.exit(1)
    print(f'OK: SVM_DEC {len(svm_dec)} 個編號，三處一致'
          f'（json PASS {len(json_pass)}）')


if __name__ == '__main__':
    main()
