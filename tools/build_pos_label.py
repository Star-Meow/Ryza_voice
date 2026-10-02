#!/usr/bin/env python3
"""從 top500 目錄產生正樣本編號清單 tools/pos_label.py。

規則（voice-scan 任務）：
- 來源為 ryza_main/top500/（已依 svm_dec 排序、sound_id 去重，並剔除
  3 個人工確認負樣本，故為 497 檔）。
- 檔名格式 NNNN_原檔名.wav；取 _ 後的原檔名。
- v 前綴檔以原數字編號對應 wav/ 主庫（與 select_samples.py 的
  index_key/sound_id 同一邏輯；v00086.wav 與 00086.wav 為同一份音訊）。
- 轉整數後跨檔去重，依 top500 序號排序（保持 svm_dec 高→低）。

--exclude-from 依 asr_screening.json 的非 PASS 記錄排除編號；
--prune-raw 同步將該 json 精簡為只保留清單內的 PASS 記錄。

輸出格式參照 tools/neg_labels.py，供 screen_asr.py 與訓練清單對照。

用法：
    python tools/build_pos_label.py \
        --source-dir ryza_main/top500 \
        --output tools/pos_label.py
"""
import argparse
import json
import os
import sys

from select_samples import sound_id


def file_id(filename):
    """檔名 → 整數編號（v 前綴剝除；00028.wav / v00086.wav 皆可）。"""
    try:
        return int(sound_id(filename))
    except ValueError:
        sys.exit(f'錯誤：無法解析編號「{filename}」')


def collect_from_dir(source_dir):
    """top500 目錄 → 去重整數編號清單（依序號排序，svm_dec 高→低）。"""
    if not os.path.isdir(source_dir):
        sys.exit(f'錯誤：來源目錄不存在：{source_dir}')
    names = sorted(f for f in os.listdir(source_dir)
                   if f.lower().endswith('.wav'))
    if not names:
        sys.exit(f'錯誤：{source_dir} 沒有任何 WAV')
    seen = set()
    ids = []
    for name in names:
        # 檔名格式 NNNN_原檔名.wav；取 _ 後的原檔名
        stem = name.split('_', 1)[1] if '_' in name else name
        n = file_id(stem)
        if n in seen:
            continue
        seen.add(n)
        ids.append(n)
    return ids


def load_raw(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def raw_excluded(recs):
    """非 PASS 記錄的編號集合。"""
    return {file_id(os.path.basename(r['path']))
            for r in recs if r.get('status') != 'PASS'}


def prune_raw(recs, keep):
    """精簡為只保留 keep 集合中的 PASS 記錄。"""
    return [r for r in recs
            if r.get('status') == 'PASS'
            and file_id(os.path.basename(r['path'])) in keep]


def render(ids, per_line=8):
    lines = [
        '"""正樣本清單（top500 剩餘段，v 前綴已併入原數字編號）。',
        '',
        '由 tools/build_pos_label.py 自 ryza_main/top500 產生；',
        '編號對應 wav/ 主庫檔名（00001.wav …），供 screen_asr.py、',
        'transcribe.py 與訓練清單對照。',
        '"""',
        '',
        'SVM_DEC = [',
    ]
    for i in range(0, len(ids), per_line):
        chunk = ids[i:i + per_line]
        lines.append('    ' + ', '.join(str(n) for n in chunk) + ',')
    lines.append(']')
    return '\n'.join(lines) + '\n'


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--source-dir', default='ryza_main/top500',
                    help='來源目錄（NNNN_原檔名.wav 格式）')
    ap.add_argument('--exclude-from',
                    help='asr_screening.json：非 PASS 編號自清單排除')
    ap.add_argument('--prune-raw',
                    help='asr_screening.json：同步精簡為只保留清單內 PASS')
    ap.add_argument('--output', default='tools/pos_label.py',
                    help='輸出 .py 檔路徑')
    args = ap.parse_args()

    ids = collect_from_dir(args.source_dir)
    n_source = len(ids)

    recs = None
    if args.exclude_from:
        recs = load_raw(args.exclude_from)
        excluded = raw_excluded(recs)
        ids = [n for n in ids if n not in excluded]
        print(f'來源 {n_source} 個編號，排除 {len(excluded)} 個非 PASS：'
              f'{sorted(excluded)}', file=sys.stderr)

    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8', newline='\n') as f:
        f.write(render(ids))

    if args.prune_raw:
        if recs is None:
            recs = load_raw(args.prune_raw)
        pruned = prune_raw(recs, set(ids))
        with open(args.prune_raw, 'w', encoding='utf-8') as f:
            json.dump(pruned, f, ensure_ascii=False, indent=1)
        print(f'json 精簡為 {len(pruned)} 筆 PASS → {args.prune_raw}',
              file=sys.stderr)

    print(f'{args.output}：{len(ids)} 個編號（來源 {n_source}）')


if __name__ == '__main__':
    main()
