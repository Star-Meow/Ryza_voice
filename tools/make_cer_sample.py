#!/usr/bin/env python3
"""產生 CER 抽查清單：從 asr_screening.json 固定 seed 隨機抽 N 段。

產出 reports/cer_sample.csv，欄位：

    id, path, duration, whisper_text, reference

reference 欄留空，由人工逐段聽音檔後填入**正確的日文台詞**。
填妥後以 tools/cer_report.py 計算 CER。

抽樣使用固定 seed（預設 42），重跑可重現同一批清單。

用法：
    python tools/make_cer_sample.py
    python tools/make_cer_sample.py --n 50 --seed 42
"""
import argparse
import csv
import json
import os
import random


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--asr-json', default='tools/asr_screening.json',
                    help='ASR 結果 JSON')
    ap.add_argument('--output', default='reports/cer_sample.csv',
                    help='抽樣清單輸出路徑')
    ap.add_argument('--n', type=int, default=50, help='抽樣段數')
    ap.add_argument('--seed', type=int, default=42, help='隨機種子')
    args = ap.parse_args()

    with open(args.asr_json, encoding='utf-8') as f:
        recs = json.load(f)
    recs = [r for r in recs if r.get('status') == 'PASS'
            and (r.get('text') or '').strip()]
    if len(recs) < args.n:
        raise SystemExit(f'可用記錄僅 {len(recs)} 筆，少於 {args.n}')

    rng = random.Random(args.seed)
    sample = rng.sample(recs, args.n)

    out_dir = os.path.dirname(os.path.abspath(args.output)) or '.'
    os.makedirs(out_dir, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['id', 'path', 'duration', 'whisper_text', 'reference'])
        for i, r in enumerate(sample, 1):
            w.writerow([f'{i:02d}', r['path'], f"{r['duration']:.2f}",
                        r['text'], ''])
    print(f'{args.output}：{len(sample)} 段（seed={args.seed}）')
    print('reference 欄留空 → 人工聽音檔填入正確台詞 → '
          '執行 tools/cer_report.py')


if __name__ == '__main__':
    main()
