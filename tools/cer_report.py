#!/usr/bin/env python3
"""計算 CER 抽查結果：讀 cer_sample.csv，對已填 reference 的段算 CER。

CER（字元錯誤率）= Levenshtein(reference, whisper) / len(reference)

- 比對前兩邊都過 core 正規化（去標點、只保留日文/英數字元），
  與 screen_asr.py 的 core 欄同一邏輯，避免標點差異干擾。
- 只計算 reference 已填寫的段；未填的段列為「待校對」。
- 產出 reports/cer_report.md：每段 CER 表、平均 CER、最差 N 段。

用法：
    python tools/cer_report.py
    python tools/cer_report.py --input reports/cer_sample.csv
"""
import argparse
import csv
import os
import re
import sys

_KEEP = re.compile(r'[一-龠ぁ-んァ-ヶーa-zA-Z0-9]')


def core_text(s):
    return ''.join(_KEEP.findall(s))


def levenshtein(a, b):
    """編輯距離（標準 DP，空間壓縮為兩行）。"""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--input', default='reports/cer_sample.csv',
                    help='抽樣清單（含 reference 欄）')
    ap.add_argument('--output', default='reports/cer_report.md',
                    help='CER 報告輸出路徑')
    ap.add_argument('--worst', type=int, default=5, help='列出最差 N 段')
    args = ap.parse_args()

    rows = []
    with open(args.input, encoding='utf-8') as f:
        for r in csv.DictReader(f):
            rows.append(r)

    scored, pending = [], []
    for r in rows:
        ref = (r.get('reference') or '').strip()
        if not ref:
            # 填寫約定：留空 = 無異常，reference 視同 whisper_text
            ref = (r.get('whisper_text') or '').strip()
            if not ref:
                pending.append(r)
                continue
        hyp = core_text(r.get('whisper_text', ''))
        ref_c = core_text(ref)
        dist = levenshtein(ref_c, hyp)
        cer = dist / len(ref_c) if ref_c else 0.0
        r['_ref_c'] = ref_c
        r['_hyp'] = hyp
        r['_dist'] = dist
        r['_cer'] = cer
        # 有無異常：reference 與 whisper_text 完全相同（含標點）視為無異常
        r['_corrected'] = ref != (r.get('whisper_text') or '').strip()
        scored.append(r)

    n_fixed = sum(1 for r in scored if r['_corrected'])
    n_ok = sum(1 for r in scored if not r['_corrected'])
    n_total = len(rows)
    out_dir = os.path.dirname(os.path.abspath(args.output)) or '.'
    os.makedirs(out_dir, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8', newline='\n') as f:
        w = f.write
        w('# CER 抽查報告\n\n')
        w(f'- 抽樣：`{args.input}`（{n_total} 段）\n')
        w(f'- 已校對：**{len(scored)}** 段（修正 {n_fixed} 段、無異常 {n_ok} 段）'
          f'｜未校對：{len(pending)} 段\n')
        w('- 填寫約定：reference 留空 = 無異常（視同 whisper_text）；'
          '有填 = 修正後的正確文字\n\n')

        if scored:
            mean = sum(r['_cer'] for r in scored) / len(scored)
            w(f'## 摘要\n\n')
            w(f'- **平均 CER：{mean:.1%}**\n')
            w(f'- 字元錯誤總數：{sum(r["_dist"] for r in scored)} / '
              f'{sum(len(r["_ref_c"]) for r in scored)} 字元\n')
            zero = sum(1 for r in scored if r['_cer'] == 0)
            w(f'- 完全正確（CER=0）：{zero} 段\n\n')

            w(f'## 有錯誤的段（{n_fixed}）\n\n')
            fixed = [r for r in scored if r['_corrected']]
            if fixed:
                w('| # | 音檔 | 參考（校對） | Whisper 輸出 | 編輯距離 | CER |\n')
                w('|---|---|---|---|---|---|\n')
                for r in fixed:
                    ref_s = r['_ref_c'][:18].replace('|', '\\|')
                    hyp_s = r['_hyp'][:18].replace('|', '\\|')
                    w(f"| {r['id']} | `{r['path']}` | {ref_s} | {hyp_s} "
                      f"| {r['_dist']} | **{r['_cer']:.1%}** |\n")
                w('\n')
            else:
                w('（無修正段）\n\n')

            worst = sorted(scored, key=lambda r: -r['_cer'])[:args.worst]
            worst = [r for r in worst if r['_cer'] > 0]
            if worst:
                w(f'## 最差 {len(worst)} 段\n\n')
                for r in worst:
                    w(f"### `{r['path']}`（CER {r['_cer']:.1%}）\n")
                    w(f"- 參考：{r['_ref_c']}\n")
                    w(f"- Whisper：{r['_hyp']}\n")
                    w(f"- 編輯距離 {r['_dist']} / 參考 {len(r['_ref_c'])} 字元\n\n")

        if pending:
            w(f'## 待校對（{len(pending)} 段）\n\n')
            w('reference 欄尚未填寫：\n\n')
            for r in pending:
                w(f"- {r['id']} `{r['path']}`\n")
            w('\n')

    if scored:
        mean = sum(r['_cer'] for r in scored) / len(scored)
        fixed = sum(1 for r in scored if r['_corrected'])
        print(f'{args.output}：{len(scored)} 段已算（修正 {fixed} 段、'
              f'無異常 {len(scored) - fixed} 段），平均 CER {mean:.1%}，'
              f'{len(pending)} 段未校對')
    else:
        print(f'{args.output}：尚無已校對段落（{len(pending)} 段未校對）')


if __name__ == '__main__':
    main()
