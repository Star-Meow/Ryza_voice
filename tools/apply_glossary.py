#!/usr/bin/env python3
"""將 glossary.py 的 CORRECTIONS 套用至 asr_screening.json。

對每筆記錄的 text 套用專有名詞修正，並**重算 core**（與
screen_asr.py 的 core_text 同一邏輯：只保留日文/英數字元），避免
text 改了但 core 沒跟著更新，導致後續跨檔去重比對失準。

segments 逐段文字同步修正；status 不改變（PASS 維持 PASS）。

用法：
    python tools/apply_glossary.py \
        --asr-json tools/asr_screening.json \
        --dry-run              # 先預覽不改檔
"""
import argparse
import importlib.util
import json
import re
import sys

# 與 screen_asr.py 相同的字元保留規則
_KEEP = re.compile(r'[一-龠ぁ-んァ-ヶーa-zA-Z0-9]')


def core_text(s):
    return ''.join(_KEEP.findall(s))


def load_glossary(path):
    spec = importlib.util.spec_from_file_location('glossary', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--asr-json', default='tools/asr_screening.json')
    ap.add_argument('--glossary', default='tools/glossary.py')
    ap.add_argument('--dry-run', action='store_true', help='只預覽，不寫檔')
    args = ap.parse_args()

    g = load_glossary(args.glossary)
    with open(args.asr_json, encoding='utf-8') as f:
        recs = json.load(f)

    n_text = n_core = 0
    for r in recs:
        old_text = r.get('text', '')
        new_text = g.apply_corrections(old_text)
        if new_text != old_text:
            n_text += 1
            r['text'] = new_text
            if 'segments' in r:
                for seg in r['segments']:
                    seg['text'] = g.apply_corrections(seg.get('text', ''))
        old_core = r.get('core', '')
        new_core = core_text(new_text)
        if new_core != old_core:
            n_core += 1
            r['core'] = new_core

    print(f'修正 text：{n_text} 筆｜重算 core：{n_core} 筆', file=sys.stderr)

    if args.dry_run:
        for r in recs:
            if r.get('core') and False:
                pass
        return

    with open(args.asr_json, 'w', encoding='utf-8') as f:
        json.dump(recs, f, ensure_ascii=False, indent=1)
    print(f'已寫回 {args.asr_json}', file=sys.stderr)


if __name__ == '__main__':
    main()
