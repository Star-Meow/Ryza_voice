#!/usr/bin/env python3
"""依 svm_dec 由高至低排序，產出ライザ候選語音清單，並可將選取段
複製到獨立資料夾供人工篩選。

規則（voice-scan 任務）：
- A_high 全部採用，不參與排序。
- 各排序層中檔名以 ``v`` 前綴者，歸入 A（全部採用，不參與排序）。
- 其餘檔案各依 svm_dec 由高至低排序；svm_dec 並立時依 index 列原順序。
- 跨層以 sound_id 去重，保留候選序列中首次出現者。
- 候選序列 = A 全部 + 各排序層依序串接，取前 N 段。

僅使用 Python 標準庫；--index 需含 wav_file 與 svm_dec 兩欄。

用法：
    python tools/select_samples.py \
        --index ryza_main_index.csv \
        --repo-root . \
        --adopt-dir ryza_main/A_high \
        --sort-dir ryza_main/B_likely \
        --sort-dir ryza_main/C_possible \
        --sort-dir ryza_main/B_review \
        --sort-dir ryza_main/D_flagged \
        --output ryza_train.list \
        --limit 500 \
        --copy-dir ryza_main/top500
"""
import argparse
import csv
import os
import shutil
import sys


def index_key(filename):
    """v 前綴檔與純數字檔對應到同一組 index 鍵（00086.wav / v00086.wav -> 00086.wav）。"""
    stem = os.path.splitext(os.path.basename(filename))[0]
    if stem.startswith('v'):
        stem = stem[1:]
    return f'{int(stem):05d}.wav'


def sound_id(filename):
    """去重用的 sound_id（不含 v 前綴、不含副檔名）。"""
    return index_key(filename).split('.')[0]


def load_index(path):
    """回傳 (列原順序, svm_dec)；wav_file 未出現在 index 者視為缺值。"""
    order, svm = {}, {}
    missing_value = []
    with open(path, newline='', encoding='utf-8-sig') as f:
        for pos, row in enumerate(csv.DictReader(f)):
            name = row['wav_file']
            order[name] = pos
            value = (row.get('svm_dec') or '').strip()
            try:
                svm[name] = float(value)
            except ValueError:
                missing_value.append(name)
    if missing_value:
        sys.exit(f'錯誤：index 中 svm_dec 缺值或格式異常，共 {len(missing_value)} 筆，'
                 f'例如：{missing_value[:5]}')
    return order, svm


def list_wav(directory):
    if not os.path.isdir(directory):
        sys.exit(f'錯誤：資料夾不存在：{directory}')
    return sorted(f for f in os.listdir(directory) if f.lower().endswith('.wav'))


def build_sequence(adopt_dirs, sort_dirs, order, svm):
    """組候選序列：(資料夾, 檔名) 序列；回傳序列與各層統計。"""
    seq = []
    stats = []

    # A：adopt 層全部採用（不參與排序）＋ 各排序層的 v 前綴檔
    v_files = []
    for directory in adopt_dirs:
        files = list_wav(directory)
        seq += [(directory, f) for f in files]
        stats.append((directory, 'adopt（全部採用）', len(files)))
    for directory in sort_dirs:
        vs = [f for f in list_wav(directory) if f.startswith('v')]
        v_files += [(directory, f) for f in vs]
    v_files.sort(key=lambda x: x[1])
    seq += v_files
    stats.append(('（v 前綴歸 A）', 'adopt（v 前綴）', len(v_files)))

    # 各排序層：svm_dec 高→低，並立時依 index 列原順序
    for directory in sort_dirs:
        rest = [f for f in list_wav(directory) if not f.startswith('v')]
        rest.sort(key=lambda f: (-svm[index_key(f)], order[index_key(f)]))
        seq += [(directory, f) for f in rest]
        stats.append((directory, 'sorted（svm_dec 高→低）', len(rest)))

    return seq, stats


def dedup(seq):
    seen = set()
    out, dropped = [], []
    for directory, filename in seq:
        sid = sound_id(filename)
        if sid in seen:
            dropped.append((directory, filename))
            continue
        seen.add(sid)
        out.append((directory, filename))
    return out, dropped


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--index', required=True, help='含 wav_file / svm_dec 的 CSV 路徑')
    ap.add_argument('--repo-root', required=True, help='用來計算輸出相對路徑的根目錄')
    ap.add_argument('--adopt-dir', action='append', default=[],
                    help='全部採用的資料夾（可多次指定，如 A_high）')
    ap.add_argument('--sort-dir', action='append', default=[],
                    help='參與排序的資料夾（可多次指定，順序即候選序列順序）')
    ap.add_argument('--output', required=True, help='輸出清單路徑')
    ap.add_argument('--limit', type=int, default=500, help='取前 N 段（0 = 不限）')
    ap.add_argument('--copy-dir', help='將選取段以序號前綴複製到此資料夾（供人工篩選）')
    ap.add_argument('--label', default='ryza', help='清單第二欄（角色標籤）')
    ap.add_argument('--lang', default='ja', help='清單第三欄（語言）')
    args = ap.parse_args()

    if not args.adopt_dir and not args.sort_dir:
        ap.error('至少需指定一個 --adopt-dir 或 --sort-dir')

    order, svm = load_index(args.index)
    seq, stats = build_sequence(args.adopt_dir, args.sort_dir, order, svm)

    # 缺 svm_dec 的檔案在排序鍵會 KeyError，先全部檢查
    for directory, filename in seq:
        if index_key(filename) not in svm:
            sys.exit(f'錯誤：{os.path.join(directory, filename)} 在 index 中找不到 svm_dec')

    uniq, dropped = dedup(seq)
    total = len(uniq)
    selected = uniq if args.limit <= 0 else uniq[:args.limit]

    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8', newline='\n') as f:
        for directory, filename in selected:
            rel = os.path.relpath(os.path.join(directory, filename), args.repo_root)
            f.write(f'{rel.replace(os.sep, "/")}|{args.label}|{args.lang}|\n')

    # 將選取段複製到獨立資料夾，檔名加 4 位序號前綴（0001_<原檔名>），
    # 讓檔案管理員直接依 svm_dec 順序排列，方便人工逐段篩選。
    copied = 0
    if args.copy_dir:
        os.makedirs(args.copy_dir, exist_ok=True)
        for rank, (directory, filename) in enumerate(selected, start=1):
            src = os.path.join(directory, filename)
            dst = os.path.join(args.copy_dir, f'{rank:04d}_{filename}')
            shutil.copy2(src, dst)
            copied += 1

    # ---- 回報 ----
    print('各層統計：')
    for directory, kind, count in stats:
        print(f'  {directory:32s} {kind:28s} {count:5d}')
    print(f'  {"（去重合計）":32s} {"":28s} {total:5d}')
    print(f'去重剔除：{len(dropped)} 筆')
    for directory, filename in dropped[:10]:
        print(f'  剔除 {os.path.join(directory, filename)}')
    print(f'選取：{len(selected)} 段 -> {args.output}')
    if args.copy_dir:
        print(f'複製：{copied} 段 -> {args.copy_dir}（序號前綴 0001_ 起）')
    if args.limit > 0 and total < args.limit:
        print(f'警告：不足 {args.limit} 段，缺少 {args.limit - total} 段')
    if selected:
        last = selected[-1]
        print(f'門檻（第 {len(selected)} 名）：'
              f'{os.path.join(last[0], last[1])} svm_dec={svm[index_key(last[1])]:.4f}')


if __name__ == '__main__':
    main()
