#!/usr/bin/env python3
"""Part 2 資料切分：自 data/ryza_train.list 分層抽樣出 train / test / eval10。

凍結規格（issue Part 2 ＋夜跑任務定義）：
- 輸入 data/ryza_train.list（491 行，四欄 音檔路徑|說話人|語言|文字），唯讀不修改。
- 句尾分類（取文字最後非空白字元）：
    Q    疑問     '?' / '？'
    EXCL 興奮／感嘆 '!' / '！'
    ELL  猶豫／餘韻 '…' 或結尾 3+ 連續 '.'
    DECL 平敘     其餘（含 '。'、'.'、語尾助詞）
- 各類別內依時長三等分桶，test 配額依桶大小比例分配（最大餘數法），
  桶內以 seed=42 隨機抽樣；train = 其餘（保持原清單順序）。
- test 配額：Q 4 / EXCL 2 / ELL 3 / DECL 15（合計 24）。
- eval10 自 test 再分層（同 seed 42）：Q 2 / EXCL 1 / ELL 1 / DECL 6（合計 10）。
- 參考音訊自 train 選：4–8 秒、無削波、清單最早列；複製到 data/ref/。
- 全部門禁在記憶體內驗證通過後才寫任何檔案；任一失敗即 abort 不寫檔。

本腳本只做切分：不訓練、不跑預處理、不改官方腳本；wav/ 只讀（參考音訊用複製）。
路徑一律由參數傳入，不寫死本機絕對路徑。

用法：
    .venv/Scripts/python.exe tools/split_train_test.py \
        --input data/ryza_train.list \
        --train-out data/finetune_train.list \
        --test-out data/finetune_test.list \
        --eval-out data/finetune_eval10.list \
        --manifest data/split_manifest.csv \
        --ref-dir data/ref \
        --report reports/split_report.md \
        --seed 42
"""
import argparse
import csv
import hashlib
import os
import random
import sys
import wave

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CLASS_ORDER = ['DECL', 'EXCL', 'ELL', 'Q']
CLASS_LABEL = {
    'Q': '疑問',
    'EXCL': '興奮/感嘆',
    'ELL': '猶豫/餘韻',
    'DECL': '平敘',
}
TEST_QUOTA = {'Q': 4, 'EXCL': 2, 'ELL': 3, 'DECL': 15}
EVAL_QUOTA = {'Q': 2, 'EXCL': 1, 'ELL': 1, 'DECL': 6}
N_BUCKETS = 3
EXPECTED_TOTAL = 491
CLIP_FULLSCALE = 32767  # int16 滿刻度；peak 達此值視為削波


def classify(text):
    """依句尾標點分類；空字串回 None。"""
    t = text.rstrip()
    if not t:
        return None
    last = t[-1]
    if last in '?？':
        return 'Q'
    if last in '!！':
        return 'EXCL'
    if last == '…' or t.endswith('...'):
        return 'ELL'
    return 'DECL'


def parse_list(path):
    """讀四欄清單；回傳 (rows, errors)。rows 保留原順序。"""
    rows, errors = [], []
    with open(path, encoding='utf-8') as f:
        for i, raw in enumerate(f, 1):
            line = raw.rstrip('\n').rstrip('\r')
            if not line.strip():
                continue
            parts = line.split('|')
            if len(parts) != 4:
                errors.append((i, '欄數不為 4', line))
                continue
            audio, speaker, lang, text = (p.strip() for p in parts)
            rows.append({
                'line': i,
                'audio': audio,
                'speaker': speaker,
                'lang': lang,
                'text': text,
            })
    return rows, errors


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def md5_of(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def wav_duration(path):
    with wave.open(path, 'rb') as w:
        return w.getnframes() / w.getframerate()


def wav_peak(path):
    """int16 峰值（絕對值最大）；用於削波判定。"""
    with wave.open(path, 'rb') as w:
        n = w.getnframes()
        raw = w.readframes(n)
    if not raw:
        return 0
    x = np.frombuffer(raw, dtype='<i2')
    return int(np.abs(x).max())


def duration_buckets(items):
    """時長排序後三等分；回傳每筆的 bucket index（0=最短, 2=最長）。"""
    order = sorted(range(len(items)),
                   key=lambda i: (items[i]['duration'], items[i]['utt_id']))
    bucket_of = [0] * len(items)
    for pos, idx in enumerate(order):
        bucket_of[idx] = (pos * N_BUCKETS) // len(items)
    return bucket_of


def allocate(quota, sizes):
    """依桶大小比例分配配額（最大餘數法）；不可超過各桶大小。"""
    total = sum(sizes)
    if quota > total:
        raise ValueError(f'配額 {quota} 超過可用數 {total}')
    exact = [quota * s / total for s in sizes]
    base = [int(e) for e in exact]
    rem = quota - sum(base)
    order = sorted(range(len(sizes)),
                   key=lambda i: (exact[i] - base[i], -sizes[i]),
                   reverse=True)
    for i in range(rem):
        base[order[i]] += 1
    return base


def sample_from_bucket(items, k, rng):
    """桶內隨機抽 k 筆；items 已按 (duration, utt_id) 排序。"""
    if k == 0:
        return []
    if k > len(items):
        raise ValueError(f'抽樣數 {k} 超過桶大小 {len(items)}')
    idxs = sorted(rng.sample(range(len(items)), k))
    return [items[i] for i in idxs]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--input', default='data/ryza_train.list')
    ap.add_argument('--train-out', default='data/finetune_train.list')
    ap.add_argument('--test-out', default='data/finetune_test.list')
    ap.add_argument('--eval-out', default='data/finetune_eval10.list')
    ap.add_argument('--manifest', default='data/split_manifest.csv')
    ap.add_argument('--ref-dir', default='data/ref')
    ap.add_argument('--report', default='reports/split_report.md')
    ap.add_argument('--audio-root', default=None,
                    help='音檔根目錄（預設為專案根目錄）')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--expected-total', type=int, default=EXPECTED_TOTAL)
    args = ap.parse_args()

    audio_root = args.audio_root or ROOT
    in_path = os.path.join(ROOT, args.input) if not os.path.isabs(args.input) else args.input
    input_sha = sha256_of(in_path)

    # ---- 1. 讀取輸入 ----
    rows, errors = parse_list(in_path)
    fail = []  # 門禁失敗原因

    if errors:
        for ln, why, line in errors:
            fail.append(f'輸入第 {ln} 行：{why}（{line[:40]}）')
    print(f'[1/7] 輸入 {len(rows)} 行（期望 {args.expected_total}），'
          f'格式錯誤 {len(errors)} 行')
    if len(rows) != args.expected_total:
        fail.append(f'輸入行數 {len(rows)} ≠ {args.expected_total}（硬性失敗）')

    # ---- 2. 路徑有效性与基本欄位檢查 ----
    for r in rows:
        r['abs'] = os.path.abspath(os.path.join(audio_root, r['audio']))
        if not os.path.exists(r['abs']):
            fail.append(f"路徑不存在：{r['audio']}（第 {r['line']} 行）")
        if r['speaker'] != 'ryza':
            fail.append(f"speaker 不為 ryza：{r['speaker']}（第 {r['line']} 行）")
        if r['lang'] != 'ja':
            fail.append(f"lang 不為 ja：{r['lang']}（第 {r['line']} 行）")
        if not r['text']:
            fail.append(f"文字為空（第 {r['line']} 行）")
        if '|' in r['text']:
            fail.append(f"文字含 '|'（第 {r['line']} 行）")
        r['utt_id'] = os.path.splitext(os.path.basename(r['audio']))[0]
    paths = [r['audio'] for r in rows]
    if len(set(paths)) != len(paths):
        dupes = {p for p in paths if paths.count(p) > 1}
        fail.append(f'輸入有重複路徑：{sorted(dupes)[:5]}')
    print(f'[2/7] 路徑與欄位檢查完成，累計失敗 {len(fail)} 項')

    # ---- 3. 分類＋時長＋分桶 ----
    for r in rows:
        r['cls'] = classify(r['text'])
        if r['cls'] is None:
            fail.append(f"無法分類（空文字）：第 {r['line']} 行")
            r['cls'] = 'DECL'
        r['duration'] = wav_duration(r['abs']) if os.path.exists(r['abs']) else 0.0
    by_class = {c: [r for r in rows if r['cls'] == c] for c in CLASS_ORDER}
    for c in CLASS_ORDER:
        items = by_class[c]
        b = duration_buckets(items)
        for r, bi in zip(items, b):
            r['bucket'] = f'{c}-d{bi}'
        items.sort(key=lambda r: (r['duration'], r['utt_id']))
    print('[3/7] 分類與時長分桶：'
          + ', '.join(f'{c}={len(by_class[c])}' for c in CLASS_ORDER))

    # 配額檢查（稀有層下限 2）
    for c in CLASS_ORDER:
        if len(by_class[c]) < 2:
            fail.append(f'稀有層 {c} 僅 {len(by_class[c])} 段 < 2（硬性失敗）')
        if len(by_class[c]) < TEST_QUOTA[c]:
            fail.append(f'{c} 可用 {len(by_class[c])} < test 配額 {TEST_QUOTA[c]}')

    # ---- 4. 分層抽 test（seed=42，類別順序固定）----
    rng = random.Random(args.seed)
    test_set, train_set = set(), set()
    alloc_detail = []
    for c in CLASS_ORDER:
        items = by_class[c]
        sizes = [sum(1 for r in items if r['bucket'] == f'{c}-d{b}')
                 for b in range(N_BUCKETS)]
        alloc = allocate(TEST_QUOTA[c], sizes)
        for b in range(N_BUCKETS):
            bucket = [r for r in items if r['bucket'] == f'{c}-d{b}']
            picked = sample_from_bucket(bucket, alloc[b], rng)
            for r in picked:
                test_set.add(r['utt_id'])
            alloc_detail.append((c, b, sizes[b], alloc[b]))
    for r in rows:
        if r['utt_id'] not in test_set:
            train_set.add(r['utt_id'])
    print(f'[4/7] test 抽樣完成：{len(test_set)} 段（期望 24）')

    # ---- 5. eval10 自 test 再分層（同 seed）----
    rng2 = random.Random(args.seed)
    eval_set = set()
    eval_detail = []
    for c in CLASS_ORDER:
        items = [r for r in by_class[c] if r['utt_id'] in test_set]
        items.sort(key=lambda r: (r['duration'], r['utt_id']))
        sizes = [sum(1 for r in items if r['bucket'] == f'{c}-d{b}')
                 for b in range(N_BUCKETS)]
        alloc = allocate(EVAL_QUOTA[c], sizes)
        for b in range(N_BUCKETS):
            bucket = [r for r in items if r['bucket'] == f'{c}-d{b}']
            picked = sample_from_bucket(bucket, alloc[b], rng2)
            for r in picked:
                eval_set.add(r['utt_id'])
            eval_detail.append((c, b, sizes[b], alloc[b]))
    print(f'[5/7] eval10 抽樣完成：{len(eval_set)} 段（期望 10）')

    # ---- 6. 門禁驗證（記憶體內，全過才寫檔）----
    if len(test_set) != 24:
        fail.append(f'len(test)={len(test_set)} ≠ 24')
    if len(train_set) != args.expected_total - 24:
        fail.append(f'len(train)={len(train_set)} ≠ {args.expected_total - 24}')
    if len(eval_set) != 10:
        fail.append(f'len(eval10)={len(eval_set)} ≠ 10')
    if test_set & train_set:
        fail.append('train ∩ test ≠ ∅')
    if test_set | train_set != {r['utt_id'] for r in rows}:
        fail.append('train ∪ test ≠ 全集')
    if not eval_set <= test_set:
        fail.append('eval10 ⊄ test')

    # 輸入未被修改
    if sha256_of(in_path) != input_sha:
        fail.append('輸入檔 SHA256 在執行過程中改變')

    # 參考音訊候選（train，4–8s，無削波，清單最早列）
    ref = None
    ref_rule = ''
    for lo, hi, rule in [(4.0, 8.0, '4–8s 無削波，清單最早列'),
                         (3.0, 10.0, '回退：3–10s 無削波，清單最早列')]:
        cand = None
        for r in rows:  # 原清單順序
            if r['utt_id'] in train_set and lo <= r['duration'] <= hi:
                if wav_peak(r['abs']) < CLIP_FULLSCALE:
                    cand = r
                    break
        if cand:
            ref, ref_rule = cand, rule
            break
    if ref is None:
        fail.append('train 中找不到 4–8s 無削波候選（回退 3–10s 亦無）')

    if fail:
        print('\n[ABORT] 門禁未過，不寫任何檔案：')
        for f_ in fail:
            print(f'  - {f_}')
        sys.exit(1)
    print(f'[6/7] 門禁全過；參考音訊 = {ref["utt_id"]}（{ref_rule}）')

    # ---- 7. 寫檔（清單沿用原順序；絕對路徑供 GPT-SoVITS 直接讀）----
    os.makedirs(os.path.dirname(os.path.join(ROOT, args.manifest)) or '.',
                exist_ok=True)
    ref_dir = os.path.join(ROOT, args.ref_dir)
    os.makedirs(ref_dir, exist_ok=True)

    def write_list(rel_path, select):
        out = os.path.join(ROOT, rel_path) if not os.path.isabs(rel_path) else rel_path
        with open(out, 'w', encoding='utf-8', newline='\n') as f:
            for r in rows:
                if select(r):
                    f.write(f"{r['abs']}|{r['speaker']}|{r['lang']}|{r['text']}\n")

    write_list(args.train_out, lambda r: r['utt_id'] in train_set)
    write_list(args.test_out, lambda r: r['utt_id'] in test_set)
    write_list(args.eval_out, lambda r: r['utt_id'] in eval_set)

    # manifest
    mpath = os.path.join(ROOT, args.manifest) if not os.path.isabs(args.manifest) else args.manifest
    with open(mpath, 'w', encoding='utf-8', newline='\n') as f:
        w = csv.writer(f)
        w.writerow(['utt_id', 'audio_path', 'speaker', 'duration', 'text',
                    'split', 'bucket', 'md5', 'source_rel', 'class',
                    'eval10'])
        for r in rows:
            split = 'train' if r['utt_id'] in train_set else 'test'
            w.writerow([
                r['utt_id'], r['abs'], r['speaker'],
                f"{r['duration']:.3f}", r['text'], split, r['bucket'],
                md5_of(r['abs']), r['audio'], r['cls'],
                1 if r['utt_id'] in eval_set else 0,
            ])

    # 參考音訊（複製，不動原始檔）
    ref_wav = os.path.join(ref_dir, 'ref.wav')
    ref_txt = os.path.join(ref_dir, 'ref.txt')
    with open(ref['abs'], 'rb') as src, open(ref_wav, 'wb') as dst:
        for chunk in iter(lambda: src.read(1 << 20), b''):
            dst.write(chunk)
    with open(ref_txt, 'w', encoding='utf-8', newline='\n') as f:
        f.write(f"{ref['text']}\n")

    # md5 重複檢查（僅記錄，不剔除；Part 1 已做文字去重）
    md5s = {}
    for r in rows:
        md5s.setdefault(md5_of(r['abs']), []).append(r['utt_id'])
    md5_dupes = {k: v for k, v in md5s.items() if len(v) > 1}

    # ---- 8. 報告 ----
    def agg(uids):
        rs = [r for r in rows if r['utt_id'] in uids]
        return len(rs), sum(r['duration'] for r in rs)

    n_tr, d_tr = agg(train_set)
    n_te, d_te = agg(test_set)
    n_ev, d_ev = agg(eval_set)

    rep = os.path.join(ROOT, args.report) if not os.path.isabs(args.report) else args.report
    os.makedirs(os.path.dirname(rep) or '.', exist_ok=True)
    with open(rep, 'w', encoding='utf-8', newline='\n') as f:
        w = f.write
        w('# Part 2 資料切分報告\n\n')
        w('## 執行摘要\n')
        w(f'- 自 `data/ryza_train.list`（{len(rows)} 行）切出 train / test / eval10。\n')
        w(f'- 方法：句尾標點分層（Q/EXCL/ELL/DECL）× 時長三等分桶，'
          f'桶內隨機抽樣，seed = {args.seed}。\n')
        w(f'- 參考音訊規則：{ref_rule}。\n')
        w(f'- **尚未開始訓練，未執行任何預處理腳本。**\n\n')

        w('## 輸入完整性\n')
        w(f'- 輸入 SHA256（前 8 碼）：`{input_sha[:8]}`\n')
        w(f'- 輸入行數：{len(rows)}（期望 {args.expected_total}）\n')
        w(f'- 執行結束後 SHA256：`{sha256_of(in_path)[:8]}`（未變）\n\n')

        w('## 分層配額實際達成\n\n')
        w('| 類別 | 全集 | test 配額 | test 實際 | eval 配額 | eval 實際 | train |\n')
        w('|---|---|---|---|---|---|---|\n')
        for c in CLASS_ORDER:
            n_all = len(by_class[c])
            n_te_c = sum(1 for r in rows
                         if r['cls'] == c and r['utt_id'] in test_set)
            n_ev_c = sum(1 for r in rows
                         if r['cls'] == c and r['utt_id'] in eval_set)
            w(f'| {c}（{CLASS_LABEL[c]}） | {n_all} | {TEST_QUOTA[c]} '
              f'| {n_te_c} | {EVAL_QUOTA[c]} | {n_ev_c} '
              f'| {n_all - n_te_c} |\n')
        w(f'| **合計** | **{len(rows)}** | **{sum(TEST_QUOTA.values())}** '
          f'| **{n_te}** | **{sum(EVAL_QUOTA.values())}** | **{n_ev}** '
          f'| **{n_tr}** |\n\n')

        w('## 切分統計\n\n')
        w('| 清單 | 段數 | 總時長 |\n|---|---|---|\n')
        w(f'| train（`{args.train_out}`） | {n_tr} | {d_tr / 60:.2f} 分 |\n')
        w(f'| test（`{args.test_out}`） | {n_te} | {d_te / 60:.2f} 分 |\n')
        w(f'| eval10（`{args.eval_out}`，⊆ test） | {n_ev} | {d_ev / 60:.2f} 分 |\n\n')
        w(f'- 全集總時長：{(d_tr + d_te) / 60:.2f} 分\n')
        w(f'- 時長範圍（全集）：'
          f'{min(r["duration"] for r in rows):.2f}–'
          f'{max(r["duration"] for r in rows):.2f} 秒\n')
        w(f'- test 時長範圍：'
          f'{min(r["duration"] for r in rows if r["utt_id"] in test_set):.2f}–'
          f'{max(r["duration"] for r in rows if r["utt_id"] in test_set):.2f} 秒\n\n')

        w('## 參考音訊\n\n')
        w(f'- utt_id：`{ref["utt_id"]}`\n')
        w('- 來源：`{}`（清單第 {} 行）\n'.format(ref['audio'], ref['line']))
        w(f'- 時長：{ref["duration"]:.2f} 秒\n')
        w(f'- 文字：{ref["text"]}\n')
        w(f'- 峰值：{wav_peak(ref["abs"])} / {CLIP_FULLSCALE}（未削波）\n')
        w(f'- 輸出：`{args.ref_dir}/ref.wav`、`{args.ref_dir}/ref.txt`\n')
        w('- 後續相似度與生成測試固定使用此參考音訊，跨版本一致。\n\n')

        w('## 門禁檢查\n\n')
        checks = [
            ('輸入 491 行', len(rows) == args.expected_total),
            ('train ∩ test = ∅', not (test_set & train_set)),
            ('train ∪ test = 491',
             test_set | train_set == {r['utt_id'] for r in rows}),
            (f'len(train) = {args.expected_total - 24}', len(train_set) == args.expected_total - 24),
            ('len(test) = 24', len(test_set) == 24),
            ('len(eval10) = 10', len(eval_set) == 10),
            ('路徑存在、四欄格式、文字非空且無 |',
             all(os.path.exists(r['abs']) and r['speaker'] == 'ryza'
                 and r['lang'] == 'ja' and r['text']
                 and '|' not in r['text'] for r in rows)),
            ('三清單無重複路徑',
             len({r['utt_id'] for r in rows}) == len(rows)),
            ('eval10 ⊆ test', eval_set <= test_set),
            ('ryza_train.list SHA256 未變',
             sha256_of(in_path) == input_sha),
            ('參考音訊來自 train（4–8s、無削波、清單最早列）',
             ref is not None and ref['utt_id'] in train_set),
        ]
        for name, ok in checks:
            w(f'- [{"x" if ok else " "}] {"PASS" if ok else "FAIL"}：{name}\n')
        w(f'\n通過 {sum(1 for _, ok in checks if ok)}/{len(checks)} 項。\n\n')

        w('## 分桶明細（test 抽樣分配）\n\n')
        w('| 類別 | 桶 | 桶內數 | 抽出 |\n|---|---|---|---|\n')
        for c, b, size, k in alloc_detail:
            w(f'| {c} | d{b} | {size} | {k} |\n')
        w('\n## 分桶明細（eval10 抽樣分配）\n\n')
        w('| 類別 | 桶 | 桶內數 | 抽出 |\n|---|---|---|---|\n')
        for c, b, size, k in eval_detail:
            w(f'| {c} | d{b} | {size} | {k} |\n')
        w('\n## 風險與待確認\n\n')
        w('- **清單檔名**：本切分依夜跑規格輸出 `data/finetune_train.list`、'
          '`data/finetune_test.list`、`data/finetune_eval10.list`；'
          '未追蹤的 `docs/03-01_dataset.md` 草稿曾規劃 `ryza_train_finetune.list` '
          '等不同檔名，兩者待統一（待確認）。\n')
        w('- **清單使用絕對路徑**：為讓 GPT-SoVITS（獨立目錄）直接讀取，'
          '三份清單與 manifest 的音檔路徑為絕對路徑；manifest 另存 '
          '`source_rel` 原始相對路徑以便重建。\n')
        w('- **句尾標點為情緒代理**：Q/EXCL/ELL/DECL 僅反映句尾標點，'
          '非語音情緒標註；如「どうかな? ちょっと恥ずましいんだけど」'
          '句尾非問號但語義上偏疑問，此類邊界案例未特別處理。\n')
        w('- **MD5 重複**：'
          f'偵測到 {len(md5_dupes)} 組音訊內容完全重複，'
          '本階段不剔除（Part 1 已以文字去重移除 6 段）；'
          '若需更嚴格去重，待確認是否重新切分。\n')
        w('- **測試集不參與訓練**：`finetune_train.list` 為 WebUI 1Aa–1Ac '
          '與 1B 唯一輸入；`finetune_test.list` 嚴禁進入任何訓練步驟。\n')
        w('- **尚未訓練**：本階段只產出資料切分與參考音訊，'
          '未啟動任何訓練或預處理腳本。\n\n')
        w('## 下一步\n\n')
        w('- 等待使用者確認切分結果後，才進入 GPT-SoVITS v4 LoRA '
          '環境與訓練配置（模塊二：環境與模型）。\n')

    print(f'[7/7] 寫檔完成：{args.train_out}({n_tr})、'
          f'{args.test_out}({n_te})、{args.eval_out}({n_ev})、'
          f'{args.manifest}、{args.ref_dir}/ref.wav、{args.report}')
    print(f'      train {d_tr / 60:.2f} 分｜test {d_te / 60:.2f} 分'
          f'｜eval10 {d_ev / 60:.2f} 分')
    print(f'      MD5 重複組數：{len(md5_dupes)}')


if __name__ == '__main__':
    main()
