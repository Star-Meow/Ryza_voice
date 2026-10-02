#!/usr/bin/env python3
"""ASR 預篩：以 faster-whisper large-v3 + VAD 過濾候選語音。

過濾條件（記錄排除原因）：
  1. 時長 < 2 秒或 > 15 秒（免模型，讀 wav header）
  2. 非說話聲（喘氣、吶喊等短促感嘆）→ 標記 REVIEW 由人工確認
  3. ASR 輸出為空（VAD 後無語音）→ 排除
  4. 重複或明顯幻覺（Whisper 常見幻覺輸出、單字重複）→ 排除

路徑一律由參數傳入，不寫死本機絕對路徑。

--pos-label 模式：讀 SVM_DEC 編號清單（如 tools/pos_label.py），從
--audio-root（預設 wav）載入對應音檔；ASR 後可用 --out-list 直接寫出
含逐字稿的訓練清單（path|label|lang|文字）。
"""
import argparse
import importlib.util
import json
import os
import re
import sys
import wave
from collections import Counter


# --- 非說話聲候選：日文短促感嘆／喘氣（無語意內容），heuristic ---
INTERJECTIONS = {
    # 單音
    'あ', 'い', 'う', 'え', 'お', 'ん', 'っ',
    # 撥氣／吃驚（促音）
    'はっ', 'あっ', 'うっ', 'えっ', 'おっ', 'いっ',
    'くっ', 'ぐっ', 'すっ', 'なっ', 'まっ', 'やっ', 'よっ', 'わっ',
    'かっ', 'さっ', 'たっ', 'らっ', 'りっ', 'しゅっ', 'ひゅっ',
    # 呼吸／嘆氣
    'はぁ', 'はあ', 'ふぅ', 'ふう', 'はー', 'ふー', 'はあぁ',
    'うー', 'あー', 'えー', 'おー', 'いー',
    # 戰吼
    'せいっ', 'ていっ', 'とうっ', 'やあっ', 'とあっ',
}

# Whisper 已知幻覺輸出（英/日）
HALLUCINATION_PATTERNS = [
    'thank you for watching', 'thanks for watching',
    'please subscribe', 'consider subscribing',
    'thank you', 'goodbye', 'bye for now',
    'ご視聴ありがとうございました', 'チャンネル登録',
    'よろしくお願いします', 'ありがとうございました',
    'さようなら', 'ご覧いただき', 'お聴きいただき',
]

# 只保留日文/英數字元，用於比對感嘆詞與重複判定
_KEEP = re.compile(r'[一-龠ぁ-んァ-ヶーa-zA-Z0-9]')
# 同一字元連續出現 4 次以上（幻覺性重複）
_REPEAT = re.compile(r'(.)\1{3,}')


def wav_duration(path):
    with wave.open(path, 'rb') as w:
        return w.getnframes() / w.getframerate()


def core_text(s):
    return ''.join(_KEEP.findall(s))


def load_svm_dec(path):
    """從 .py 檔載入 SVM_DEC 編號清單（正樣本編號，對應 wav/NNNNN.wav）。"""
    spec = importlib.util.spec_from_file_location('pos_label', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, 'SVM_DEC'):
        sys.exit(f'錯誤：{path} 缺少 SVM_DEC')
    return mod.SVM_DEC


def classify(rec, min_dur, max_dur):
    """回傳 (status, reason, evidence)；status 為 PASS / EXCLUDE / REVIEW。"""
    d = rec['duration']
    if d < min_dur:
        return 'EXCLUDE', 'DURATION_SHORT', f'{d:.1f}s < {min_dur}s'
    if d > max_dur:
        return 'EXCLUDE', 'DURATION_LONG', f'{d:.1f}s > {max_dur}s'

    text = rec['text'].strip()
    core = rec['core']
    if not text:
        return 'EXCLUDE', 'EMPTY', 'VAD 後無文字輸出'

    low = text.lower()
    for p in HALLUCINATION_PATTERNS:
        if p in low:
            return 'EXCLUDE', 'HALLUCINATION', f'命中「{p}」'

    if _REPEAT.search(core):
        return 'EXCLUDE', 'REPETITION', f'單字重複 {core[:14]}'

    if core in INTERJECTIONS:
        return 'REVIEW', 'NON_SPEECH', f'疑似感嘆詞「{core}」'

    return 'PASS', '', ''


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--list', default='ryza_train.list',
                    help='輸入清單（path|speaker|lang| 格式）')
    ap.add_argument('--pos-label',
                    help='改讀 SVM_DEC 編號清單（如 tools/pos_label.py），'
                         '從 --audio-root 載入對應音檔（優先於 --list）')
    ap.add_argument('--audio-root', default='wav',
                    help='--pos-label 模式的音檔來源目錄')
    ap.add_argument('--label', default='ryza',
                    help='--pos-label 模式的說話者標籤（清單第二欄）')
    ap.add_argument('--lang', default='ja',
                    help='--pos-label 模式的語言（清單第三欄）')
    ap.add_argument('--out-list',
                    help='已停用：清單產出改由 tools/transcribe.py 接手'
                         '（保留選項以維持向後相容，指定時無效）')
    ap.add_argument('--temperature', type=float, default=None,
                    help='解碼溫度；不指定時用 faster-whisper 預設梯度'
                         '（0.0→1.0 fallback，可逃出解碼迴圈、抗幻覺）。'
                         '指定固定值（如 0.0）可完全重現，但會關閉 fallback')
    ap.add_argument('--report', default='asr_filter_report.md',
                    help='報告輸出路徑')
    ap.add_argument('--raw-out', default='tools/asr_screening.json',
                    help='原始 ASR 結果（供 --out-list 與後續分析重用）')
    ap.add_argument('--model', default='large-v3')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--compute-type', default='float16')
    ap.add_argument('--min-dur', type=float, default=2.0)
    ap.add_argument('--max-dur', type=float, default=15.0)
    ap.add_argument('--limit', type=int, default=0, help='只跑前 N 段（0 = 全跑）')
    args = ap.parse_args()

    # --- 讀清單（或從 pos_label 編號展開）---
    entries = []
    if args.pos_label:
        ids = load_svm_dec(args.pos_label)
        entries = [{'path': f'{args.audio_root}/{n:05d}.wav',
                    'speaker': args.label, 'lang': args.lang}
                   for n in ids]
    else:
        with open(args.list, encoding='utf-8') as f:
            for line in f:
                line = line.rstrip('\n')
                if not line.strip():
                    continue
                parts = line.split('|')
                entries.append({
                    'path': parts[0],
                    'speaker': parts[1] if len(parts) > 1 else '',
                    'lang': parts[2] if len(parts) > 2 else 'ja',
                    'line': line,
                })
    if args.limit:
        entries = entries[:args.limit]
    print(f'清單 {len(entries)} 段，模型 {args.model}'
          f'（{args.device}/{args.compute_type}，temperature='
          f'{"梯度 fallback" if args.temperature is None else args.temperature}）',
          file=sys.stderr)

    # --- 載入模型 ---
    from faster_whisper import WhisperModel
    model = WhisperModel(args.model, device=args.device,
                         compute_type=args.compute_type)

    # --- 逐段預篩 ---
    results = []
    for i, e in enumerate(entries, 1):
        rec = {'path': e['path'], 'speaker': e['speaker'], 'lang': e['lang']}
        if not os.path.exists(e['path']):
            rec.update(status='EXCLUDE', reason='MISSING',
                       evidence='音檔不存在', duration=None,
                       text='', core='')
            results.append(rec)
            continue
        try:
            rec['duration'] = wav_duration(e['path'])
        except Exception as ex:
            rec.update(status='EXCLUDE', reason='READ_ERROR',
                       evidence=str(ex)[:60], duration=None,
                       text='', core='')
            results.append(rec)
            continue

        try:
            # temperature=None 時不傳，沿用 faster-whisper 預設梯度
            # （0.0→1.0 fallback，可逃出解碼迴圈）
            tkwargs = {}
            if args.temperature is not None:
                tkwargs['temperature'] = args.temperature
            segments, info = model.transcribe(
                e['path'], language=e['lang'] or 'ja',
                vad_filter=True, beam_size=5, **tkwargs)
            segs = list(segments)
        except Exception as ex:
            rec.update(status='EXCLUDE', reason='ASR_ERROR',
                       evidence=str(ex)[:60], text='', core='')
            results.append(rec)
            continue

        text = ''.join(s.text for s in segs).strip()
        rec['text'] = text
        rec['core'] = core_text(text)
        rec['no_speech_prob'] = round(
            sum(s.no_speech_prob for s in segs) / max(len(segs), 1), 4)
        rec['avg_logprob'] = round(
            sum(s.avg_logprob for s in segs) / max(len(segs), 1), 3)
        rec['compression_ratio'] = round(
            sum(s.compression_ratio for s in segs) / max(len(segs), 1), 2)
        rec['segments'] = [
            {'start': round(s.start, 2), 'end': round(s.end, 2),
             'text': s.text.strip()} for s in segs]

        status, reason, evidence = classify(rec, args.min_dur, args.max_dur)
        rec['status'] = status
        rec['reason'] = reason
        rec['evidence'] = evidence
        results.append(rec)

        if i % 50 == 0:
            print(f'  進度 {i}/{len(entries)}', file=sys.stderr)

    # --- 跨檔重複（同一文字出現 ≥2 次 → 排除後續重複者，保留首次出現者）---
    text_first = {}   # core → 首次出現的 path
    for r in results:
        if r['status'] == 'PASS' and r['core']:
            if r['core'] not in text_first:
                text_first[r['core']] = r['path']
    text_cnt = Counter(r['core'] for r in results
                       if r['status'] == 'PASS' and r['core'])
    for r in results:
        if (r['status'] == 'PASS' and text_cnt[r['core']] >= 2
                and text_first[r['core']] != r['path']):
            r['status'] = 'EXCLUDE'
            r['reason'] = 'DUPLICATE_TEXT'
            r['evidence'] = (f'「{r["core"][:14]}」與 '
                             f'`{text_first[r["core"]]}` 文本重複')

    # --- 高壓縮比（Whisper 幻覺指標），標記 REVIEW ---
    for r in results:
        if (r['status'] == 'PASS' and r.get('compression_ratio', 0) > 3.0):
            r['status'] = 'REVIEW'
            r['reason'] = 'HIGH_COMPRESSION'
            r['evidence'] = f'compression_ratio={r["compression_ratio"]}'

    # --- 統計 ---
    def cnt(status, reason=None):
        return sum(1 for r in results
                   if r['status'] == status and (reason is None or r['reason'] == reason))
    n_pass = cnt('PASS')
    excl = [r for r in results if r['status'] == 'EXCLUDE']
    rev = [r for r in results if r['status'] == 'REVIEW']

    # --- 輸出原始 JSON ---
    os.makedirs(os.path.dirname(args.raw_out) or '.', exist_ok=True)
    with open(args.raw_out, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print(f'原始結果 → {args.raw_out}', file=sys.stderr)

    # --- 輸出報告 ---
    with open(args.report, 'w', encoding='utf-8') as f:
        w = f.write
        w(f'# ASR 預篩報告\n\n')
        w(f'- 清單：`{args.list}`（{len(entries)} 段）\n')
        w(f'- 模型：{args.model}（{args.device}/{args.compute_type}），VAD 開啟\n')
        w(f'- 時長門檻：{args.min_dur}s ~ {args.max_dur}s\n\n')
        w(f'## 摘要\n\n')
        w(f'| 結果 | 數量 |\n|---|---|\n')
        w(f'| ✅ 通過 | **{n_pass}** |\n')
        w(f'| ❌ 排除（確定） | {len(excl)} |\n')
        w(f'| ⚠️ 待人工確認 | {len(rev)} |\n\n')

        w(f'## ❌ 排除明細（依原因）\n\n')
        for reason in ['DURATION_SHORT', 'DURATION_LONG', 'EMPTY',
                       'HALLUCINATION', 'REPETITION', 'DUPLICATE_TEXT',
                       'MISSING', 'READ_ERROR', 'ASR_ERROR']:
            items = [r for r in excl if r['reason'] == reason]
            if not items:
                continue
            w(f'### {reason}（{len(items)}）\n\n')
            w(f'| 音檔 | 證據 |\n|---|---|\n')
            for r in items:
                w(f'| `{r["path"]}` | {r["evidence"]} |\n')
            w('\n')

        w(f'## ⚠️ 待人工確認明細\n\n')
        if rev:
            w(f'| 音檔 | 原因 | 證據 | ASR 文字 |\n|---|---|---|---|\n')
            for r in rev:
                txt = (r.get('text') or '')[:40].replace('|', '\\|')
                w(f'| `{r["path"]}` | {r["reason"]} | {r["evidence"]} '
                  f'| {txt} |\n')
        else:
            w('（無）\n')
        w('\n')

        w(f'## 時長分佈（通過者）\n\n')
        ds = [r['duration'] for r in results if r['status'] == 'PASS']
        if ds:
            w(f'- 平均 {sum(ds)/len(ds):.1f}s｜最短 {min(ds):.1f}s｜'
              f'最長 {max(ds):.1f}s｜合計 {sum(ds)/60:.1f} 分鐘\n')

    print(f'報告 → {args.report}', file=sys.stderr)
    print(f'結果：通過 {n_pass} / 排除 {len(excl)} / 待確認 {len(rev)}',
          file=sys.stderr)


if __name__ == '__main__':
    main()
