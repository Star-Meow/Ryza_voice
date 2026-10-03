#!/usr/bin/env python3
"""訓練音訊品質量測：比對 GPT-SoVITS 官方 wiki 的「低音質訓練集」判定條件。

官方 wiki（features/Latest）列出會把訓練集歸類為低音質、因而建議走
V2(V2Pro) VITS 路線的現象：被壓縮得比較嚴重、16k 以下頻譜有頻段缺失、
較多混響／背景噪音。本工具對訓練清單隨機抽樣做純 CPU 頻譜量測，
產出可重現的判定表，做為模型版本選型的客觀依據。

四項指標：
  1. LTAS（長時平均頻譜）相對 0.5–4k 的能量，輸出 16k 等關鍵頻點
  2. -60 dB 頻寬中位數（低通截斷的直接證據）
  3. 10–20k 區間內 1kHz 視窗的最大陡降（codec cliff）
  4. 安靜段 vs 語音段的頻譜輪廓差（區分寬頻背景噪音與殘響尾）

全程使用 Python 標準庫 + numpy；不載入模型、不使用 GPU。

用法：
    python tools/audio_quality.py \
        --train-list data/ryza_train.list \
        --sample 60 --seed 0 \
        --output reports/audio_quality.md

    # 全數量測（抽樣數 = 清單行數）
    python tools/audio_quality.py --train-list data/ryza_train.list --sample 0
"""
import argparse
import json
import os
import random
import sys

import numpy as np


def read_wav(path):
    """標準庫讀 WAV → (float32, sr)。僅支援 16bit PCM（本庫格式）。"""
    import wave
    with wave.open(path, 'rb') as w:
        sr = w.getframerate()
        ch = w.getnchannels()
        raw = w.readframes(w.getnframes())
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(1)
    return x, sr


def frame_powers(x, n=4096, hop=None, rms_floor=0.005):
    """逐窗 FFT 功率譜；丟掉 RMS 過低的窗（靜音填充不應计入頻譜輪廓）。

    回傳 (T, F) 功率矩陣。
    """
    hop = hop or n
    out = []
    for i in range(0, len(x) - n, hop):
        f = x[i:i + n]
        if np.sqrt(np.mean(f ** 2)) < rms_floor:
            continue
        win = f * np.hanning(n)
        out.append(np.abs(np.fft.rfft(win)) ** 2)
    return np.stack(out) if out else np.empty((0, n // 2 + 1))


def bandwidth_db(P, freqs, ref_band=(500., 4000.), db=-60.):
    """相對參考頻帶，能量仍高於 db 的最高頻率（Hz）。"""
    plog = 10 * np.log10(P + 1e-20)
    ref = np.median(plog[(freqs >= ref_band[0]) & (freqs <= ref_band[1])])
    above = freqs[plog - ref > db]
    return float(above.max()) if above.size else 0.0


def max_cliff(P, freqs, lo=10000., hi=20000., step=500., window=1000.,
              threshold=18.):
    """10–20k 內最陡的 window 視窗下降（dB）；回傳 (最大陡降, 是否超門檻)。"""
    plog = 10 * np.log10(P + 1e-20)
    ref = np.median(plog[(freqs >= 500) & (freqs <= 4000)])
    rel = plog - ref

    def at(f):
        return rel[np.argmin(np.abs(freqs - f))]

    worst, n_over = 0.0, 0
    k = lo
    while k < hi:
        drop = at(k) - at(k + window)
        worst = max(worst, float(drop))
        if drop > threshold:
            n_over += 1
        k += step
    return worst, n_over


def quiet_profile(x, n=2048, sr=48000., quiet_rms=0.01):
    """安靜段 vs 語音段的輪廓差（dB）。

    殘響尾仍是語音染色，低頻與高頻會同步下降；恆定寬頻噪音則在高頻
    相對接近語音段。回傳 (低頻差, 高頻差, 寬頻噪音判定)。
    """
    hop = n
    loud, quiet = [], []
    for i in range(0, len(x) - n, hop):
        f = x[i:i + n]
        r = np.sqrt(np.mean(f ** 2))
        if r <= 0:
            continue  # 精確數位零（靜音填充）非真實底噪，排除
        P = np.abs(np.fft.rfft(f * np.hanning(n))) ** 2
        (quiet if r < quiet_rms else loud).append(P)
    if not loud or not quiet:
        return None
    freqs = np.fft.rfftfreq(n, 1 / sr)
    L = 10 * np.log10(np.mean(loud, axis=0) + 1e-20)
    Q = 10 * np.log10(np.mean(quiet, axis=0) + 1e-20)
    low = float(np.mean(Q[(freqs >= 100) & (freqs <= 300)])
                - np.mean(L[(freqs >= 100) & (freqs <= 300)]))
    high = float(np.mean(Q[(freqs >= 4000) & (freqs <= 8000)])
                 - np.mean(L[(freqs >= 4000) & (freqs <= 8000)]))
    # 高頻落差明顯小於低頻（相對接近 0）→ 寬頻噪音而非殘響尾
    broadband = high > -15.0
    return low, high, broadband


def measure(path, n_fft=4096):
    """單檔四項量測。"""
    x, sr = read_wav(path)
    frames = frame_powers(x, n=n_fft)
    if frames.shape[0] < 2:
        return None
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    P = frames.mean(0)
    plog = 10 * np.log10(P + 1e-20)
    ref = np.median(plog[(freqs >= 500) & (freqs <= 4000)])

    pts = {f'{int(f)}k': round(float(plog[np.argmin(np.abs(freqs - f * 1000))]
                                   - ref), 1)
           for f in (4, 8, 12, 14, 16, 18, 20)}
    bw60 = bandwidth_db(P, freqs, db=-60)
    cliff, n_cliff = max_cliff(P, freqs)

    qp = quiet_profile(x, sr=sr)
    if qp is None:
        return None
    low, high, broadband = qp
    return {
        'path': path,
        'sr': sr,
        'duration': round(len(x) / sr, 2),
        'ltas_rel_db': pts,
        'bw60_hz': round(bw60, 0),
        'cliff_db': round(cliff, 1),
        'n_cliff_windows': n_cliff,
        'quiet_low_db': round(low, 1),
        'quiet_high_db': round(high, 1),
        'broadband_noise': bool(broadband),
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--train-list', default='data/ryza_train.list',
                    help='GPT-SoVITS 格式清單（第一欄為音檔路徑）')
    ap.add_argument('--sample', type=int, default=60,
                    help='隨機抽樣段數；0 代表全數量測')
    ap.add_argument('--seed', type=int, default=0, help='隨機種子')
    ap.add_argument('--output', default=None, help='報告輸出路徑（.md）')
    ap.add_argument('--json', default=None, help='逐筆量測結果 JSON 輸出')
    args = ap.parse_args()

    with open(args.train_list, encoding='utf-8') as f:
        paths = [l.split('|')[0] for l in f.read().splitlines()
                 if l.strip()]
    if not paths:
        sys.exit(f'錯誤：{args.train-list} 無有效行')

    # 清單依 svm_dec 排序（見 README），抽樣時記錄位置以證明涵蓋前後段
    idx_all = list(range(len(paths)))
    if args.sample and 0 < args.sample < len(paths):
        random.seed(args.seed)
        picked = sorted(random.sample(idx_all, args.sample))
    else:
        picked = idx_all
    selected = [paths[i] for i in picked]

    n = len(paths)
    tercile = lambda i: 'front' if i < n / 3 else ('mid' if i < 2 * n / 3
                                                   else 'back')
    cover = {k: 0 for k in ('front', 'mid', 'back')}
    for i in picked:
        cover[tercile(i)] += 1

    results = []
    for p in selected:
        try:
            r = measure(p)
        except Exception as e:  # 單檔失敗不中斷
            print(f'警告：{p} 量測失敗（{e}）', file=sys.stderr)
            continue
        if r:
            results.append(r)

    if not results:
        sys.exit('錯誤：無任何檔案完成量測')

    m = len(results)
    bw = np.array([r['bw60_hz'] for r in results])
    cl = np.array([r['cliff_db'] for r in results])
    n_cliff = sum(1 for r in results if r['n_cliff_windows'] > 0)
    n_noise = sum(1 for r in results if r['broadband_noise'])
    ql = np.array([r['quiet_low_db'] for r in results])
    qh = np.array([r['quiet_high_db'] for r in results])

    # 全體 LTAS：把各檔功率譜平均後再取 dB
    n_fft = 4096
    stack = []
    for p in selected:
        try:
            x, sr = read_wav(p)
        except Exception:
            continue
        fr = frame_powers(x, n=n_fft)
        if fr.shape[0]:
            stack.append(fr.mean(0))
    freqs = np.fft.rfftfreq(n_fft, 1 / results[0]['sr'])
    P = np.mean(stack, axis=0)
    plog = 10 * np.log10(P + 1e-20)
    ref = np.median(plog[(freqs >= 500) & (freqs <= 4000)])
    ltas = {f'{int(f)}k': round(float(plog[np.argmin(np.abs(freqs - f * 1000))]
                                    - ref), 1)
            for f in (4, 8, 12, 14, 16, 18, 20)}

    e16 = ltas['16k']
    bw_med = float(np.median(bw))
    cliff_med = float(np.median(cl))

    # 官方 wiki 條件判定
    # 「16k 以下頻段缺失」：-60dB 頻寬收在 16k 以下，或 16k 能量被
    # 壓到中頻以下 40dB（兩者皆屬低通截斷的特徵）。本庫 48kHz 取樣，
    # 高品質檔的 -60dB 頻寬應接近 20k 以上。
    c_band = bw_med < 16000 or e16 < -40
    c_cliff = n_cliff > m * 0.2
    c_noise = n_noise > m * 0.2
    lowq = c_band or c_cliff or c_noise

    lines = []
    A = lines.append
    A('# 音訊品質實測報告\n')
    A('## 目的')
    A('驗證訓練資料是否符合 GPT-SoVITS 官方 wiki 的「低音質」判定條件，')
    A('以決定資料品質是否支持 V2Pro 路線的選型論點。\n')
    A('## 方法')
    A(f'- 抽樣：seed {args.seed}，{"隨機 " + str(m) + " 段" if args.sample else "全數 " + str(m) + " 段"}'
      f'（清單共 {n} 段，依 svm_dec 排序）')
    A(f'- 位置涵蓋：front {cover["front"]} / mid {cover["mid"]} / back {cover["back"]}（前/中/後段）')
    A('- 工具：`tools/audio_quality.py`，標準庫 + numpy 純 CPU FFT；')
    A('  **不載入模型、不使用 GPU**\n')
    A('## 官方判定條件對照')
    A('| 條件 | 官方描述 | 量測結果 | 判定 |')
    A('|---|---|---|---|')
    A(f'| 16k 以下頻段缺失 | 壓縮造成低通 | {e16:+.1f} dB @16k，'
      f'-60dB 頻寬中位數 {bw_med / 1000:.1f} kHz | '
      f'{"觸發" if c_band else "未觸發"} |')
    A(f'| 壓縮陡降 | 低通截斷懸崖 | {n_cliff}/{m} 檔 >18dB，'
      f'中位數 {cliff_med:.1f} dB | {"觸發" if c_cliff else "未觸發"} |')
    A(f'| 背景噪音 | 寬頻噪音 | 安靜段低/高頻 {np.median(ql):+.1f}/'
      f'{np.median(qh):+.1f} dB，{n_noise}/{m} 寬頻特徵 | '
      f'{"觸發" if c_noise else "未觸發"} |\n')
    A('## LTAS 長時平均頻譜（相對 0.5–4k 頻帶，dB）\n')
    A('| 頻率 | 相對能量 |')
    A('|---|---|')
    for f in (4, 8, 12, 14, 16, 18, 20):
        A(f'| {f} kHz | {ltas[f"{f}k"]:+.1f} dB |')
    A('')
    A('## 結論')
    if lowq:
        A('本批音訊**符合**官方 wiki 的「低音質」分類，')
        A('資料品質支持選擇對品質較寬容的 V2Pro 路線。')
    else:
        A('本批音訊**不符合**官方 wiki 的「低音質」分類。')
        A('資料品質不構成選擇 V2Pro 的理由；V2Pro 優先的依據')
        A('改為資源成本（同卡 batch 6 vs 1）與官方 benchmark 打平。')
    A('')
    A('> 殘響尚無法由本工具精確量化：安靜段的語音染色輪廓只能排除')
    A('> 「恆定寬頻背景噪音」，無法區分輕微空間感與無殘響。')
    A('> issue 所述「少量音訊帶輕微空間感」屬於這一類未量化的殘留風險。\n')

    text = '\n'.join(lines)
    if args.output:
        d = os.path.dirname(os.path.abspath(args.output)) or '.'
        os.makedirs(d, exist_ok=True)
        with open(args.output, 'w', encoding='utf-8', newline='\n') as f:
            f.write(text)
        print(f'{args.output} 已產生（{m} 段量測）')
    else:
        print(text)

    if args.json:
        d = os.path.dirname(os.path.abspath(args.json)) or '.'
        os.makedirs(d, exist_ok=True)
        with open(args.json, 'w', encoding='utf-8') as f:
            json.dump({'sample': m, 'total': n, 'seed': args.seed,
                       'cover': cover, 'ltas': ltas, 'results': results},
                      f, ensure_ascii=False, indent=1)
        print(f'{args.json} 已產生')

    print(f'判定：16k缺頻段={c_band} 陡降={c_cliff} 噪音={c_noise} '
          f'→ 低音質={lowq}')


if __name__ == '__main__':
    main()
