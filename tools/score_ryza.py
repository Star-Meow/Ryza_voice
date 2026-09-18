"""分析：以已知ライザ樣本為參考，為主庫 9569 檔評分並找出ライザ語音。

校準來源：
- 正樣本：使用者指定的 00025/00030/00042（主庫對話語音）
- 正樣本：battle_voice 中 54 個 RYZA 戰鬥語音（不同錄音條件）
- 負樣本：battle_voice 中 KLAU/TAO/LENT/PAT/ANCIENT 等其他角色
"""
import os
import sys
import csv
import glob
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

N = 9569


def load_index():
    rows = {}
    with open('voice_index.csv', newline='', encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            rows[int(r['wav_file'].split('.')[0])] = r
    return rows


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def main():
    E = np.load('tools/embeddings.npy')
    assert E.shape == (N + 1, 192), E.shape

    # --- 戰鬥語音參考嵌入（具名角色，用於校準）---
    import spk_model
    groups = {}
    for p in sorted(glob.glob('battle_voice/*.wav')):
        base = os.path.basename(p)
        for name in ['RYZA', 'KLAU', 'TAO', 'LENT', 'PAT', 'ANCIENT', 'FI', 'CELIA',
                     'CLAUDIA', 'Lila', 'EMEL', 'DENIVAL', 'Agnes']:
            if f'_{name}_' in base:
                groups.setdefault(name, []).append(p)
                break

    print('=== battle_voice 角色樣本數 ===')
    for k, v in sorted(groups.items(), key=lambda x: -len(x[1])):
        print(f'{k:10s} {len(v)}')

    user_pos = [25, 30, 42]
    pos_user = np.mean([E[i] for i in user_pos], axis=0)
    pos_user /= np.linalg.norm(pos_user)

    # 交叉驗證：使用者指定的 3 個主庫樣本 vs 戰鬥庫 RYZA
    print('\n=== 使用者指定樣本 vs 戰鬥庫 RYZA（跨條件驗證）===')
    ryza_battle = groups.get('RYZA', [])
    ryza_battle_emb = [spk_model.embed(p) for p in ryza_battle[:12]]
    for i in user_pos:
        cs = [cos(E[i], e) for e in ryza_battle_emb]
        print(f'{i:05d}.wav  vs RYZA battle: min={min(cs):.3f} mean={np.mean(cs):.3f}')

    # 負樣本分布
    print('\n=== 各角色 vs 使用者指定 centroid（負樣本校準）===')
    for name in ['KLAU', 'TAO', 'LENT', 'PAT', 'ANCIENT']:
        embs = [spk_model.embed(p) for p in groups.get(name, [])[:12]]
        if embs:
            cs = [cos(pos_user, e) for e in embs]
            print(f'{name:10s} mean={np.mean(cs):.3f} max={max(cs):.3f}')

    # --- 主庫評分 ---
    scores = E[1:] @ pos_user  # (9569,)
    idx = np.arange(1, N + 1)
    order = np.argsort(-scores)

    print('\n=== 主庫分數分布 ===')
    for q in [50, 75, 90, 95, 98, 99, 99.5]:
        print(f'  p{q}: {np.percentile(scores, q):.4f}')

    np.save('tools/ryza_scores.npy', scores)

    print('\n=== Top 40 最相似主庫檔 ===')
    rows = load_index()
    for i in order[:40]:
        r = rows[i]
        print(f'{i:05d}.wav {float(r["duration_seconds"]):>6}s  cos={scores[i-1]:.4f}')

    print('\n=== 分數區間計數 ===')
    for lo, hi in [(0.9, 1.01), (0.85, 0.9), (0.8, 0.85), (0.75, 0.8), (0.7, 0.75), (0.6, 0.7), (0, 0.6)]:
        c = ((scores >= lo) & (scores < hi)).sum()
        print(f'  [{lo:.2f},{hi:.2f}): {c}')


if __name__ == '__main__':
    main()
