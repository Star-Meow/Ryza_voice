"""迭代精煉：從 3 個種子出發，反覆用候選集 centroid 重排，檢查收斂。

內部一律使用 0-based index（檔號 = index + 1）。
"""
import numpy as np
import csv

E = np.load('tools/embeddings.npy')[1:]              # (9569,192)
X = E / np.linalg.norm(E, axis=1, keepdims=True)
SEED = [25, 30, 42]                                  # 檔號
SEED_I = [p - 1 for p in SEED]                       # 0-based

rows = {}
with open('voice_index.csv', newline='', encoding='utf-8-sig') as f:
    for r in csv.DictReader(f):
        rows[int(r['wav_file'].split('.')[0])] = r


def refine(N, iters=20):
    S = set(SEED_I)
    for it in range(iters):
        c = X[sorted(S)].mean(0); c /= np.linalg.norm(c)
        sc = X @ c
        top = set(np.argsort(-sc)[:N].tolist())
        if top == S:
            return S, it + 1, sc
        S = top
    c = X[sorted(S)].mean(0); c /= np.linalg.norm(c)
    sc = X @ c
    return S, iters, sc


for N in [300, 600, 1000, 1500, 2500]:
    S, iters, sc = refine(N)
    inseed = [p + 1 for p in SEED_I if p in S]
    srt = np.argsort(-sc)  # 0-based 降序
    lo_in = min(sc[p] for p in S)
    print(f'N={N:5d}: 收斂於 {iters:2d} 次迭代, 種子保留 {inseed}, '
          f'集合內最低 cos={lo_in:.4f}, 第{N}名/第{N+1}名: '
          f'{sc[srt[N-1]]:.4f}/{sc[srt[N]]:.4f}')
