"""跨條件角色辨識測試：用戰鬥庫各角色 centroid 分類，驗證 3 個種子。

1. leave-one-out：戰鬥庫各角色樣本能否被正確分到自己的 centroid。
2. 3 個指定種子（主庫對話音）是否判為 RYZA。
"""
import glob
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spk_model

CHARS = ['RYZA', 'KLAU', 'ANCIENT', 'LENT', 'TAO', 'PAT']

groups = {}
for p in sorted(glob.glob('battle_voice/*.wav')):
    base = os.path.basename(p)
    for name in CHARS:
        if f'_{name}_' in base:
            groups.setdefault(name, []).append(p)
            break

cent = {}
for name, files in groups.items():
    embs = np.stack([spk_model.embed(f) for f in files])
    c = embs.mean(0); c /= np.linalg.norm(c)
    cent[name] = c
    print(f'{name:9s} {len(files):3d} 檔 -> centroid')

names = CHARS
C = np.stack([cent[n] for n in names])

# 1. leave-one-out（用「其他角色 + 自身其他樣本」的 centroid 分類）
print('\n=== leave-one-out 角色分類混淆 ===')
conf = {n: {m: 0 for m in names} for n in names}
for name, files in groups.items():
    for p in files:
        e = spk_model.embed(p)
        # 重算排除此樣本的 centroid
        others = [spk_model.embed(f) for f in files if f != p]
        c = np.mean(others, axis=0); c /= np.linalg.norm(c)
        sims = {n2: float(c @ cent[n2]) for n2 in names}
        best = max(sims, key=sims.get)
        conf[name][best] += 1
for n in names:
    tot = sum(conf[n].values())
    print(f'  {n:9s} -> ' + ' '.join(f'{m}={conf[n][m]/tot*100:.0f}%' for m in names))

# 2. 3 個種子
print('\n=== 3 個指定種子的角色分數 ===')
for i in [25, 30, 42]:
    e = spk_model.embed(f'wav/{i:05d}.wav')
    sims = {n: float(e @ cent[n]) for n in names}
    best = max(sims, key=sims.get)
    print(f'  {i:05d}.wav -> ' + ' '.join(f'{n}={v:.3f}' for n, v in sims.items()) + f'  => {best}')

np.save('tools/battle_centroids.npy', C)
with open('tools/battle_char_names.txt', 'w') as f:
    f.write('\n'.join(names))
