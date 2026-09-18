"""聚類分析：找出 3 個指定ライザ樣本所在的群。"""
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering

E = np.load('tools/embeddings.npy')[1:]          # (9569, 192)
X = E / np.linalg.norm(E, axis=1, keepdims=True)
POS = [25, 30, 42]                               # 1-based 檔號 -> index-1

for K in [20, 40, 60]:
    km = KMeans(n_clusters=K, n_init=10, random_state=0).fit(X)
    labels = km.labels_
    pos_labels = set(labels[[p - 1 for p in POS]])
    print(f'\n=== K={K}：指定樣本落在 cluster {pos_labels} ===')
    for pl in sorted(pos_labels):
        idx = np.where(labels == pl)[0]
        pc = X[[p - 1 for p in POS]].mean(0); pc /= np.linalg.norm(pc)
        s = X[idx] @ pc
        print(f'  cluster {pl}: {len(idx)} 檔, 與指定 centroid 平均 cos={s.mean():.3f}')
        top = idx[np.argsort(-s)[:8]]
        print('    成員範例:', [f'{t+1:05d}({sv:.2f})' for t, sv in zip(top, np.sort(s)[::-1][:8])])

print('\n=== 指定樣本附近（檔 1-100）的分數 ===')
import csv
rows = {}
with open('voice_index.csv', newline='', encoding='utf-8-sig') as f:
    for r in csv.DictReader(f):
        rows[int(r['wav_file'].split('.')[0])] = r
P = X[[p - 1 for p in POS]].mean(0); P /= np.linalg.norm(P)
sc = X @ P
for i in range(1, 101):
    if sc[i - 1] > 0.80:
        print(f'  {i:05d}.wav {float(rows[i]["duration_seconds"]):>5}s cos={sc[i-1]:.4f}')
