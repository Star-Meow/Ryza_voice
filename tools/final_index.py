"""產出最終ライザ候選索引。

錨點策略：
- 00030/00042（乾淨對話錄音，互比 0.769）為主錨點，consensus = 兩者 cos 平均。
- 00025（背景噪音較多、嵌入偏離）為次要參考，單獨列出。
- 戰鬥庫各角色 centroid 分數列為參考（已知跨條件不可靠）。
"""
import csv
import glob
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spk_model

E = np.load('tools/embeddings.npy')                # (9570,192)，index = 檔號
X = E[1:] / np.linalg.norm(E[1:], axis=1, keepdims=True)

a30, a42, a25 = 30, 42, 25
cos30 = X @ E[a30]; cos42 = X @ E[a42]; cos25 = X @ E[a25]
cons = (cos30 + cos42) / 2
mins = np.min(np.stack([cos30, cos42, cos25]), axis=0)

rows = {}
with open('voice_index.csv', newline='', encoding='utf-8-sig') as f:
    for r in csv.DictReader(f):
        rows[int(r['wav_file'].split('.')[0])] = r

# 戰鬥庫 centroid（參考用）
C = np.load('tools/battle_centroids.npy')
names = open('tools/battle_char_names.txt').read().split('\n')
battle = X @ C.T                                   # (9569, 6)

order = np.argsort(-cons)
print(f'=== consensus 分數分布 ===')
for q in [50, 75, 90, 95, 98, 99, 99.5, 99.9]:
    print(f'  p{q}: {np.percentile(cons, q):.4f}')
print(f'  種子: 00025={cons[24]:.4f} 00030={cons[29]:.4f} 00042={cons[41]:.4f}')
print(f'  排名: 00025={int(np.where(order==24)[0][0])+1} 00030={int(np.where(order==29)[0][0])+1} 00042={int(np.where(order==41)[0][0])+1}')

print('\n=== 門檻計數（consensus）===')
for t in [0.85, 0.82, 0.80, 0.78, 0.76, 0.74, 0.72, 0.70]:
    print(f'  >= {t}: {(cons >= t).sum()}')

print('\n=== Top 30（consensus）===')
for k in order[:30]:
    fn = k + 1
    r = rows[fn]
    bc = battle[k]
    print(f'{fn:05d}.wav cue={r["cue_index"]:>5} {float(r["duration_seconds"]):>6}s '
          f'cons={cons[k]:.4f} (30:{cos30[k]:.3f} 42:{cos42[k]:.3f} 25:{cos25[k]:.3f}) '
          f'battle_best={names[np.argmax(bc)]}')

# 寫完整索引
with open('ryza_main_index.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['rank', 'wav_file', 'cue_index', 'sound_id', 'duration_seconds',
                'consensus_score', 'cos_to_00030', 'cos_to_00042', 'cos_to_00025',
                'min3', 'battle_best_guess', 'battle_ryza_score'])
    for rank, k in enumerate(order, 1):
        fn = k + 1
        r = rows[fn]
        w.writerow([rank, f'{fn:05d}.wav', r['cue_index'], r['sound_id'],
                    r['duration_seconds'], f'{cons[k]:.4f}',
                    f'{cos30[k]:.4f}', f'{cos42[k]:.4f}', f'{cos25[k]:.4f}',
                    f'{mins[k]:.4f}', names[np.argmax(battle[k])],
                    f'{battle[k][names.index("RYZA")]:.4f}'])
print('\nsaved ryza_main_index.csv')
