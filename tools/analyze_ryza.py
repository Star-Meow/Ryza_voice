"""分析主庫分數：修正檔名對應、連續區塊結構、門檻建議。"""
import csv
import numpy as np

N = 9569

rows = {}
with open('voice_index.csv', newline='', encoding='utf-8-sig') as f:
    for r in csv.DictReader(f):
        rows[int(r['wav_file'].split('.')[0])] = r

E = np.load('tools/embeddings.npy')
pos_idx = [25, 30, 42]
P = E[pos_idx]                                   # (3,192)
P = P / np.linalg.norm(P, axis=1, keepdims=True)
centroid = P.mean(0); centroid /= np.linalg.norm(centroid)

scores = E[1:] @ centroid                        # scores[k] = 檔 k+1
np.save('tools/ryza_scores.npy', scores)

print('=== 3 個指定樣本互比（一致性）===')
for i, a in enumerate(pos_idx):
    print(f'  {a:05d} vs centroid: {scores[a-1]:.4f}   '
          f'vs 其他兩檔: {[round(float(P[i] @ P[j]),3) for j in range(3) if j != i]}')

order = np.argsort(-scores)
print('\n=== Top 50（正確對應）===')
for k in order[:50]:
    fn = k + 1
    r = rows[fn]
    print(f'{fn:05d}.wav cue={r["cue_index"]:>5} {float(r["duration_seconds"]):>6}s '
          f'sound_id={r["sound_id"]:>10} cos={scores[k]:.4f}')

print('\n=== 分數區間計數 ===')
for lo, hi in [(0.95, 1.01), (0.90, 0.95), (0.85, 0.90), (0.80, 0.85),
               (0.75, 0.80), (0.70, 0.75), (0.65, 0.70), (0.60, 0.65), (0, 0.60)]:
    print(f'  [{lo:.2f},{hi:.2f}): {((scores >= lo) & (scores < hi)).sum()}')

# --- 連續區塊分析：相鄰 cue 大多屬同一角色/場景 ---
print('\n=== 連續高分區塊（滑動窗口 W=11，平均分數 > 0.78）===')
W = 11
half = W // 2
sm = np.convolve(scores, np.ones(W) / W, mode='same')
inblock = sm > 0.78
blocks = []
s = 0
for i in range(1, N + 1):
    if i == N or inblock[i - 1] != inblock[s - 1]:
        if inblock[s - 1]:
            blocks.append((s, i - 1))
        s = i
tot = 0
for a, b in blocks:
    m = scores[a - 1:b].mean()
    tot += b - a + 1
    print(f'  cue {a:5d}-{b:5d} ({b-a+1:4d} 檔, 平均 {m:.3f}, 最高 {scores[a-1:b].max():.3f})')
print(f'  區塊內總檔數: {tot}')

# --- 個別參考一致性：同時與 3 個指定樣本都高才算 ---
print('\n=== 三向一致性（與 3 個指定樣本各自 cos 的最小值）===')
mins = np.min(E[1:] @ P.T, axis=1)
o2 = np.argsort(-mins)
for k in o2[:30]:
    print(f'  {k+1:05d}.wav min_cos={mins[k]:.4f} '
          f'({[round(float(E[k+1] @ P[j]),3) for j in range(3)]})')
np.save('tools/ryza_min3.npy', mins)
