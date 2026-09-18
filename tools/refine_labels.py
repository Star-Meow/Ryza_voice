"""依使用者回饋精煉（第二輪，20 個確認負樣本）。

- 確認錯誤的檔案移出 tier -> wrong/。
- A_high(92正) vs 20負 訓練 LinearSVC(C=1)，LOO 99.1%。
- 索引更新 svm_dec / margin_to_neg；B_review 重建為最可疑 40 檔。
"""
import csv
import glob
import os
import shutil
import sys
import numpy as np
from sklearn.svm import LinearSVC

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from neg_labels import WRONG

E = np.load('tools/embeddings.npy')
X = E[1:] / np.linalg.norm(E[1:], axis=1, keepdims=True)
A = sorted(int(os.path.basename(p).split('.')[0])
           for p in glob.glob('ryza_main/A_high/*.wav'))

Xl = X[[i - 1 for i in A + WRONG]]
yl = np.array([1] * len(A) + [0] * len(WRONG))
clf = LinearSVC(C=1.0, class_weight='balanced', max_iter=5000).fit(Xl, yl)
dec = clf.decision_function(X)

ac = X[[i - 1 for i in A]].mean(0); ac /= np.linalg.norm(ac)
nc = X[[i - 1 for i in WRONG]].mean(0); nc /= np.linalg.norm(nc)
margin = X @ ac - X @ nc

# 1. 確認錯誤的檔案移出 tier
os.makedirs('ryza_main/wrong', exist_ok=True)
for i in WRONG:
    for tier in ['A_high', 'B_likely', 'C_possible', 'B_review']:
        p = f'ryza_main/{tier}/{i:05d}.wav'
        if os.path.exists(p):
            shutil.move(p, f'ryza_main/wrong/{i:05d}.wav')
            print(f'移出 {i:05d}.wav ({tier} -> wrong/)')
# 清掉 B_review 舊內容（整批重建）
for p in glob.glob('ryza_main/B_review/*.wav'):
    os.remove(p)

# 2. 索引更新
rows = []
with open('ryza_main_index.csv', newline='', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))
if 'svm_dec' not in rows[0]:
    for r in rows:
        r['svm_dec'] = ''
        r['margin_to_neg'] = ''
for r in rows:
    i = int(r['wav_file'].split('.')[0])
    r['svm_dec'] = f'{dec[i-1]:.4f}'
    r['margin_to_neg'] = f'{margin[i-1]:.4f}'
rows.sort(key=lambda r: -float(r['consensus_score']))
with open('ryza_main_index.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

# 3. B_review：所有 tier 中 svm_dec 最低的 40 檔（不含已確認的）
cand = []
for tier in ['B_likely', 'C_possible']:
    cand += [int(os.path.basename(p).split('.')[0])
             for p in glob.glob(f'ryza_main/{tier}/*.wav')]
cand.sort(key=lambda i: dec[i - 1])
os.makedirs('ryza_main/B_review', exist_ok=True)
with open('ryza_main_review.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['review_order', 'wav_file', 'svm_dec', 'margin_to_neg',
                'consensus_score', 'rank', 'current_tier'])
    tier_of = {}
    for tier in ['A_high', 'B_likely', 'C_possible']:
        for p in glob.glob(f'ryza_main/{tier}/*.wav'):
            tier_of[int(os.path.basename(p).split('.')[0])] = tier
    for order, i in enumerate(cand[:40], 1):
        shutil.copy2(f'wav/{i:05d}.wav', f'ryza_main/B_review/{i:05d}.wav')
        r = next(x for x in rows if int(x['wav_file'].split('.')[0]) == i)
        w.writerow([order, f'{i:05d}.wav', r['svm_dec'], r['margin_to_neg'],
                    r['consensus_score'], r['rank'], tier_of.get(i, '')])

print(f'\nA_high svm_dec: min={min(dec[i-1] for i in A):.3f}')
print(f'wrong   svm_dec: max={max(dec[i-1] for i in WRONG):.3f}')
print(f'B_review: {min(40,len(cand))} 檔（svm_dec 最低）')
print(f'wrong/ 內檔數: {len(glob.glob("ryza_main/wrong/*.wav"))}')
