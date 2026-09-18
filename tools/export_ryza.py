"""將ライザ候選依置信度分級複製到 ryza_main/。"""
import csv
import shutil
import os

TIERS = [
    ('A_high', 0.80, 1.01),   # consensus >= 0.80
    ('B_likely', 0.76, 0.80), # 0.76-0.80
    ('C_possible', 0.72, 0.76),
]

os.makedirs('ryza_main', exist_ok=True)
stats = {}
with open('ryza_main_index.csv', newline='', encoding='utf-8') as f:
    for r in csv.DictReader(f):
        c = float(r['consensus_score'])
        for name, lo, hi in TIERS:
            if lo <= c < hi:
                d = f'ryza_main/{name}'
                os.makedirs(d, exist_ok=True)
                shutil.copy2(f'wav/{r["wav_file"]}', f'{d}/{r["wav_file"]}')
                stats[name] = stats.get(name, 0) + 1
                break

for name, lo, hi in TIERS:
    print(f'{name}: {stats.get(name, 0)} files')
print('total:', sum(stats.values()))
