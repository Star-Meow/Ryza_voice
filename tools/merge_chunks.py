"""合併 tools/chunks/*.npy -> tools/embeddings.npy（9570 x 192）。"""
import glob
import os
import numpy as np

N = 9569
E = np.zeros((N + 1, 192), dtype=np.float32)
covered = np.zeros(N + 1, dtype=bool)

for p in sorted(glob.glob('tools/chunks/*.npy')):
    start = int(os.path.basename(p).split('.')[0])
    arr = np.load(p)
    end = start + arr.shape[0]
    E[start:end] = arr
    covered[start:end] = True
    assert arr.shape == (min(end, N + 1) - start, 192), p

missing = np.where(~covered[1:])[0] + 1
print(f'chunks merged, covered {covered[1:].sum()}/{N}')
if len(missing):
    print('MISSING:', missing[:50], '...' if len(missing) > 50 else '')
else:
    np.save('tools/embeddings.npy', E)
    print('saved tools/embeddings.npy')
