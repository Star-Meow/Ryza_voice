"""批次編碼主庫 9569 個 wav -> embeddings.npy（GPU 循序）。

GPU 上多執行緒並發 CUDA 呼叫會造成 illegal memory access，故循序處理；
單檔約 46ms，全程約 7-8 分鐘。
"""
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spk_model

N = 9569
OUT = os.path.join('tools', 'embeddings.npy')


def work(i):
    try:
        return i, spk_model.embed(f'wav/{i:05d}.wav')
    except Exception as e:
        print(f'ERR {i:05d}.wav: {e}', flush=True)
        return i, None


if __name__ == '__main__':
    spk_model.get_model()
    E = np.zeros((N + 1, 192), dtype=np.float32)
    errs = []
    t0 = time.time()
    for i in range(1, N + 1):
        r = work(i)
        if r[1] is None:
            errs.append(i)
        else:
            E[r[0]] = r[1]
        if i % 500 == 0:
            np.save(OUT, E)
            el = time.time() - t0
            print(f'{i}/{N}  {el/60:.1f}min  ETA {el/i*(N-i)/60:.1f}min', flush=True)
    np.save(OUT, E)
    print(f'DONE {N} in {(time.time()-t0)/60:.1f} min -> {OUT}')
    if errs:
        print(f'ERRORS ({len(errs)}):', errs)
