"""編碼一段檔案 [start, end) 到獨立 .npy（供分段子進程策略使用）。

用法: python encode_chunk.py START END OUT.npy [cpu|cuda]
"""
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spk_model


def main():
    start, end, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    spk_model.get_model()
    E = np.zeros((end - start, 192), dtype=np.float32)
    t0 = time.time()
    for j, i in enumerate(range(start, end)):
        E[j] = spk_model.embed(f'wav/{i:05d}.wav')
    np.save(out, E)
    print(f'chunk {start}-{end} done in {time.time()-t0:.1f}s ({spk_model.DEVICE})', flush=True)


if __name__ == '__main__':
    main()
