# -*- coding: utf-8 -*-
"""T0 實作：pytorch_model.bin -> model.safetensors（官方預訓練 BERT / HuBERT）。

不刪不改 .bin，只新增 model.safetensors；key 名稱不變；共享記憶體的張量先 clone 再存；
metadata 標註 pt 格式。安全載入模式（weights_only=True）為唯一路徑；若失敗則拋錯由呼叫端回報。
"""
import argparse
import hashlib
import os

import torch
from safetensors.torch import save_file


def sha256(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def load_state_dict_safe(path: str) -> dict:
    """優先（且唯一）使用安全載入模式讀 .bin。"""
    return torch.load(path, map_location="cpu", weights_only=True)


def detach_shared(state_dict: dict):
    """safetensors 不允許多個 key 共用儲存；以 data_ptr 偵測，第二個以後的 key clone 一份。

    clone 只複製數值，不改變 key 集合或任何數值內容。
    """
    seen, out, cloned = {}, {}, []
    for key, tensor in state_dict.items():
        t = tensor.detach()
        if not t.is_contiguous():
            t = t.contiguous()
        ptr = t.data_ptr()
        if ptr in seen:
            t = t.clone()
            cloned.append(key)
        else:
            seen[ptr] = key
        out[key] = t
    return out, cloned


def convert(bin_path: str, out_path: str) -> dict:
    sd = load_state_dict_safe(bin_path)
    tensors, cloned = detach_shared(sd)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    save_file(tensors, out_path, metadata={"format": "pt"})
    return {
        "bin_path": bin_path,
        "out_path": out_path,
        "num_keys": len(tensors),
        "cloned_shared_keys": cloned,
        "out_sha256": sha256(out_path),
        "out_size_bytes": os.path.getsize(out_path),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    result = convert(args.bin, args.out)
    for k, v in result.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
