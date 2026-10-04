# -*- coding: utf-8 -*-
"""T0 獨立驗證：只依賴 input (.bin) 與 output (model.safetensors) 兩個外部檔案，
不讀取轉檔實作的內部狀態。同時提供 --negative 模式，故意製造損壞檔以證明本驗證能辨識失敗。

檢查項：
  V1 key 保留：safetensors 的 key 集合、每個 key 的 dtype/shape 與 .bin 完全一致
  V2 乾淨載入：from_pretrained 載入目錄成功，missing/unexpected/mismatched keys 與 error_msgs 全空
  V3 數值等價：固定輸入下，from_pretrained(新檔) 與 from_config+手動載 .bin 的輸出最大絕對差 <= 1e-6
  V4 .bin 完整性：.bin 的 SHA256 與轉檔前記錄值相同（未遭修改）
"""
import argparse
import hashlib
import json
import os
import sys

import torch

PRE = r"H:\git\GPT-SoVITS\GPT_SoVITS\pretrained_models"
MODELS = {
    "roberta": {
        "dir": os.path.join(PRE, "chinese-roberta-wwm-ext-large"),
        "bin": os.path.join(PRE, "chinese-roberta-wwm-ext-large", "pytorch_model.bin"),
        "bin_sha256": "e53a693acc59ace251d143d068096ae0d7b79e4b1b503fa84c9dcf576448c1d8",
    },
    "hubert": {
        "dir": os.path.join(PRE, "chinese-hubert-base"),
        "bin": os.path.join(PRE, "chinese-hubert-base", "pytorch_model.bin"),
        "bin_sha256": "24164f129c66499d1346e2aa55f183250c223161ec2770c0da3d3b08cf432d3c",
    },
}
THRESHOLD = 1e-6


def sha256(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def check_keyset(bin_path: str, st_path: str) -> dict:
    from safetensors import safe_open

    sd_bin = torch.load(bin_path, map_location="cpu", weights_only=True)
    with safe_open(st_path, framework="pt", device="cpu") as f:
        st_keys = list(f.keys())
        meta = f.metadata()
    kb, ks = set(sd_bin), set(st_keys)
    dtype_mismatch, shape_mismatch = [], []
    with safe_open(st_path, framework="pt", device="cpu") as f:
        for k in sorted(kb & ks):
            t = f.get_tensor(k)
            if t.dtype != sd_bin[k].dtype:
                dtype_mismatch.append((k, str(sd_bin[k].dtype), str(t.dtype)))
            if tuple(t.shape) != tuple(sd_bin[k].shape):
                shape_mismatch.append((k, list(sd_bin[k].shape), list(t.shape)))
    ok = not kb ^ ks and not dtype_mismatch and not shape_mismatch
    return {
        "pass": ok,
        "only_in_bin": sorted(kb - ks),
        "only_in_st": sorted(ks - kb),
        "dtype_mismatch": dtype_mismatch,
        "shape_mismatch": shape_mismatch,
        "num_keys_bin": len(kb),
        "num_keys_st": len(ks),
        "safetensors_metadata": meta,
    }


def _loaders(kind: str):
    if kind == "roberta":
        from transformers import AutoModelForMaskedLM

        return AutoModelForMaskedLM
    from transformers import HubertModel

    return HubertModel


def check_from_pretrained(kind: str, model_dir: str) -> dict:
    """V2：依各腳本實際用法載入目錄（roberta: AutoModelForMaskedLM；hubert: HubertModel）。"""
    from transformers import logging as tf_logging

    tf_logging.set_verbosity_warning()
    load_cls = _loaders(kind)
    with __import__("warnings").catch_warnings(record=True) as wlog:
        __import__("warnings").simplefilter("always")
        if kind == "roberta":
            model, info = load_cls.from_pretrained(model_dir, output_loading_info=True)
        else:
            model, info = load_cls.from_pretrained(model_dir, local_files_only=True, output_loading_info=True)
    keys = ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    ok = all(len(info[k]) == 0 for k in keys)
    return {
        "pass": ok,
        "missing_keys": info["missing_keys"],
        "unexpected_keys": info["unexpected_keys"],
        "mismatched_keys": [str(x) for x in info["mismatched_keys"]],
        "error_msgs": [str(x) for x in info["error_msgs"]],
        "warnings": [str(w.message)[:200] for w in wlog],
        "model_class": type(model).__name__,
        "num_params": sum(p.numel() for p in model.parameters()),
    }


def check_numerical(kind: str, model_dir: str, bin_path: str) -> dict:
    """V3：固定輸入下比較兩種載入路徑的輸出。"""
    load_cls = _loaders(kind)
    if kind == "roberta":
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model_a = AutoModelForMaskedLM.from_pretrained(model_dir).eval()
        cfg = model_a.config
        model_b = AutoModelForMaskedLM.from_config(cfg).eval()
        sd_bin = torch.load(bin_path, map_location="cpu", weights_only=True)
        load_info_b = model_b.load_state_dict(sd_bin, strict=False)
        inputs = tokenizer("テスト用の文章です。", return_tensors="pt")
        with torch.no_grad():
            out_a = model_a(**inputs).logits
            out_b = model_b(**inputs).logits
    else:
        from transformers import HubertModel, Wav2Vec2FeatureExtractor

        fe = Wav2Vec2FeatureExtractor.from_pretrained(model_dir, local_files_only=True)
        model_a = HubertModel.from_pretrained(model_dir, local_files_only=True).eval()
        model_b = HubertModel(model_a.config).eval()
        sd_bin = torch.load(bin_path, map_location="cpu", weights_only=True)
        load_info_b = model_b.load_state_dict(sd_bin, strict=False)
        torch.manual_seed(2024)
        wav = torch.randn(1, 16000).numpy()
        inputs = fe(wav, sampling_rate=16000, return_tensors="pt").input_values
        with torch.no_grad():
            out_a = model_a(inputs).last_hidden_state
            out_b = model_b(inputs).last_hidden_state
    diff = (out_a.float() - out_b.float()).abs()
    max_abs = float(diff.max().item())
    scale = max(float(out_a.abs().max().item()), float(out_b.abs().max().item()))
    return {
        "pass": max_abs <= THRESHOLD,
        "max_abs_diff": max_abs,
        "output_abs_max": scale,
        "threshold": THRESHOLD,
        "out_shape": list(out_a.shape),
        "load_info_b_missing": sorted(load_info_b.missing_keys),
        "load_info_b_unexpected": sorted(load_info_b.unexpected_keys),
    }


def validate(kind: str, st_path: str) -> dict:
    m = MODELS[kind]
    st_dir = os.path.dirname(os.path.abspath(st_path))
    v1 = check_keyset(m["bin"], st_path)
    # V2 以 st 所在目錄為單位測試（from_pretrained 讀的是目錄）
    v2 = check_from_pretrained(kind, st_dir)
    v3 = check_numerical(kind, st_dir, m["bin"])
    v4 = {"pass": sha256(m["bin"]) == m["bin_sha256"], "bin_sha256": sha256(m["bin"])}
    return {
        "kind": kind,
        "st_path": st_path,
        "V1_keyset": v1,
        "V2_clean_load": v2,
        "V3_numerical": v3,
        "V4_bin_integrity": v4,
        "overall_pass": v1["pass"] and v2["pass"] and v3["pass"] and v4["pass"],
    }


def make_broken(src_st: str, dst_st: str, mode: str) -> None:
    """negative fixture：故意製造損壞的 safetensors。"""
    from safetensors import safe_open
    from safetensors.torch import save_file

    with safe_open(src_st, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        meta = f.metadata()
        tensors = {k: f.get_tensor(k) for k in keys}
    if mode == "perturb":
        # 改動一個權重（數值偏移 1.0）
        target = next(k for k in keys if "weight" in k and tensors[k].numel() > 1)
        t = tensors[target].clone()
        t.add_(1.0)
        tensors[target] = t
    elif mode == "rename":
        # 移除一個 key 並以改名 key 補回
        target = next(k for k in keys if "weight" in k)
        tensors[target + "_BROKEN"] = tensors.pop(target).clone()
    else:
        raise ValueError(mode)
    os.makedirs(os.path.dirname(os.path.abspath(dst_st)), exist_ok=True)
    save_file(tensors, dst_st, metadata=meta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=list(MODELS))
    ap.add_argument("--st", required=True, help="要驗證的 model.safetensors 路徑（所在目錄需含 config.json 等）")
    ap.add_argument("--negative", action="store_true", help="先製造損壞檔並驗證其必須失敗")
    ap.add_argument("--neg-dir", default=None, help="損壞暫存檔目錄")
    args = ap.parse_args()

    report = {"kind": args.kind, "target_st": args.st}
    if args.negative:
        import shutil

        src_dir = MODELS[args.kind]["dir"]
        results = {}
        for mode in ("perturb", "rename"):
            neg_dir = os.path.join(args.neg_dir or os.path.dirname(os.path.abspath(args.st)), f"_broken_{mode}")
            os.makedirs(neg_dir, exist_ok=True)
            for fname in os.listdir(src_dir):
                if fname.endswith((".json", ".txt", ".model")) and not fname.endswith("safetensors"):
                    shutil.copy2(os.path.join(src_dir, fname), os.path.join(neg_dir, fname))
            broken = os.path.join(neg_dir, "model.safetensors")
            make_broken(args.st, broken, mode)
            r = validate(args.kind, broken)
            results[mode] = {
                "overall_pass": r["overall_pass"],
                "V1_pass": r["V1_keyset"]["pass"],
                "V2_pass": r["V2_clean_load"]["pass"],
                "V3_pass": r["V3_numerical"]["pass"],
                "V3_max_abs_diff": r["V3_numerical"]["max_abs_diff"],
                "V1_only_in_st": r["V1_keyset"]["only_in_st"][:3],
                "V1_only_in_bin": r["V1_keyset"]["only_in_bin"][:3],
            }
            shutil.rmtree(neg_dir, ignore_errors=True)
        report["negative"] = {
            "expect_all_fail": True,
            "results": results,
            "detects_failure": all(not v["overall_pass"] for v in results.values()),
        }
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    r = validate(args.kind, args.st)
    report.update(r)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    sys.exit(0 if r["overall_pass"] else 1)


if __name__ == "__main__":
    main()
