"""事實層級的遺失稽核：基準版本的每個「事實片段」是否還出現在新版本的任何文件中。

check.py 只比對章節的去向（且只看目標檔）；本腳本改看整個新版本，
找出在所有現行 .md 中都找不到的數字、程式碼片段、檔名、hash。

輸出 .restructure/audit_missing.csv：一列一個遺失片段，附來源章節與 inventory 的處置，
供人工判定「刻意不保留」或「需要補回」。

用法：python -I .restructure/audit_facts.py
"""
import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / ".restructure"))
from build_inventory import norm, split_sections  # noqa: E402
from check import current_md_files, git  # noqa: E402

PATTERNS = [
    r"`([^`]{3,})`",                                            # 行內程式碼
    r"[\w\-/\\.]+\.(?:py|md|csv|list|json|yaml|wav|pth|ckpt|txt|npy|bin|tsv|safetensors)",  # 檔名
    r"\d[\d,]*\.\d+\s?%?|\d[\d,]{2,}\s?%?|\d+\s?%",              # 數字（小數、≥3 位、百分比）
    r"\b[0-9a-f]{7,40}\b",                                      # hash
]


def facts(body):
    out = []
    for pat in PATTERNS:
        for m in re.finditer(pat, body):
            raw = m.group(1) if m.groups() else m.group(0)
            k = norm(raw)
            if len(k) >= 3 and (k, raw) not in out:
                out.append((k, raw.strip()))
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    inv = {(r["src_file"], r["heading"]): r for r in
           csv.DictReader((ROOT / ".restructure/inventory.csv").open(encoding="utf-8-sig"))}
    corpus = "".join(norm((ROOT / p).read_text(encoding="utf-8")) for p in current_md_files())
    corpus += norm((ROOT / "LICENSE").read_text(encoding="utf-8"))
    base_files = [p for p in git("ls-tree", "-r", "--name-only", "docs-baseline").splitlines()
                  if p.endswith(".md") and not p.startswith("tools/vgmstream/")]
    rows, total = [], 0
    for path in base_files:
        text = git("show", f"docs-baseline:{path}")
        for lvl, head, a, b, body in split_sections(text):
            r = inv.get((path, head), {})
            for k, raw in facts(body):
                total += 1
                if k not in corpus:
                    rows.append({"src_file": path, "heading": head, "lines": f"{a}-{b}",
                                 "action": r.get("action", ""), "dest": r.get("dest", ""),
                                 "fact": raw[:120]})
    out = ROOT / ".restructure/audit_missing.csv"
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["src_file", "heading", "lines", "action", "dest", "fact"])
        w.writeheader()
        w.writerows(rows)
    print(f"事實片段 {total} 個；新版本中找不到 {len(rows)} 個 -> {out.relative_to(ROOT)}")
    by = {}
    for r in rows:
        by.setdefault(r["src_file"], 0)
        by[r["src_file"]] += 1
    for k, v in sorted(by.items(), key=lambda x: -x[1]):
        print(f"  {v:>4}  {k}")


if __name__ == "__main__":
    main()
