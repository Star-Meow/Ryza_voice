"""比對 inventory.csv 與目前工作樹，確認每段原內容都到了該去的地方。

檢查項目：
  1. 覆蓋   ：每列都有合法 action；delete 有 reason；協作產物的 delete 附 GitHub 位置
  2. 去向   ：keep/edit 的 heading 仍在；move/merge 在 dest 找得到 heading 與原文關鍵字，且源已移除；
             delete 的源已移除；add 的 heading 已出現
  3. 反向   ：目前每個章節都能對回 inventory 的某一列（否則為來路不明的新內容）
  4. 連結   ：所有 .md 的相對連結目標存在；已整檔移除的檔名不再被引用

狀態：OK／PENDING（尚未執行）／PARTIAL（做了一半）／FAIL（違反規則）
只讀不寫。用法：python -I .restructure/check.py [-v]
"""
import csv
import hashlib
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INV = ROOT / ".restructure" / "inventory.csv"
BASELINE = "docs-baseline"
ACTIONS = {"keep", "edit", "move", "merge", "delete", "add"}
SKIP_PREFIX = ("tools/vgmstream/", ".restructure/")
# 整檔改名：源章節改到新檔名下查找
RENAMES = {"docs/02_issue_part2.md": "docs/02_model_selection.md"}

sys.path.insert(0, str(ROOT / ".restructure"))
from build_inventory import norm, split_sections  # noqa: E402


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                          check=True).stdout.decode("utf-8")


def current_md_files():
    out = git("ls-files", "--cached", "--others", "--exclude-standard", "*.md")
    return sorted(p for p in out.splitlines()
                  if not p.startswith(SKIP_PREFIX) and (ROOT / p).exists())


def load_current():
    """{file: {norm_heading: heading}}, {file: norm 全文}；並記錄各章節內文 hash"""
    heads, texts = {}, {}
    for p in current_md_files():
        text = (ROOT / p).read_text(encoding="utf-8")
        texts[p] = norm(text)
        secs = split_sections(text)
        heads[p] = {norm(h): h for _, h, *_ in secs if h != "(前言)"}
        for _, h, _, _, body in secs:
            HASHES[(p, norm(h))] = hashlib.sha1(norm(body).encode()).hexdigest()[:10]
    return heads, texts


def split_dest(dest):
    f, _, h = dest.partition("#")
    return f, h


KEY_PATTERNS = [
    r"`([^`]{4,})`",                                   # 行內程式碼
    r"[\w\-/]+\.(?:py|md|csv|list|json|yaml|wav|pth|ckpt|txt)",  # 檔名
    r"\d+\.\d+%?|\d{2,}%",                             # 小數、百分比
    r"\b[0-9a-f]{7,40}\b",                             # commit hash
]


def keys_of(body):
    found = []
    for pat in KEY_PATTERNS:
        for m in re.finditer(pat, body):
            k = norm(m.group(1) if m.groups() else m.group(0))
            if len(k) >= 4 and k not in found:
                found.append(k)
    return found


_baseline_cache = {}
HASHES = {}


def baseline_body(row):
    path = row["src_file"]
    if path not in _baseline_cache:
        _baseline_cache[path] = git("show", f"{BASELINE}:{path}").splitlines()
    a, b = map(int, row["lines"].split("-"))
    return "\n".join(_baseline_cache[path][a - 1:b])


def judge(row, heads, texts):
    act, head = row["action"], row["heading"]
    src = RENAMES.get(row["src_file"], row["src_file"])
    dest = row["dest"] or (f"{src}#{head}" if src else "")
    dfile, dhead = split_dest(dest)
    src_present = src in heads and norm(head) in heads[src]
    dest_present = dfile in heads and norm(dhead) in heads[dfile]

    if act not in ACTIONS:
        return "FAIL", f"未知 action：{act!r}"
    if act == "delete" and not row["reason"].strip():
        return "FAIL", "delete 未寫 reason"
    if act in {"move", "merge", "add"} and not row["dest"]:
        return "FAIL", f"{act} 未寫 dest"
    if row["category"] == "協作產物" and act == "delete" and "github.com" not in row["reason"]:
        return "FAIL", "協作產物刪除未附 GitHub 位置"

    if act == "keep":
        if not src_present:
            return "FAIL", "keep 的章節不見了"
        if HASHES.get((src, norm(head))) != row["hash"]:
            return "PARTIAL", "標為 keep 但內容已變動；確認後請把 action 改為 edit"
        return "OK", ""
    if act == "edit":
        renamed = norm(dhead) != norm(head) or dfile != src
        if dest_present:
            if renamed and src_present and dfile == src:
                return "PARTIAL", "新標題已出現但舊標題仍在"
            if not renamed and HASHES.get((dfile, norm(dhead))) == row["hash"]:
                return "PENDING", ""
            return "OK", ""
        if src_present:
            return "PENDING", ""
        return "FAIL", f"找不到章節（預期 {dest}）"
    if act == "add":
        return ("OK", "") if dest_present else ("PENDING", "")
    if act == "delete":
        return ("PENDING", "") if src_present else ("OK", "")
    # move / merge
    keys = keys_of(baseline_body(row))
    hit = [k for k in keys if k in texts.get(dfile, "")]
    ratio = f"關鍵字 {len(hit)}/{len(keys)}" if keys else "無關鍵字可比"
    if src_present and not (dfile == src and norm(dhead) == norm(head)):
        # 源還在：尚未搬移（dest 若已有部分關鍵字，代表原本就重複）
        return "PENDING", f"dest 現有{ratio}" if dest_present else ""
    if not dest_present:
        return "FAIL", f"源已移除但 {dest} 不存在"
    if keys and len(hit) * 2 < len(keys):
        if "人工確認：" in row["reason"]:
            return "OK", f"{ratio}（已人工確認）"
        return "PARTIAL", f"源已移除，但 dest 僅{ratio}，請人工確認是否遺漏"
    return "OK", ratio


def reverse_check(rows, heads):
    mapped = defaultdict(set)
    for r in rows:
        if r["action"] in {"keep", "edit"}:
            f, h = split_dest(r["dest"] or f"{r['src_file']}#{r['heading']}")
            mapped[f].add(norm(h))
            mapped[RENAMES.get(r["src_file"], r["src_file"])].add(norm(r["heading"]))  # edit 尚未改標題時
        elif r["action"] == "delete":
            mapped[RENAMES.get(r["src_file"], r["src_file"])].add(norm(r["heading"]))  # 尚未刪除時
        elif r["action"] in {"move", "merge", "add"}:
            f, h = split_dest(r["dest"])
            mapped[f].add(norm(h))
            if r["action"] != "add":
                mapped[r["src_file"]].add(norm(r["heading"]))  # 尚未搬移時
    out = []
    for f, hs in heads.items():
        for nh, h in hs.items():
            if nh not in mapped.get(f, set()):
                out.append(f"{f}#{h}")
    return out


LINK = re.compile(r"\]\(([^)\s]+)\)")


def link_check(rows):
    broken = []
    for p in current_md_files():
        text = (ROOT / p).read_text(encoding="utf-8")
        for m in LINK.finditer(text):
            tgt = m.group(1).split("#")[0]
            if not tgt or re.match(r"^[a-z]+:", tgt):
                continue
            if not (ROOT / p).parent.joinpath(tgt).resolve().exists():
                broken.append(f"{p} → {m.group(1)}")
    # 整檔移出的來源，不應再被引用
    by_file = defaultdict(set)
    for r in rows:
        if r["src_file"]:
            by_file[r["src_file"]].add(r["action"])
    gone = [f for f, acts in by_file.items() if not acts & {"keep", "edit"}]
    stale = []
    for p in current_md_files():
        text = (ROOT / p).read_text(encoding="utf-8")
        for f in gone:
            if p != f and Path(f).name in text:
                stale.append(f"{p} 仍提到 {Path(f).name}")
    return broken, stale


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    verbose = "-v" in sys.argv
    rows = list(csv.DictReader(INV.open(encoding="utf-8-sig")))
    heads, texts = load_current()

    status = Counter()
    per_file = defaultdict(Counter)
    problems = []
    for r in rows:
        st, msg = judge(r, heads, texts)
        status[st] += 1
        per_file[r["src_file"] or "(新增)"][st] += 1
        if st in {"FAIL", "PARTIAL"} or verbose or (st == "PENDING" and r["action"] in {"move", "merge"}):
            problems.append(f"  [{st}] {r['id']} {r['action']:<6} {r['src_file']}#{r['heading']}"
                            f" → {r['dest'] or '-'}  {msg}")

    print(f"inventory {len(rows)} 列：" + "、".join(f"{k} {v}" for k, v in sorted(status.items())))
    print("\n各檔進度（OK/PENDING/PARTIAL/FAIL）：")
    for f, c in per_file.items():
        print(f"  {f:<36} {c['OK']:>3}/{c['PENDING']:>3}/{c['PARTIAL']:>3}/{c['FAIL']:>3}")
    if problems:
        print("\n需處理：")
        print("\n".join(problems))

    unmapped = reverse_check(rows, heads)
    print(f"\n反向檢查：{len(unmapped)} 個章節對不回 inventory")
    for u in unmapped:
        print(f"  [UNMAPPED] {u}")

    broken, stale = link_check(rows)
    print(f"\n連結檢查：斷鏈 {len(broken)}、引用已移除檔 {len(stale)}")
    for b in broken:
        print(f"  [BROKEN] {b}")
    for s in stale:
        print(f"  [STALE] {s}")

    bad = status["FAIL"] + len(unmapped) + len(broken)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
