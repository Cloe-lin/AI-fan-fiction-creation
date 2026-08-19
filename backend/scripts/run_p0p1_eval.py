# -*- coding: utf-8 -*-
"""P0/P1 自动测评：黄金集质量 + 挑战集拦截。"""
from __future__ import annotations

import csv
import json
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVAL_CSV = ROOT / "docs" / "评测集-扩展版.csv"
OUT_CSV = ROOT / "docs" / "评测跑分结果-P0P1.csv"
OUT_MD = ROOT / "docs" / "评测报告-P0P1.md"
CREDS = ROOT / "backend" / "app" / "data" / "credentials_initial.txt"
API = "http://127.0.0.1:8010/api"

# P0/P1 抽样：覆盖多书 + 关键失败模式（控制时长与费用）
GOLD_IDS = [
    "GOLD-MDZS-101",
    "GOLD-MDZS-102",
    "GOLD-MDZS-106",
    "GOLD-MDZS-109",
    "GOLD-SYZ-101",
    "GOLD-SYZ-103",
    "GOLD-XMG-101",
    "GOLD-QQGK-101",
    "GOLD-MM-101",
    "GOLD-EDGE-001",
]
CHAL_IDS = [
    "CHAL-MDZS-101",  # 中间篇-求学
    "CHAL-MDZS-103",  # OOC
    "CHAL-MDZS-105",  # 偏离设定
    "CHAL-MDZS-106",  # 禁忌
    "CHAL-SYZ-101",  # 非结局后
    "CHAL-EDGE-001",  # 世界观崩
]


def http_json(method: str, url: str, body=None, headers=None, timeout=600):
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            return resp.status, json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
        try:
            payload = json.loads(raw)
        except Exception:
            payload = {"detail": raw[:500]}
        return e.code, payload


def login() -> str:
    text = CREDS.read_text(encoding="utf-8")
    m = re.search(r"^admin\s+(\S+)", text, re.M)
    if not m:
        m = re.search(r"^user01\s+(\S+)", text, re.M)
        user, pw = "user01", m.group(1)
    else:
        user, pw = "admin", m.group(1)
    code, data = http_json("POST", f"{API}/auth/login", {"username": user, "password": pw}, timeout=30)
    if code != 200:
        raise RuntimeError(f"login failed: {code} {data}")
    print(f"[login] ok as {user}")
    return data["access_token"]


def load_cases():
    with EVAL_CSV.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    by_id = {r["case_id"]: r for r in rows}
    selected = []
    for cid in GOLD_IDS + CHAL_IDS:
        if cid not in by_id:
            print(f"[warn] missing {cid}")
            continue
        selected.append(by_id[cid])
    return selected


def parse_chars(s: str) -> list[str]:
    s = (s or "").strip().strip('"')
    if not s:
        return []
    return [c.strip() for c in s.split(",") if c.strip()]


def issue_types(report: dict) -> list[str]:
    return [i.get("issue_type", "") for i in (report or {}).get("issues") or []]


def challenge_caught(report: dict, category: str) -> bool:
    """挑战集：认为「拦住」= 未通过 或 分<60 或 命中关键 issue。"""
    if not report:
        return False
    score = float(report.get("score") or 0)
    passed = bool(report.get("passed"))
    types = set(issue_types(report))
    hard = {"mid_chapter_fill", "canon_violation", "ooc", "direction_mismatch"}
    if score < 60 or not passed:
        return True
    if types & hard:
        return True
    # 禁忌/OOC 类也可能只给 warning
    if category in {"中间篇", "OOC", "禁忌", "偏离设定", "世界观崩", "续写攻击"} and types:
        return True
    return False


def gold_pass_auto(report: dict, min_score: float) -> bool:
    if not report:
        return False
    score = float(report.get("score") or 0)
    types = set(issue_types(report))
    if "mid_chapter_fill" in types:
        return False
    if score < min_score:
        return False
    return True


def run_one(token: str, case: dict) -> dict:
    chars = parse_chars(case["characters"])
    scenario = case["user_scenario"]
    prior = (case.get("prior_context") or "").strip()
    notes = ""
    if prior:
        notes = f"【前文记忆】\n{prior}\n请自然承接，不要失忆重开。"
        scenario = f"{scenario}\n\n（续写约束）必须承接上文。"

    length = case.get("length") or "short"
    # 测评加速：非 short 也尽量用 short，但保留 medium 若干条测质量
    if case["case_id"] not in {"GOLD-MDZS-102", "GOLD-SYZ-101", "CHAL-MDZS-101"}:
        length = "short"

    body = {
        "novel_id": case["novel_id"],
        "title": f"[EVAL]{case['case_id']}",
        "characters": chars,
        "scenario": scenario,
        "tone": case.get("tone") or "warm",
        "perspective": "",
        "length": length,
        "additional_notes": notes,
        "auto_save": False,
    }
    t0 = time.time()
    code, data = http_json(
        "POST",
        f"{API}/stories/create",
        body,
        headers={"Authorization": f"Bearer {token}"},
        timeout=600,
    )
    elapsed = round(time.time() - t0, 1)
    row = {
        **{k: case.get(k, "") for k in case.keys()},
        "run_date": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
        "prompt_version": "baseline_v0.7",
        "http_status": code,
        "elapsed_sec": elapsed,
        "auto_score": "",
        "auto_passed": "",
        "auto_issues": "",
        "gate_pass": "",
        "caught": "",
        "error": "",
        "content_preview": "",
    }
    if code != 200:
        row["error"] = json.dumps(data, ensure_ascii=False)[:400]
        row["gate_pass"] = "false"
        row["caught"] = "false"
        print(f"  FAIL http={code} {case['case_id']} {elapsed}s")
        return row

    report = data.get("consistency_report") or {}
    score = report.get("score")
    row["auto_score"] = score
    row["auto_passed"] = report.get("passed")
    issues = report.get("issues") or []
    row["auto_issues"] = "; ".join(
        f"{i.get('issue_type')}:{i.get('severity')}:{i.get('description','')[:40]}" for i in issues
    )
    preview = (data.get("content") or "")[:120].replace("\n", " ")
    row["content_preview"] = preview

    expect_block = str(case.get("expect_block", "")).lower() == "true"
    min_score = float(case["expect_auto_score_min"]) if case.get("expect_auto_score_min") else 60.0

    if expect_block:
        caught = challenge_caught(report, case.get("category", ""))
        row["caught"] = caught
        row["gate_pass"] = caught  # 挑战：拦住=通过测评
    else:
        ok = gold_pass_auto(report, min_score)
        row["gate_pass"] = ok
        row["caught"] = ""

    print(
        f"  ok {case['case_id']} score={score} passed={report.get('passed')} "
        f"gate={row['gate_pass']} {elapsed}s"
    )
    return row


def summarize(rows: list[dict]) -> str:
    gold = [r for r in rows if r.get("set_type") == "gold"]
    chal = [r for r in rows if r.get("set_type") == "challenge"]

    def fbool(v):
        return str(v).lower() in {"true", "1", "yes"}

    gold_scores = [float(r["auto_score"]) for r in gold if r.get("auto_score") not in ("", None)]
    gold_pass = sum(1 for r in gold if fbool(r.get("gate_pass")))
    chal_catch = sum(1 for r in chal if fbool(r.get("caught")))
    revise = sum(1 for r in gold_scores if r < 60)

    ooc_n = sum(1 for r in rows if "ooc" in (r.get("auto_issues") or ""))
    mid_n = sum(1 for r in rows if "mid_chapter_fill" in (r.get("auto_issues") or ""))
    dir_n = sum(1 for r in rows if "direction_mismatch" in (r.get("auto_issues") or ""))

    avg = sum(gold_scores) / len(gold_scores) if gold_scores else 0
    gold_rate = gold_pass / len(gold) if gold else 0
    chal_rate = chal_catch / len(chal) if chal else 0
    revise_rate = revise / len(gold_scores) if gold_scores else 0

    lines = [
        "# P0 / P1 测评报告",
        "",
        f"- 跑分时间：{datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %z')}",
        f"- Prompt 版本：baseline_v0.7",
        f"- API：{API}",
        f"- 样本：黄金 {len(gold)} + 挑战 {len(chal)}",
        "",
        "## P0 指标",
        "",
        f"| 指标 | 结果 | 目标 | 判定 |",
        f"|------|------|------|------|",
        f"| 黄金集自动均分 | {avg:.1f} | ≥ 60（校准前参考）/ 理想≥70 | {'通过' if avg >= 60 else '未通过'} |",
        f"| 黄金集门禁通过率 | {gold_pass}/{len(gold)} = {gold_rate:.0%} | ≥ 70%（首轮基线） | {'通过' if gold_rate >= 0.7 else '未通过'} |",
        f"| 挑战集拦截率 | {chal_catch}/{len(chal)} = {chal_rate:.0%} | ≥ 90% | {'通过' if chal_rate >= 0.9 else '未通过'} |",
        f"| 修订触发率（黄金 score&lt;60） | {revise}/{len(gold_scores)} = {revise_rate:.0%} | 观察项 | — |",
        "",
        "## P1 指标",
        "",
        f"| 指标 | 结果 |",
        f"|------|------|",
        f"| OOC issue 出现次数 | {ooc_n} |",
        f"| mid_chapter_fill 出现次数 | {mid_n} |",
        f"| direction_mismatch 出现次数 | {dir_n} |",
        "",
        "## 逐条结果",
        "",
        "| case_id | set | score | passed | gate/caught | issues | sec |",
        "|---------|-----|-------|--------|-------------|--------|-----|",
    ]
    for r in rows:
        gc = r.get("gate_pass") if r.get("set_type") == "gold" else r.get("caught")
        lines.append(
            f"| {r['case_id']} | {r['set_type']} | {r.get('auto_score','')} | {r.get('auto_passed','')} | "
            f"{gc} | {(r.get('auto_issues') or r.get('error') or '')[:50]} | {r.get('elapsed_sec','')} |"
        )

    lines += [
        "",
        "## 结论与建议",
        "",
    ]
    if chal_rate < 0.9:
        lines.append("- 挑战集拦截不足：优先加强**审查 Prompt**（中间篇/OOC/跑题），开 Prompt 变更单。")
    if gold_rate < 0.7 or avg < 60:
        lines.append("- 黄金集偏弱：检查深档注入与**创作 Prompt**，或适当放宽非致命扣分后重测。")
    if mid_n == 0 and any(r["case_id"].startswith("CHAL-MDZS-101") for r in chal):
        # if challenge mid chapter not caught specifically
        pass
    lines.append("- 本报告为**自动审查分基线**；建议再人工抽 5 条黄金做 D1–D5 标定。")
    lines.append(f"- 明细 CSV：`{OUT_CSV.name}`")
    return "\n".join(lines)


def main():
    print("loading cases…")
    cases = load_cases()
    print(f"selected {len(cases)} cases")
    token = login()
    results = []
    for i, case in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] {case['case_id']} …")
        try:
            results.append(run_one(token, case))
        except Exception as e:
            print(f"  EXC {case['case_id']}: {e}")
            results.append(
                {
                    **case,
                    "run_date": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
                    "prompt_version": "baseline_v0.7",
                    "http_status": 0,
                    "elapsed_sec": "",
                    "auto_score": "",
                    "auto_passed": "",
                    "auto_issues": "",
                    "gate_pass": "false",
                    "caught": "false",
                    "error": str(e)[:400],
                    "content_preview": "",
                }
            )

    fieldnames = list(results[0].keys()) if results else []
    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(results)

    md = summarize(results)
    OUT_MD.write_text(md, encoding="utf-8")
    print("\n" + md)
    print(f"\nwrote {OUT_CSV} and {OUT_MD}")


if __name__ == "__main__":
    main()
