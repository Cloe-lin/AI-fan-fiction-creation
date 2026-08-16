"""
预置账号：1 管理员 + user01..user20（每人不同初始密码）。
不开放注册；凭证明文仅写入本地文件（勿提交 Git）。

用法（在 backend 目录）:
  python scripts/seed_users.py
  python scripts/seed_users.py --force   # 重置全部预置账号密码
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.services.auth_store import count_users, init_db, seed_preset_accounts  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Seed admin + user01..user20")
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重置预置账号密码（会覆盖已有密码）",
    )
    args = parser.parse_args()

    init_db()
    creds = seed_preset_accounts(force_reset_passwords=args.force)
    out = ROOT / "app" / "data" / "credentials_initial.txt"
    out.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# AI 同人文平台 · 预置账号（请妥善保管，勿提交到 Git）",
        "# 生成后请分发给对应人员；此文件含明文密码。",
        "",
    ]
    if not creds:
        lines.append("# 本次未新建/重置账号（库中已存在预置用户）。")
        lines.append(f"# 当前用户总数: {count_users()}")
        lines.append("# 若需重置密码，请运行: python scripts/seed_users.py --force")
    else:
        lines.append(f"# 本次写入 {len(creds)} 个账号")
        lines.append("")
        lines.append(f"{'username':<12} {'password':<14} {'role':<8} display_name")
        lines.append("-" * 52)
        for c in creds:
            lines.append(
                f"{c['username']:<12} {c['password']:<14} {c['role']:<8} {c['display_name']}"
            )

    # 若非 force 且部分已存在，把已有说明写进文件头部；完整清单在 force 时最完整
    text = "\n".join(lines) + "\n"
    if args.force or not out.exists() or creds:
        # force：覆盖写完整清单；首次：写入；增量：追加本次新增
        if args.force or not out.exists():
            out.write_text(text, encoding="utf-8")
        else:
            with open(out, "a", encoding="utf-8") as f:
                f.write("\n# ---- 追加生成 ----\n")
                f.write(text)

    print(f"数据库用户数: {count_users()}")
    print(f"本次处理账号: {len(creds)}")
    print(f"凭据文件: {out}")
    if creds:
        print("请打开 credentials_initial.txt 查看并分发密码。")


if __name__ == "__main__":
    main()
