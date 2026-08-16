#!/usr/bin/env python
"""RAG 一键配置与入库脚本（支持多作品隔离）。

用法:
  python scripts/setup_rag.py --novel modaozuoshi
  python scripts/setup_rag.py --novel shiyongzhuyizhe
  python scripts/setup_rag.py --novel shiyongzhuyizhe --skip-ingest
  python scripts/setup_rag.py --novel modaozuoshi --use-demo
"""
import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.novel_registry import novel_registry
from app.services.rag_service import rag_service


def sync_novel_source(novel_id: str) -> Path | None:
    novel = novel_registry.require(novel_id)
    source = novel.novel_source.strip()
    if not source:
        return None

    src = Path(source)
    if not src.is_absolute():
        src = Path(__file__).resolve().parent.parent / src

    if not src.exists():
        print(f"[WARN] 原著文件不存在: {src}")
        return None

    size = src.stat().st_size
    if size == 0:
        print(f"[FAIL] 原著文件为空（0 字节）: {src}")
        return None

    dst = novel.text_dir / novel.source_file
    novel.text_dir.mkdir(parents=True, exist_ok=True)

    encoding = novel.source_encoding or "utf-8"
    if encoding.lower() != "utf-8":
        text = src.read_text(encoding=encoding, errors="replace")
        dst.write_text(text, encoding="utf-8")
        print(
            f"\n[OK] 已同步并转码: {src.name} ({size / 1024 / 1024:.2f} MB, {encoding}->utf-8) -> {dst}"
        )
    else:
        shutil.copy2(src, dst)
        print(f"\n[OK] 已同步原著: {src.name} ({size / 1024 / 1024:.2f} MB) -> {dst}")
    return dst


def main():
    parser = argparse.ArgumentParser(description="RAG 配置检查与入库（多作品）")
    parser.add_argument(
        "--novel",
        required=True,
        help="作品 ID，如 modaozuoshi / shiyongzhuyizhe",
    )
    parser.add_argument("--skip-ingest", action="store_true", help="仅检查，不执行入库")
    parser.add_argument("--use-demo", action="store_true", help="复制 demo 摘要到 text/ 用于测试")
    args = parser.parse_args()

    novel = novel_registry.get(args.novel)
    if not novel:
        print(f"[FAIL] 未知作品: {args.novel}")
        print("可用作品:", ", ".join(n.id for n in novel_registry.list_all()))
        sys.exit(1)

    print("=" * 50)
    print(f"  RAG 配置检查 · 《{novel.title}》")
    print("=" * 50)

    env_path = Path(".env")
    if not env_path.exists():
        print("[WARN] 未找到 .env，请运行: copy .env.example .env")
    else:
        print("[OK] .env 已存在")

    print()
    print("Embedding 配置:")
    info = rag_service.get_embedding_info()
    print(f"  服务商: {info.get('provider', 'unknown')}")
    print(f"  模式: {info['mode']}")
    print(f"  模型: {info['model']}")
    if info.get("note"):
        print(f"  说明: {info['note']}")

    text_dir = novel.text_dir
    text_dir.mkdir(parents=True, exist_ok=True)

    if args.use_demo:
        demo_src = text_dir / "samples" / "demo_passages.txt"
        demo_dst = text_dir / "demo_passages.txt"
        if demo_src.exists():
            shutil.copy(demo_src, demo_dst)
            print(f"\n[OK] 已复制 demo 摘要到 {demo_dst}")
    elif novel.novel_source:
        synced = sync_novel_source(args.novel)
        if synced is None and not args.skip_ingest:
            sys.exit(1)

    txt_files = list(text_dir.glob("*.txt"))
    print()
    print(f"文本目录: {text_dir}")
    print(f"Chroma 集合: {rag_service._collection_name(args.novel)}")
    if txt_files:
        print(f"  找到 {len(txt_files)} 个 txt 文件:")
        for f in txt_files:
            size_kb = f.stat().st_size / 1024
            print(f"    - {f.name} ({size_kb:.1f} KB)")
    else:
        print("  [WARN] 未找到 txt 文件")
        if not args.skip_ingest:
            sys.exit(1)

    if args.skip_ingest:
        print("\n[SKIP] 跳过入库")
        return

    print()
    print("=" * 50)
    print(f"  开始重建索引 · 《{novel.title}》 (--force)")
    print("=" * 50)

    result = rag_service.ingest(novel_id=args.novel, force=True)
    if result.success:
        print(f"[OK] {result.message}")
        status = rag_service.get_status(args.novel)
        print(f"     Embedding: {status.embedding_mode} / {status.embedding_model}")
        print(f"     索引块数: {status.total_chunks}")
        print(f"     集合名: {status.collection_name}")
    else:
        print(f"[FAIL] {result.message}")
        sys.exit(1)


if __name__ == "__main__":
    main()
