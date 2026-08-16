#!/usr/bin/env python
"""原著文本入库脚本（多作品）。

用法:
    python scripts/ingest_novel.py --novel modaozuoshi
    python scripts/ingest_novel.py --novel shiyongzhuyizhe --force
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.novel_registry import novel_registry
from app.services.rag_service import rag_service


def main():
    parser = argparse.ArgumentParser(description="原著文本 RAG 入库（多作品）")
    parser.add_argument("--novel", required=True, help="作品 ID")
    parser.add_argument(
        "--force",
        action="store_true",
        help="清空现有索引后重新入库",
    )
    args = parser.parse_args()

    novel = novel_registry.get(args.novel)
    if not novel:
        print(f"[FAIL] 未知作品: {args.novel}")
        sys.exit(1)

    embedding = rag_service.get_embedding_info()
    print(f"作品: {novel.title} ({novel.id})")
    print(f"Embedding: {embedding['mode']} / {embedding['model']}")
    if embedding.get("note"):
        print(f"[INFO] {embedding['note']}")
    print(f"文本目录: {novel.text_dir}")
    print(f"索引集合: {rag_service._collection_name(args.novel)}")
    print("开始入库...")

    result = rag_service.ingest(novel_id=args.novel, force=args.force)

    if result.success:
        print(f"[OK] {result.message}")
    else:
        print(f"[FAIL] {result.message}")
        sys.exit(1)


if __name__ == "__main__":
    main()
