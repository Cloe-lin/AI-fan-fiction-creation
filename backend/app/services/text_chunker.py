import re
from dataclasses import dataclass, field


CHAPTER_PATTERN = re.compile(
    r"^(?:第[0-9一二三四五六七八九十百千零两]+章|[Cc]hapter\s+\d+)(?:\s*[·\.、：:\s]\s*(.*))?$"
)


@dataclass
class TextChunk:
    content: str
    source_file: str
    chapter: str = ""
    chapter_title: str = ""
    chunk_index: int = 0
    characters: list[str] = field(default_factory=list)


def split_into_chapters(text: str, source_file: str) -> list[tuple[str, str, str]]:
    """将文本按章节切分，返回 (chapter_label, chapter_title, chapter_body) 列表。"""
    lines = text.splitlines()
    chapters: list[tuple[str, str, str]] = []
    current_label = "序章"
    current_title = ""
    current_lines: list[str] = []

    def flush():
        nonlocal current_lines
        body = "\n".join(current_lines).strip()
        if body:
            chapters.append((current_label, current_title, body))
        current_lines = []

    for line in lines:
        stripped = line.strip()
        match = CHAPTER_PATTERN.match(stripped) if stripped else None
        if match:
            flush()
            current_label = stripped.split("·")[0].split("、")[0].strip()
            current_title = (match.group(1) or "").strip()
        else:
            current_lines.append(line)

    flush()

    if not chapters and text.strip():
        chapters.append(("全文", "", text.strip()))

    return chapters


def chunk_text(
    text: str,
    source_file: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    character_names: list[str] | None = None,
) -> list[TextChunk]:
    """将文本切分为带重叠的块，并标注出现的角色名。"""
    character_names = character_names or []
    all_chunks: list[TextChunk] = []

    for chapter_label, chapter_title, body in split_into_chapters(text, source_file):
        start = 0
        chunk_index = 0
        body_len = len(body)

        while start < body_len:
            end = min(start + chunk_size, body_len)
            if end < body_len:
                boundary = _find_boundary(body, end, window=80)
                if boundary > start:
                    end = boundary

            content = body[start:end].strip()
            if content:
                mentioned = _extract_characters(content, character_names)
                all_chunks.append(
                    TextChunk(
                        content=content,
                        source_file=source_file,
                        chapter=chapter_label,
                        chapter_title=chapter_title,
                        chunk_index=chunk_index,
                        characters=mentioned,
                    )
                )
                chunk_index += 1

            if end >= body_len:
                break
            start = max(end - chunk_overlap, start + 1)

    return all_chunks


def _find_boundary(text: str, position: int, window: int = 80) -> int:
    """在 position 附近寻找自然断句点。"""
    search_start = max(0, position - window)
    segment = text[search_start:position]
    for sep in ("。", "！", "？", "……", "\n", "；", "，"):
        idx = segment.rfind(sep)
        if idx != -1:
            return search_start + idx + len(sep)
    return position


def _extract_characters(text: str, character_names: list[str]) -> list[str]:
    found = []
    for name in sorted(character_names, key=len, reverse=True):
        if name in text and name not in found:
            found.append(name)
    return found
