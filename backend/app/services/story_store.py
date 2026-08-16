import json
from pathlib import Path

from app.config import settings
from app.models.story import StorySeries, StorySeriesSummary, _now


class StoryStore:
    def __init__(self):
        self.root = Path(settings.data_dir) / "stories"
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, series_id: str) -> Path:
        return self.root / f"{series_id}.json"

    def save(self, series: StorySeries) -> StorySeries:
        series.updated_at = _now()
        path = self._path(series.id)
        path.write_text(
            series.model_dump_json(indent=2),
            encoding="utf-8",
        )
        return series

    def get(self, series_id: str) -> StorySeries | None:
        path = self._path(series_id)
        if not path.exists():
            return None
        return StorySeries.model_validate_json(path.read_text(encoding="utf-8"))

    def delete(self, series_id: str) -> bool:
        path = self._path(series_id)
        if not path.exists():
            return False
        path.unlink()
        return True

    def list_by_novel(self, novel_id: str) -> list[StorySeriesSummary]:
        items: list[StorySeriesSummary] = []
        for path in sorted(self.root.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if data.get("novel_id") != novel_id:
                continue
            chapters = data.get("chapters") or []
            last = chapters[-1] if chapters else {}
            items.append(
                StorySeriesSummary(
                    id=data.get("id", path.stem),
                    novel_id=data.get("novel_id", ""),
                    novel_title=data.get("novel_title", ""),
                    title=data.get("title", ""),
                    characters=data.get("characters", []),
                    character_names=data.get("character_names", []),
                    chapter_count=len(chapters),
                    plot_direction=data.get("plot_direction", ""),
                    updated_at=data.get("updated_at", ""),
                    last_summary=last.get("summary", ""),
                )
            )
        return items


story_store = StoryStore()
