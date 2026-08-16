from pydantic import BaseModel, Field


class CanonChunk(BaseModel):
    content: str
    source_file: str = ""
    chapter: str = ""
    chapter_title: str = ""
    characters: list[str] = Field(default_factory=list)
    score: float = Field(default=0.0, description="相关度分数，越高越相关")
    chunk_index: int = 0


class RagSearchRequest(BaseModel):
    novel_id: str = Field(..., description="作品 ID")
    query: str = Field(..., description="检索 query")
    characters: list[str] = Field(default_factory=list, description="相关人物 ID 或姓名")
    time_period: str = Field(default="", description="时间节点，用于增强检索")
    top_k: int = Field(default=5, ge=1, le=20)


class RagSearchResponse(BaseModel):
    query: str
    results: list[CanonChunk]
    total_in_index: int


class RagIngestResponse(BaseModel):
    success: bool
    novel_id: str = ""
    files_processed: int
    chunks_indexed: int
    message: str


class RagStatusResponse(BaseModel):
    enabled: bool
    novel_id: str = ""
    novel_title: str = ""
    collection_name: str
    total_chunks: int
    text_dir: str
    text_files: list[str]
    persist_dir: str
    embedding_mode: str = Field(default="local", description="api 或 local_bge")
    embedding_provider: str = Field(default="local_bge")
    embedding_model: str = ""
    embedding_ready: bool = False
    hint: str = ""
