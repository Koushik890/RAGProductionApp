import pydantic


class RAGChunkAndSrc(pydantic.BaseModel):
    chunks: list[str]
    source_id: str = None

class RAGUpsertResult(pydantic.BaseModel):
    inngested: int

class RAGSearchResult(pydantic.BaseModel):
    contexts: list[str]
    sources: list[str]

class RAQQueryResult(pydantic.BaseModel):
    answer: str
    sources: list[str]
    num_contexts: int