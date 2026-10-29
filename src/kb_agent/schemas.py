"""数据格式：文档、对话、答案、工具参数和调用上限"""

from datetime import date
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# 拒绝多余字段，去掉字符串两端空白。
CONFIG = ConfigDict(extra="forbid", str_strip_whitespace=True)


# 知识库文档：每份文档包含多个片段。
class Chunk(BaseModel):
    model_config = CONFIG

    chunk_id: str = Field(min_length=1)
    heading: str
    text: str = Field(min_length=1)


class Document(BaseModel):
    model_config = CONFIG

    doc_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    version_date: date
    scope: str | None = None
    supersedes: str | None = None
    chunks: list[Chunk] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_chunks(self):
        if len({c.chunk_id for c in self.chunks}) != len(self.chunks):
            raise ValueError("Chunk IDs must be unique within a document.")
        return self


# 用户提供的对话历史。
class HistoryMessage(BaseModel):
    model_config = CONFIG

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)


# 答案及其引用，由 agent.py 检查引文是否匹配原文
class Citation(BaseModel):
    model_config = CONFIG

    doc_id: str = Field(min_length=1)
    chunk_id: str = Field(min_length=1)
    version_date: date
    quote: str = Field(min_length=1)


Status = Literal[
    "answered", "needs_clarification", "insufficient_evidence", "tool_error", "incomplete"
]


class Answer(BaseModel):
    model_config = CONFIG

    # 所有字段都要明确提供，没有内容时使用 [] 或 None
    status: Status
    answer: str = Field(min_length=1, description="Complete answer in the user's language, including any conflicting rules.")
    citations: list[Citation] = Field(description="Exact quotations from sources read in this run, or [].")
    conflicts: list[str] = Field(description="Unresolved conflict descriptions, or [].")
    gaps: list[str] = Field(description="Missing evidence; must be nonempty unless status is answered.")
    follow_up: str | None = Field(description="A question for needs_clarification, otherwise null.")

    @field_validator("conflicts", "gaps", mode="before")
    @classmethod
    def normalize_single_description(cls, value):
        # 兼容模型把单条描述写成字符串，例如 "范围未知" → ["范围未知"]
        if isinstance(value, str):
            return [value.strip()] if value.strip() else []
        return value

    @model_validator(mode="after")
    def consistent_status(self):
        """
        字段之间的关系：有依据才能回答，未完成必须说明缺口
        """
        
        if self.status == "answered" and (not self.citations or self.conflicts or self.gaps):
            raise ValueError("answered requires citations and no unresolved conflicts or gaps.")
        
        if self.status != "answered" and not self.gaps:
            raise ValueError("An unfinished answer must explain its evidence gaps.")
        
        if self.status == "needs_clarification" and not self.follow_up:
            raise ValueError("needs_clarification requires a concrete follow-up question.")
        
        if self.conflicts and len({c.doc_id for c in self.citations}) < 2:
            raise ValueError("Conflicts require citations from at least two documents.")
        
        return self


class SearchDocuments(BaseModel):
    """
    Search documents. Read the source before citing a search result
    """

    model_config = CONFIG

    query: str = Field(min_length=1, max_length=300)
    top_k: int = Field(default=5, ge=1, le=10)
    rationale: str = Field(min_length=1, max_length=300, description="Brief public action reason, not private reasoning.")


class ReadDocument(BaseModel):
    """
    Read source text and applicability. Omit chunk_id to read the whole document
    """

    model_config = CONFIG

    doc_id: str = Field(min_length=1)
    chunk_id: str | None = None
    rationale: str = Field(min_length=1, max_length=300)


class Finish(Answer):
    """
    Submit an answer or clarification question. Invalid citations will be rejected.
    """

    rationale: str = Field(min_length=1, max_length=300)


# 只能执行三种动作
ACTION_SCHEMAS = {
    "search_documents": SearchDocuments,
    "read_document": ReadDocument,
    "finish": Finish,
}


# 运行预算，防止循环无限执行
class Limits(BaseModel):
    model_config = CONFIG

    max_model_calls: int = Field(default=10, ge=1, le=100)
    max_tool_calls: int = Field(default=8, ge=1, le=100)
