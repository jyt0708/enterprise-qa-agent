"""TF-IDF知识库检索"""

import json
from pathlib import Path
from pydantic import TypeAdapter
from sklearn.feature_extraction.text import TfidfVectorizer
from kb_agent.schemas import Document


# 加载JSON文档、数据校验、向量构建、search和read
class KnowledgeBase:
    def __init__(self, path: str | Path):
        raw = Path(path).read_bytes()
        
        # 用TypeAdapter可以直接为 list[Document] 校验
        documents = TypeAdapter(list[Document]).validate_json(raw)
        self.documents = {doc.doc_id: doc for doc in documents}
        
        if len(self.documents) != len(documents):
            raise ValueError("Document IDs must be unique.")
        
        self.rows = [(doc, chunk) for doc in documents for chunk in doc.chunks]
        
        # 中文环境下更适合
        self.vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4))
        self.matrix = None
        
        # TF-IDF 矩阵
        if self.rows:
            self.matrix = self.vectorizer.fit_transform(
                [f"{doc.title} {chunk.heading} {chunk.text}" for doc, chunk in self.rows]
            )

    def search(self, query: str, top_k: int = 5) -> dict:
        if self.matrix is None:
            return {"ok": True, "query": query, "hits": [], "total_matches": 0, "truncated": False}
        
        scores = (self.matrix @ self.vectorizer.transform([query]).T).toarray().ravel()
        candidates = {}
        
        for (doc, chunk), score in zip(self.rows, scores):
            if score < 0.08:
                continue
            if doc.doc_id not in candidates:
                candidates[doc.doc_id] = {
                    "doc_id": doc.doc_id,
                    "title": doc.title,
                    "version_date": doc.version_date.isoformat(),
                    "score": 0.0,
                    "previews": [],
                }
            candidate = candidates[doc.doc_id]
            candidate["score"] = max(candidate["score"], round(float(score), 4))
            
            # 预览片段
            candidate["previews"].append({
                "chunk_id": chunk.chunk_id,
                "heading": chunk.heading,
                "excerpt": chunk.text[:90],
            })
            
        ranked = sorted(candidates.values(), key=lambda item: (-item["score"], item["doc_id"]))
        
        return {
            "ok": True,
            "query": query,
            "hits": ranked[:top_k],
            "total_matches": len(ranked),
            "truncated": len(ranked) > top_k,
            "note": "Lexical matches only. Read relevant sources before citing. Reformulate weak searches.",
        }

    def read(self, doc_id: str, chunk_id: str | None = None) -> dict:
        doc = self.documents.get(doc_id)
        if doc is None:
            return {"ok": False, "error": "document_not_found", "doc_id": doc_id, "retryable": False}
        
        result = doc.model_dump(mode="json")
        if chunk_id is not None:
            result["chunks"] = [c for c in result["chunks"] if c["chunk_id"] == chunk_id]
            if not result["chunks"]:
                return {"ok": False, "error": "chunk_not_found", "doc_id": doc_id, "chunk_id": chunk_id, "retryable": False}
        
        return {"ok": True, "document": result}


class LocalTools:
    """
    引入failure_mode用于可控的模拟工具故障
    none: 工具正常执行
    read_once: 第一次读取失败，后续读取正常；是否重试由模型决定
    """

    def __init__(self, knowledge: KnowledgeBase, failure_mode: str = "none"):
        if failure_mode not in {"none", "read_once"}:
            raise ValueError("Unknown failure mode.")
        
        self.knowledge = knowledge
        self.failure_mode = failure_mode
        self.read_attempts = 0
    
    
    # 调用工具
    def execute(self, name: str, arguments: dict) -> dict:
        if name == "search_documents":
            return self.knowledge.search(**arguments)
        
        if name == "read_document":
            self.read_attempts += 1
            if self.failure_mode == "read_once" and self.read_attempts == 1:
                return {
                    "ok": False, 
                    "error": "read_unavailable", 
                    "retryable": True, 
                    "doc_id": arguments["doc_id"]
                    }
            
            return self.knowledge.read(**arguments)
        
        raise ValueError("Unknown local tool.")


# 总结执行一个动作后得到的结果或反馈，给模型提供下一步决策支持
# 方便返回摘要 和 人阅读执行记录
def summarize(observation: dict) -> str:
    if not observation.get("ok"):
        return json.dumps(observation, ensure_ascii=False)
    
    # 成功检索文档
    if "hits" in observation:
        ids = [hit["doc_id"] for hit in observation["hits"]]
        return f"Found {observation['total_matches']} matching documents; returned {ids}; truncated={observation['truncated']}"
    
    # 成功读取文档
    if "document" in observation:
        doc = observation["document"]
        return f"Read {doc['doc_id']} ({doc['version_date']}), chunks: {[c['chunk_id'] for c in doc['chunks']]}; scope={doc['scope']}; supersedes={doc['supersedes']}"
    
    return json.dumps(observation, ensure_ascii=False)
