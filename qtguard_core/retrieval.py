# qtguard_core/retrieval.py
from __future__ import annotations
import json
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
import numpy as np


@dataclass
class Evidence:
    doc_id: str
    title: str
    section: str
    text: str
    score: float


class HybridRetriever:
    def __init__(
        self,
        docs_path: str = "assets/knowledge/chunks.jsonl",
        embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_k: int = 8,
        candidate_k: int = 40,
    ):
        self.top_k = top_k
        self.candidate_k = candidate_k

        # Load chunk corpus
        self.chunks: List[Dict[str, Any]] = []
        texts: List[str] = []
        with open(docs_path, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                self.chunks.append(row)
                texts.append(row["text"])

        # BM25
        self._bm25_tokens = [t.lower().split() for t in texts]
        self.bm25 = BM25Okapi(self._bm25_tokens)

        # Embeddings + FAISS
        self.embedder = SentenceTransformer(embed_model)
        embs = self.embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        self.embs = np.asarray(embs, dtype=np.float32)

        self.index = faiss.IndexFlatIP(self.embs.shape[1])
        self.index.add(self.embs)

        # Re-ranker
        self.reranker = CrossEncoder(rerank_model)

    def search(self, query: str) -> List[Evidence]:
        # 1) BM25 candidates
        bm25_scores = self.bm25.get_scores(query.lower().split())
        bm25_top = np.argsort(bm25_scores)[::-1][: self.candidate_k].tolist()

        # 2) Vector candidates
        q = self.embedder.encode([query], normalize_embeddings=True)
        q = np.asarray(q, dtype=np.float32)
        _, vec_top = self.index.search(q, self.candidate_k)
        vec_top = vec_top[0].tolist()

        # 3) Union + dedupe
        cand_ids = list(dict.fromkeys(bm25_top + vec_top))[: (2 * self.candidate_k)]

        candidates = [self.chunks[i] for i in cand_ids]
        pairs = [[query, c["text"]] for c in candidates]

        # 4) Re-rank
        rr_scores = self.reranker.predict(pairs)
        ranked = sorted(zip(candidates, rr_scores), key=lambda x: x[1], reverse=True)[: self.top_k]

        out: List[Evidence] = []
        for c, s in ranked:
            out.append(
                Evidence(
                    doc_id=c.get("doc_id", ""),
                    title=c.get("title", ""),
                    section=c.get("section", ""),
                    text=c["text"],
                    score=float(s),
                )
            )
        return out
