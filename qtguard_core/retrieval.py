from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Dict, Any

import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder


@dataclass
class Evidence:
    title: str
    section: str
    chunk_id: str
    text: str
    score: float


class HybridRetriever:
    """
    Hybrid retrieval:
      1) BM25 keyword retrieval
      2) Dense vector retrieval (SentenceTransformer + FAISS)
      3) Cross-encoder reranking (query, chunk) -> relevance logit

    Notes:
      - Cross-encoder scores are raw logits and can be negative.
      - We deduplicate by chunk_id after reranking.
      - This is designed to work locally and on Kaggle without GPU-specific FAISS.
    """

    def __init__(
        self,
        chunks_path: str = "assets/knowledge/chunks.jsonl",
        embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        candidate_k: int = 30,
        top_k: int = 6,
    ):
        self.candidate_k = candidate_k
        self.top_k = top_k

        # Load corpus
        self.rows: List[Dict[str, Any]] = []
        texts: List[str] = []
        with open(chunks_path, "r", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                # Defensive checks
                if "text" not in r:
                    continue
                self.rows.append(r)
                texts.append(r["text"])

        if not self.rows:
            raise RuntimeError(f"No chunks loaded from {chunks_path}. Is the file empty?")

        # BM25 setup
        self.bm25_tokens = [t.lower().split() for t in texts]
        self.bm25 = BM25Okapi(self.bm25_tokens)

        # Dense embeddings + FAISS
        self.embedder = SentenceTransformer(embed_model)
        embs = self.embedder.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        self.embs = np.asarray(embs, dtype=np.float32)

        # Inner-product index (since vectors are normalized, IP == cosine similarity)
        self.index = faiss.IndexFlatIP(self.embs.shape[1])
        self.index.add(self.embs)

        # Cross-encoder reranker
        self.reranker = CrossEncoder(rerank_model)

    def _bm25_candidates(self, query: str) -> List[int]:
        scores = self.bm25.get_scores(query.lower().split())
        return np.argsort(scores)[::-1][: self.candidate_k].tolist()

    def _vector_candidate
