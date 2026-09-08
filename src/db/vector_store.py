"""
vector_store.py — In-memory Qdrant vector store for AST code chunks.

Indexes code snippets extracted by ``ast_chunker`` and supports semantic
search over the codebase via cosine-similarity nearest-neighbour lookup.

Embeddings are produced by `FastEmbed <https://github.com/qdrant/fastembed>`_
(ONNX-backed, no PyTorch required) using a compact model suitable for
code / natural-language hybrid queries.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional, Sequence

from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_COLLECTION = "code_chunks"
# Fast, 384-dim model — good balance of speed and quality for code search.
_DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
_BATCH_SIZE = 64


# ---------------------------------------------------------------------------
# CodeVectorStore
# ---------------------------------------------------------------------------


class CodeVectorStore:
    """In-memory Qdrant vector store for code chunk retrieval.

    Parameters
    ----------
    collection_name:
        Name of the Qdrant collection.
    embedding_model:
        Any model name accepted by ``fastembed.TextEmbedding``.
    """

    def __init__(
        self,
        collection_name: str = _DEFAULT_COLLECTION,
        embedding_model: str = _DEFAULT_MODEL,
    ) -> None:
        self.collection_name = collection_name

        # ── Embedding model (downloaded / cached on first use) ──────────
        self._embedder = TextEmbedding(model_name=embedding_model)
        self._vector_size = self._probe_vector_size()

        # ── In-memory Qdrant instance ──────────────────────────────────
        self._client = QdrantClient(":memory:")
        self._create_collection()
        self._point_counter: int = 0

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _probe_vector_size(self) -> int:
        """Embed a throwaway string to discover the model's output dim."""
        sample = list(self._embedder.embed(["probe"]))[0]
        return len(sample)

    def _create_collection(self) -> None:
        self._client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=self._vector_size,
                distance=models.Distance.COSINE,
            ),
        )

    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of strings and return a list of float vectors."""
        return [vec.tolist() for vec in self._embedder.embed(texts)]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def index(self, chunks: Sequence[Dict[str, Any]]) -> int:
        """Index a list of AST code-chunk dicts produced by ``ast_chunker``.

        Each dict is expected to contain at least:

        * ``code_snippet`` — source text (used for embedding)
        * ``file_path``
        * ``function_name``
        * ``start_line``
        * ``end_line``

        Returns the number of points upserted.
        """
        if not chunks:
            return 0

        texts = [chunk["code_snippet"] for chunk in chunks]
        vectors = self._embed_texts(texts)

        points: List[models.PointStruct] = []
        for vec, chunk in zip(vectors, chunks):
            point_id = self._point_counter
            self._point_counter += 1

            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=vec,
                    payload={
                        "code_snippet": chunk["code_snippet"],
                        "file_path": chunk["file_path"],
                        "function_name": chunk["function_name"],
                        "start_line": chunk["start_line"],
                        "end_line": chunk["end_line"],
                    },
                )
            )

        # Upsert in batches to keep memory usage predictable
        for i in range(0, len(points), _BATCH_SIZE):
            batch = points[i : i + _BATCH_SIZE]
            self._client.upsert(
                collection_name=self.collection_name,
                points=batch,
            )

        return len(points)

    def search(
        self,
        query: str,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """Semantic search over indexed code chunks.

        Parameters
        ----------
        query:
            Free-text query (natural language *or* code fragment).
        top_k:
            Number of results to return.

        Returns
        -------
        list[dict]
            Each dict contains the original metadata plus a ``score``
            key indicating cosine similarity.
        """
        query_vector = self._embed_texts([query])[0]

        hits = self._client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
        ).points

        results: List[Dict[str, Any]] = []
        for hit in hits:
            entry = dict(hit.payload)  # code_snippet, file_path, etc.
            entry["score"] = hit.score
            results.append(entry)

        return results

    def count(self) -> int:
        """Return the total number of indexed points."""
        info = self._client.get_collection(self.collection_name)
        return info.points_count


# ---------------------------------------------------------------------------
# CLI convenience — index a repo and run interactive queries
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python vector_store.py <repo_path> [query]")
        sys.exit(1)

    repo_path = sys.argv[1]

    # Import the chunker (handle both package and standalone execution)
    try:
        from src.parser.ast_chunker import parse_repository
    except ImportError:
        import importlib.util

        _spec = importlib.util.spec_from_file_location(
            "ast_chunker",
            str(__import__("pathlib").Path(__file__).resolve().parent.parent / "parser" / "ast_chunker.py"),
        )
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        parse_repository = _mod.parse_repository

    print(f"Parsing repository: {repo_path}")
    chunks = parse_repository(repo_path)
    print(f"  ↳ Found {len(chunks)} code chunks")

    print("Building vector index …")
    store = CodeVectorStore()
    n = store.index(chunks)
    print(f"  ↳ Indexed {n} vectors  (dim={store._vector_size})")

    # If a query was provided on the CLI, run it
    query = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else None
    if query:
        print(f"\nSearching for: {query!r}\n")
        results = store.search(query, top_k=3)
        for i, r in enumerate(results, 1):
            print(f"── Result {i}  (score={r['score']:.4f}) ──")
            print(f"   {r['file_path']}:{r['start_line']}-{r['end_line']}  {r['function_name']}")
            snippet = r["code_snippet"]
            if len(snippet) > 200:
                snippet = snippet[:200] + " …"
            print(f"   {snippet}\n")
    else:
        print("\nTip: pass a query string as the second argument to search.")
