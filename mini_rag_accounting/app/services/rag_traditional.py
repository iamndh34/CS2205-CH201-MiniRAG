import json
import os
import time
from typing import Optional

import numpy as np
import ollama
from sentence_transformers import SentenceTransformer

from app.core.config import settings
from .stream_utils import stream_by_sentence
from .prompt_template import SYSTEM_PROMPT, build_user_prompt, build_traditional_context

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COA_FILE = os.path.join(BASE_DIR, "services", "rag_json", "coa_99.json")

if not os.path.exists(COA_FILE):
    COA_DATA = []
else:
    with open(COA_FILE, "r", encoding="utf-8") as f:
        COA_DATA = json.load(f)

COA_BY_CODE = {acc["code"]: acc for acc in COA_DATA}

EMBEDDING_MODEL_NAME = "bkai-foundation-models/vietnamese-bi-encoder"
EMBEDDING_DIM = 768
FLOAT32_BYTES = 4


def get_model_size_mb(model_name: str) -> float:
    """Get actual model size from HuggingFace cache"""
    try:
        from pathlib import Path
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        total_size = 0
        for model_dir in cache_dir.glob(f"models--{model_name.replace('/', '--')}*"):
            for f in model_dir.rglob("*"):
                if f.is_file():
                    total_size += f.stat().st_size
        if total_size > 0:
            return total_size / (1024 * 1024)
    except:
        pass
    return 0


def get_embeddings_size_bytes(embeddings: dict) -> int:
    """Get actual memory size of embeddings dict"""
    import sys
    total = sys.getsizeof(embeddings)
    for code, item in embeddings.items():
        total += sys.getsizeof(code)
        total += sys.getsizeof(item)
        total += item["embedding"].nbytes  # numpy array actual size
        total += sys.getsizeof(item["data"])
    return total


class RagTraditional:
    _embed_model = None
    _embeddings = None
    _initialized = False
    _model_size_mb = 0
    _embeddings_size_kb = 0
    _init_time_ms = 0

    @classmethod
    def _init(cls):
        """Initialize embedding model and compute all embeddings"""
        if cls._initialized:
            return

        init_start = time.perf_counter()

        print("[Traditional RAG] Initializing embedding model...")
        cls._embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

        cls._model_size_mb = get_model_size_mb(EMBEDDING_MODEL_NAME)
        if cls._model_size_mb == 0:
            cls._model_size_mb = 540  # PhoBERT-base fallback

        print("[Traditional RAG] Computing embeddings for all documents...")
        cls._embeddings = {}
        for acc in COA_DATA:
            # Create rich text for embedding
            text = f"Tài khoản {acc['code']} {acc['name']} {acc.get('name_en', '')} loại {acc['type_name']}"
            cls._embeddings[acc["code"]] = {
                "embedding": cls._embed_model.encode(text, normalize_embeddings=True),
                "data": acc
            }

        cls._embeddings_size_kb = get_embeddings_size_bytes(cls._embeddings) / 1024
        cls._init_time_ms = (time.perf_counter() - init_start) * 1000
        cls._initialized = True
        print(f"[Traditional RAG] Initialized with {len(cls._embeddings)} documents in {cls._init_time_ms:.0f}ms")
        print(f"[Traditional RAG] Model: {cls._model_size_mb:.1f}MB, Embeddings: {cls._embeddings_size_kb:.1f}KB")

    @classmethod
    def get_resource_info(cls) -> dict:
        """Return resource usage info for Traditional RAG"""
        num_docs = len(COA_DATA)

        # Get actual values if initialized, otherwise estimate
        if cls._initialized:
            index_size_kb = cls._embeddings_size_kb
            model_size = cls._model_size_mb
        else:
            # Estimate before init
            index_size_kb = (num_docs * EMBEDDING_DIM * FLOAT32_BYTES) / 1024
            model_size = get_model_size_mb(EMBEDDING_MODEL_NAME)

        return {
            "index_type": "Vector Embeddings",
            "index_size_kb": round(index_size_kb, 2),
            "embedding_model": EMBEDDING_MODEL_NAME,
            "embedding_model_size_mb": round(model_size, 1) if model_size > 0 else "Chưa load",
            "embedding_dim": EMBEDDING_DIM,
            "total_documents": num_docs,
            "initialized": cls._initialized
        }

    @classmethod
    def search(cls, query: str, top_k: int = 3) -> tuple:
        """
        Search documents by embedding similarity.
        Returns (results, search_details) for explainability.
        """
        cls._init()

        # Time the embedding
        embed_start = time.perf_counter()
        query_emb = cls._embed_model.encode(query, normalize_embeddings=True)
        embed_time = (time.perf_counter() - embed_start) * 1000

        # Time the search
        search_start = time.perf_counter()
        scores = []
        for code, item in cls._embeddings.items():
            sim = float(np.dot(query_emb, item["embedding"]))
            scores.append((code, sim, item["data"]))

        scores.sort(key=lambda x: -x[1])
        results = [(code, score, data) for code, score, data in scores[:top_k]]
        search_time = (time.perf_counter() - search_start) * 1000

        search_details = {
            "embed_time_ms": round(embed_time, 3),
            "knn_time_ms": round(search_time, 3),
            "total_compared": len(cls._embeddings),
            "top_k": top_k
        }

        return results, search_details

    @classmethod
    def ask(cls, question: str):
        print(f"\n[Traditional RAG] Question: {question}")

        if not cls._initialized:
            cls._init()

        retrieval_start = time.perf_counter()
        query_emb = cls._embed_model.encode(question, normalize_embeddings=True)

        scores = []
        for code, item in cls._embeddings.items():
            sim = float(np.dot(query_emb, item["embedding"]))
            scores.append((code, sim, item["data"]))
        scores.sort(key=lambda x: -x[1])

        results = scores[:1]
        retrieval_time = time.perf_counter() - retrieval_start

        if not results:
            yield "Không tìm thấy thông tin phù hợp."
            yield f"__RETRIEVAL__:{json.dumps({'time_ms': round(retrieval_time * 1000, 3), 'found': False})}"
            return

        top_result = results[0]

        print(f"[Traditional RAG] Found result in {retrieval_time*1000:.3f}ms")

        retrieval_info = {
            "time_ms": round(retrieval_time * 1000, 3),
            "found": True,
            "doc_count": 1,
            "confidence": f"{top_result[1]:.1%}"
        }
        yield f"__RETRIEVAL__:{json.dumps(retrieval_info, ensure_ascii=False)}"

        # ===== GENERATION PHASE =====
        context = build_traditional_context(results)
        user_prompt = build_user_prompt(question, context)

        generation_start = time.perf_counter()
        slm_output = ""
        try:
            client = ollama.Client(host=settings.OLLAMA_HOST)
            stream = client.chat(
                model=settings.GENERATION_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                stream=True
            )
            for sentence in stream_by_sentence(stream):
                slm_output += sentence
                yield sentence
        except Exception as e:
            print(f"[Traditional RAG Error] {e}")
            yield f"Lỗi: {e}"

        generation_time = time.perf_counter() - generation_start

        if slm_output:
            yield "\n\n(Căn cứ: Thông tư 99/2025 - Traditional RAG)"

        # Send generation time
        yield f"__GENERATION__:{json.dumps({'time_ms': round(generation_time * 1000, 3)})}"
