"""
Artifex Assistant V5 — Embedding Pipeline.
Sentence embeddings for semantic search and RAG using sentence-transformers.
"""

import gc
import os

import numpy as np

from core.pipelines.base import BasePipeline, PipelineResult


class EmbeddingPipeline(BasePipeline):
    """Generate sentence embeddings for semantic search and RAG."""

    pipeline_type = "feature-extraction"
    display_name = "Embedding / RAG"
    output_type = "embedding"

    DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self):
        self.model = None
        self._model_path = None
        self._use_sentence_transformers = False

    def load(self, model_path: str, status_callback=None, **kwargs):
        """Load an embedding model.

        Tries sentence-transformers first (best quality), falls back to
        transformers AutoModel + mean pooling.

        Args:
            model_path: Local path or HuggingFace repo ID
                        (default: sentence-transformers/all-MiniLM-L6-v2)
            status_callback: Progress callback
        """
        if not model_path:
            model_path = self.DEFAULT_MODEL

        if status_callback:
            status_callback(f"Loading embedding model: {os.path.basename(model_path)}...")

        # Try sentence-transformers first (better API, optimized)
        try:
            from sentence_transformers import SentenceTransformer
            device = kwargs.get("device", "cpu")  # Embedding models run great on CPU
            self.model = SentenceTransformer(model_path, device=device)
            self._use_sentence_transformers = True
            self._model_path = model_path
            if status_callback:
                status_callback(f"Embedding model loaded (sentence-transformers): {os.path.basename(model_path)}")
            return
        except ImportError:
            pass

        # Fallback: raw transformers + mean pooling
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self._base_model = AutoModel.from_pretrained(
            model_path, trust_remote_code=True,
        )
        self._base_model.eval()
        self._use_sentence_transformers = False
        self.model = True  # sentinel
        self._model_path = model_path

        if status_callback:
            status_callback(f"Embedding model loaded (transformers fallback): {os.path.basename(model_path)}")

    def unload(self, status_callback=None):
        if self._use_sentence_transformers:
            del self.model
        else:
            for attr in ("_base_model", "_tokenizer"):
                if hasattr(self, attr):
                    delattr(self, attr)
        self.model = None
        self._model_path = None
        gc.collect()
        if status_callback:
            status_callback("Embedding model unloaded.")

    def is_loaded(self) -> bool:
        return self.model is not None

    def run(self, **kwargs) -> PipelineResult:
        """Generate embeddings or perform similarity search.

        Args:
            texts: List of strings to embed
            query: Query string for similarity search
            corpus_embeddings: Pre-computed corpus embeddings (numpy array)
            top_k: Number of results for similarity search (default 5)

        Returns:
            PipelineResult with:
              - If texts: content = numpy array of embeddings
              - If query + corpus_embeddings: content = list of (score, index) tuples
        """
        from core.pipelines.schemas import EmbeddingInput
        from pydantic import ValidationError
        try:
            params = EmbeddingInput(**kwargs)
        except ValidationError as e:
            return PipelineResult(success=False, output_type="embedding", error=f"Invalid input: {e}")

        if not self.is_loaded():
            return PipelineResult(
                success=False, output_type="embedding",
                error="No model loaded."
            )

        try:
            if params.texts:
                embeddings = self._encode(params.texts)
                return PipelineResult(
                    success=True,
                    output_type="embedding",
                    content=embeddings,
                    metadata={"count": len(params.texts), "dim": embeddings.shape[1]},
                )

            if params.query and params.corpus_embeddings is not None:
                query_emb = self._encode([params.query])
                scores = self._cosine_similarity(query_emb, params.corpus_embeddings)[0]
                top_indices = np.argsort(scores)[::-1][:params.top_k]
                results = [(float(scores[i]), int(i)) for i in top_indices]
                return PipelineResult(
                    success=True,
                    output_type="embedding",
                    content=results,
                    metadata={"query": params.query, "top_k": params.top_k},
                )

            return PipelineResult(
                success=False, output_type="embedding",
                error="Provide either 'texts' to embed or 'query' + 'corpus_embeddings' to search."
            )

        except Exception as e:
            return PipelineResult(
                success=False, output_type="embedding",
                error=str(e),
            )

    def _encode(self, texts):
        """Encode texts to embeddings."""
        if self._use_sentence_transformers:
            return self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

        # Transformers fallback with mean pooling
        import torch
        encoded = self._tokenizer(
            texts, padding=True, truncation=True, max_length=512,
            return_tensors="pt",
        )
        with torch.no_grad():
            output = self._base_model(**encoded)
        # Mean pooling over token dimension, respecting attention mask
        mask = encoded["attention_mask"].unsqueeze(-1).float()
        embeddings = (output.last_hidden_state * mask).sum(1) / mask.sum(1)
        # L2 normalize
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings.numpy()

    def _cosine_similarity(self, a, b):
        """Compute cosine similarity between two embedding matrices."""
        # Both should be L2-normalized already
        return np.dot(a, b.T)

    def get_vram_estimate(self, model_path: str) -> float:
        """Embedding models are tiny: 0.2-0.5 GB."""
        return 0.5

    def get_capabilities(self) -> dict:
        caps = super().get_capabilities()
        caps["default_model"] = self.DEFAULT_MODEL
        caps["modes"] = ["embed", "search"]
        caps["runs_on_cpu"] = True
        return caps
