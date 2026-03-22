"""
Artifex Assistant V5 — RAG (Retrieval-Augmented Generation) engine.
Indexes documents/code into a vector store for semantic retrieval.
"""

import os
import json

import numpy as np

from core.logging_config import get_logger

_log = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Document chunking
# ─────────────────────────────────────────────────────────────────────────────

_TEXT_EXTENSIONS = {
    ".py", ".js", ".ts", ".go", ".rs", ".java", ".c", ".cpp", ".h",
    ".md", ".txt", ".rst", ".yaml", ".yml", ".json", ".toml", ".ini",
    ".cfg", ".xml", ".html", ".css", ".sql", ".sh", ".bat", ".ps1",
    ".rb", ".php",
}

_SKIP_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "venv", "env",
    ".mypy_cache", ".pytest_cache", "dist", "build", ".tool_cache",
    ".tox", "egg-info", ".idea", ".vs", ".vscode",
}


def _chunk_text(text, chunk_size=512, overlap=64):
    """Split text into overlapping chunks by character count.

    Args:
        text: Full text content
        chunk_size: Target chars per chunk
        overlap: Overlap between consecutive chunks

    Returns:
        List of text chunks
    """
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def _collect_files(directory, extensions=None):
    """Recursively collect text files from a directory.

    Returns:
        List of (filepath, content) tuples
    """
    exts = extensions or _TEXT_EXTENSIONS
    files = []

    for root, dirs, filenames in os.walk(directory):
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]
        for fname in filenames:
            ext = os.path.splitext(fname)[1].lower()
            if ext not in exts:
                continue
            path = os.path.join(root, fname)
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
                if content.strip():
                    files.append((path, content))
            except (OSError, PermissionError):
                continue

    return files


# ─────────────────────────────────────────────────────────────────────────────
# Vector Store
# ─────────────────────────────────────────────────────────────────────────────

class VectorStore:
    """In-memory vector store with numpy cosine similarity search.

    Stores document chunks with their embeddings and metadata for fast
    semantic retrieval. Persists to disk as .npz + .json.
    """

    def __init__(self):
        self.embeddings = None   # numpy array (N, D)
        self.documents = []      # list of text chunks
        self.metadata = []       # list of metadata dicts per chunk

    @property
    def count(self):
        return len(self.documents)

    def add_documents(self, texts, embeddings, metadata=None):
        """Add embedded documents to the store.

        Args:
            texts: List of text chunks
            embeddings: numpy array (N, D) of corresponding embeddings
            metadata: Optional list of dicts with source info per chunk
        """
        if metadata is None:
            metadata = [{}] * len(texts)

        if self.embeddings is None:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])

        self.documents.extend(texts)
        self.metadata.extend(metadata)

    def search(self, query_embedding, top_k=5):
        """Find the most similar documents to a query embedding.

        Args:
            query_embedding: numpy array (1, D) or (D,)
            top_k: Number of results to return

        Returns:
            List of (score, text, metadata) tuples, sorted by descending score
        """
        if self.embeddings is None or len(self.documents) == 0:
            return []

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # Cosine similarity (embeddings assumed normalized)
        scores = np.dot(query_embedding, self.embeddings.T)[0]
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for i in top_indices:
            if scores[i] > 0:
                results.append((
                    float(scores[i]),
                    self.documents[i],
                    self.metadata[i],
                ))

        return results

    def save(self, path):
        """Persist vector store to disk.

        Saves embeddings as .npz and metadata as .json.
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        if self.embeddings is not None:
            np.savez_compressed(path + ".npz", embeddings=self.embeddings)

        meta = {
            "documents": self.documents,
            "metadata": self.metadata,
        }
        with open(path + ".json", "w", encoding="utf-8") as f:
            json.dump(meta, f)

        _log.info("VectorStore saved: %d documents to %s", len(self.documents), path)

    def load(self, path):
        """Load vector store from disk."""
        npz_path = path + ".npz"
        json_path = path + ".json"

        if not os.path.isfile(json_path):
            _log.warning("VectorStore not found: %s", path)
            return False

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            self.documents = meta.get("documents", [])
            self.metadata = meta.get("metadata", [])

            if os.path.isfile(npz_path):
                data = np.load(npz_path)
                self.embeddings = data["embeddings"]

            _log.info("VectorStore loaded: %d documents from %s", len(self.documents), path)
            return True
        except Exception as e:
            _log.error("Failed to load VectorStore: %s", e)
            return False

    def clear(self):
        """Clear all stored data."""
        self.embeddings = None
        self.documents = []
        self.metadata = []


# ─────────────────────────────────────────────────────────────────────────────
# RAG Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class RAGPipeline:
    """Orchestrates embedding + retrieval + context injection for RAG.

    Usage:
        rag = RAGPipeline(embedding_pipeline)
        rag.index_directory("/path/to/project")
        context = rag.augment_prompt("How does auth work?", top_k=5)
    """

    def __init__(self, embedding_pipeline=None):
        self.embedding = embedding_pipeline
        self.store = VectorStore()

    def index_directory(self, directory, extensions=None, chunk_size=512,
                       status_callback=None):
        """Index all text files in a directory.

        Args:
            directory: Path to directory
            extensions: Set of file extensions to include
            chunk_size: Characters per chunk
            status_callback: Progress callback
        """
        if self.embedding is None or not self.embedding.is_loaded():
            raise RuntimeError("Embedding pipeline must be loaded before indexing")

        if status_callback:
            status_callback(f"Scanning {directory}...")

        files = _collect_files(directory, extensions)
        if status_callback:
            status_callback(f"Found {len(files)} files to index")

        all_chunks = []
        all_meta = []

        for filepath, content in files:
            chunks = _chunk_text(content, chunk_size=chunk_size)
            rel_path = os.path.relpath(filepath, directory)
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_meta.append({
                    "file": rel_path,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                })

        if not all_chunks:
            if status_callback:
                status_callback("No content found to index")
            return 0

        if status_callback:
            status_callback(f"Embedding {len(all_chunks)} chunks...")

        # Batch embed
        batch_size = 64
        all_embeddings = []
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i:i + batch_size]
            result = self.embedding.run(texts=batch)
            if result.success:
                all_embeddings.append(result.content)
            else:
                _log.warning("Embedding batch failed: %s", result.error)

        if not all_embeddings:
            return 0

        embeddings = np.vstack(all_embeddings)
        self.store.add_documents(all_chunks, embeddings, all_meta)

        if status_callback:
            status_callback(f"Indexed {len(all_chunks)} chunks from {len(files)} files")

        _log.info("Indexed %d chunks from %d files in %s", len(all_chunks), len(files), directory)
        return len(all_chunks)

    def index_knowledge(self, knowledge_manager):
        """Index all knowledge entries for semantic search.

        Args:
            knowledge_manager: KnowledgeManager instance
        """
        if self.embedding is None or not self.embedding.is_loaded():
            raise RuntimeError("Embedding pipeline must be loaded before indexing")

        texts = []
        meta = []

        for store_name, store in [("session", knowledge_manager.session_store),
                                   ("reference", knowledge_manager.reference_store)]:
            if store is None:
                continue
            for entry_meta in store.list_entries():
                text = f"{entry_meta.get('tier1', '')} {entry_meta.get('category', '')}"
                texts.append(text)
                meta.append({"store": store_name, "id": entry_meta["id"]})

        if not texts:
            return 0

        result = self.embedding.run(texts=texts)
        if result.success:
            self.store.add_documents(texts, result.content, meta)
            return len(texts)
        return 0

    def augment_prompt(self, query, top_k=5):
        """Retrieve relevant context for a query.

        Args:
            query: User question or prompt
            top_k: Number of chunks to retrieve

        Returns:
            Formatted context string for injection into the system prompt
        """
        if self.embedding is None or not self.embedding.is_loaded():
            return ""

        if self.store.count == 0:
            return ""

        # Embed the query
        result = self.embedding.run(texts=[query])
        if not result.success:
            return ""

        query_emb = result.content[0]
        results = self.store.search(query_emb, top_k=top_k)

        if not results:
            return ""

        lines = ["[RAG CONTEXT — retrieved from indexed documents]"]
        for score, text, meta in results:
            source = meta.get("file", "knowledge")
            chunk_info = ""
            if "chunk_index" in meta:
                chunk_info = f" (chunk {meta['chunk_index'] + 1}/{meta['total_chunks']})"
            lines.append(f"\n--- {source}{chunk_info} [score: {score:.2f}] ---")
            lines.append(text.strip())

        return "\n".join(lines)

    def save(self, path):
        """Save the vector store to disk."""
        self.store.save(path)

    def load(self, path):
        """Load a previously saved vector store."""
        return self.store.load(path)
