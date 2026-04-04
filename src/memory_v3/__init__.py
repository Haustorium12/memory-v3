"""memory-v3: Next-generation brain-inspired persistent memory for AI coding assistants."""

__version__ = "1.0.0"
__author__ = "Sean Pembroke"

_embedder = None


def get_embedder():
    """
    Lazy-load and warm up the embedding function.
    Returns the embed_text callable for use throughout the system.
    """
    global _embedder
    if _embedder is None:
        from .embeddings import embed_text

        embed_text("warmup")
        _embedder = embed_text
    return _embedder


def get_version_info() -> dict:
    """Return version metadata for diagnostics."""
    return {
        "name": "memory-v3",
        "version": __version__,
        "author": __author__,
        "description": "Brain-inspired persistent memory with governance layers, "
                       "sensory gating, and multi-graph knowledge representation.",
    }
