"""VidyaRAG: an agentic, self-correcting RAG study assistant over OpenStax textbooks."""

# The only place the version is written. pyproject.toml reads it (hatch
# dynamic version) and the API reports it. A literal rather than
# importlib.metadata on purpose: the Hugging Face Space imports this package
# from src/ without installing it, where package metadata does not exist.
__version__ = "1.2.1"

__all__ = ["__version__"]
