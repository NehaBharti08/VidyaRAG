"""Corpus ingestion: fetch, parse, chunk.

The ingestion path is deliberately separable from retrieval. It runs rarely,
costs money once, and its output -- chunks with accurate (book, chapter,
section, page) metadata -- is what every later phase depends on. A citation can
only be as trustworthy as the metadata attached during ingestion.
"""

from vidyarag.ingest.chunk import Chunk, chunk_pages, split_sentences
from vidyarag.ingest.corpus import CORPUS, BookSpec, get_book
from vidyarag.ingest.download import (
    DownloadRecord,
    download_book,
    download_corpus,
    load_manifest,
)
from vidyarag.ingest.parse import (
    OutlineEntry,
    PageText,
    build_structure_map,
    extract_pages,
    load_outline,
)

__all__ = [
    "CORPUS",
    "BookSpec",
    "Chunk",
    "DownloadRecord",
    "OutlineEntry",
    "PageText",
    "build_structure_map",
    "chunk_pages",
    "download_book",
    "download_corpus",
    "extract_pages",
    "get_book",
    "load_manifest",
    "load_outline",
    "split_sentences",
]
