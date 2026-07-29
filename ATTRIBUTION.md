# Corpus Attribution & Licensing

VidyaRAG indexes openly licensed textbooks published by **OpenStax**, a nonprofit
educational initiative of Rice University.

The VidyaRAG source code is MIT licensed (see [LICENSE](LICENSE)). **The corpus is
not.** Textbook content — including passages returned by retrieval, quoted in
citations, or paraphrased in generated answers — remains under the Creative
Commons license below, and the attribution requirement travels with it.

---

## Indexed titles

| Title | Edition | Publisher | License | Source |
|---|---|---|---|---|
| Biology 2e | 2nd | OpenStax, Rice University | **CC BY 4.0** | https://openstax.org/details/books/biology-2e |
| Microbiology | 1st | OpenStax, Rice University | **CC BY 4.0** | https://openstax.org/details/books/microbiology |

License verified against each title's preface, which states:

> "<Title> is licensed under a Creative Commons Attribution 4.0 International
> (CC BY) license, which means that you can distribute, remix, and build upon
> the content, as long as you provide attribution to OpenStax and its content
> contributors."

Full license text: https://creativecommons.org/licenses/by/4.0/

### Why these two, and not others

- **Concepts of Biology was deliberately excluded.** It is also CC BY 4.0 and was
  an obvious third candidate, but its content substantially overlaps Biology 2e.
  Indexing both would let the same fact be retrieved from two titles, which makes
  RAGAS context-precision scores ambiguous — a retrieved chunk would be
  "irrelevant" only because a near-duplicate outranked it. Corpus redundancy would
  have measured as retrieval error.
- **Biology 2e + Microbiology are complementary**, giving genuine cross-title
  multi-hop questions (e.g. membrane structure in general biology → antibiotic
  mechanism in microbiology) rather than redundant ones.
- **Not all OpenStax titles are CC BY 4.0.** Some, including several AP-branded
  editions, are CC BY-NC-SA 4.0. The NonCommercial and ShareAlike terms would
  restrict downstream reuse of this repo, so any title added later must have its
  license verified individually before ingestion.

---

## Required attribution

Per CC BY 4.0 and OpenStax's stated terms, any redistribution of this content
must credit OpenStax, name the title, and link to the free version.

VidyaRAG satisfies this in three places:

1. **This file** — canonical, machine-readable record of titles and licenses.
2. **Every retrieved chunk** carries `book_title`, `license`, and `source_url` in
   its Qdrant payload, so attribution survives retrieval rather than being bolted
   on at the UI layer.
3. **Every generated answer** renders citations as
   `Biology 2e, §7.3, p.214 (OpenStax, CC BY 4.0)` with a link to the free book.

> Download Biology 2e for free at https://openstax.org/details/books/biology-2e
>
> Download Microbiology for free at https://openstax.org/details/books/microbiology

---

## Ingestion provenance

Raw PDFs are **not committed** to this repository. They are fetched and verified
by `vidyarag.ingest.download`, which records SHA-256 checksums, retrieval
timestamps, and page counts to `data/raw/manifest.json` so any ingest run is
reproducible and auditable.

Checksums and page counts are populated by the Phase 1 ingestion run; see
[docs/EVALUATION.md](docs/EVALUATION.md) for corpus statistics.

---

## Disclaimer

VidyaRAG is an independent student project. It is **not affiliated with,
endorsed by, or sponsored by OpenStax or Rice University.** Generated answers are
produced by a language model and may be incorrect; they are not a substitute for
the source textbooks.
