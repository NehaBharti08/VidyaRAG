# Corpus Attribution & Licensing

VidyaRAG indexes openly licensed textbooks published by **OpenStax**, a nonprofit
educational initiative of Rice University.

The VidyaRAG source code is MIT licensed (see [LICENSE](LICENSE)). **The corpus is
not.** Textbook content — including passages returned by retrieval, quoted in
citations, or paraphrased in generated answers — remains under the Creative
Commons license below, and the attribution requirement travels with it.

---

## Indexed titles

| Title | Edition | Published | License | Source |
|---|---|---|---|---|
| Biology | 1st | 2016-10-21 | **CC BY 4.0** | https://openstax.org/details/books/biology |
| Anatomy and Physiology | 1st | 2013-04-25 | **CC BY 4.0** | https://openstax.org/details/books/anatomy-and-physiology |

Print ISBNs: Biology `978-1-938168-09-3` · Anatomy and Physiology `978-1-938168-13-0`.
OpenStax book UUIDs: `185cbf87-c72e-48f5-b51e-f14f21b5eabd` · `14fb4ad7-39a1-4eee-ab6e-3ef2482e3e22`.

### How the license was verified

Queried OpenStax's own content API, which is the publisher's authoritative record:

```
curl "https://openstax.org/apps/cms/api/v2/pages/?type=books.Book&fields=*&slug=biology"
```

Both titles return:

```
license_name    : Creative Commons Attribution License
license_version : 4.0
license_url     : https://creativecommons.org/licenses/by/4.0/
```

Full license text: https://creativecommons.org/licenses/by/4.0/

### ⚠️ Edition matters more than title — verify every addition

**Most OpenStax second editions are _not_ CC BY.** OpenStax relicensed much of its
catalog to CC BY-NC-SA 4.0, and the change is invisible from the title alone:

| Title | License |
|---|---|
| Biology **1st ed** | CC BY 4.0 ✅ |
| Biology **2e** | CC BY-**NC-SA** 4.0 ❌ |
| Anatomy and Physiology **1st ed** | CC BY 4.0 ✅ |
| Anatomy and Physiology **2e** | CC BY-**NC-SA** 4.0 ❌ |
| Microbiology | CC BY-**NC-SA** 4.0 ❌ |
| Concepts of Biology | CC BY-**NC-SA** 4.0 ❌ |

The NonCommercial and ShareAlike terms would restrict downstream reuse of this
repository and sit badly beside an MIT-licensed codebase. **Any title added later
must have its license checked individually, by edition, against the API above —
not assumed from a sibling edition and not taken from a search result.**

### Why these two

- **They are the only two CC BY 4.0 biology titles OpenStax publishes.** Every
  other life-science title in the catalog is NC-SA (table above), so this is the
  complete usable corpus, not a preference among many.
- **They are complementary rather than redundant.** *Biology* covers general
  biology — cell structure, genetics, evolution, ecology. *Anatomy and Physiology*
  covers human organ systems in far greater depth. Overlap is confined to
  *Biology*'s animal-systems unit.
- **The overlap that exists is shallow enough to keep evaluation honest.** Heavily
  redundant titles would make RAGAS context-precision ambiguous: a retrieved chunk
  would score as "irrelevant" only because a near-duplicate from the other book
  outranked it, and corpus redundancy would show up as retrieval error. This is
  the reason *Concepts of Biology* would have been excluded even if it were CC BY.
- **The seam between them produces genuine multi-hop questions** — e.g. membrane
  transport in *Biology* → nephron filtration in *Anatomy and Physiology* — which
  is exactly what the multi-hop slice of the gold set needs.

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
   `Biology, §7.3, p.214 (OpenStax, CC BY 4.0)` with a link to the free book.

> Download Biology for free at https://openstax.org/details/books/biology
>
> Download Anatomy and Physiology for free at https://openstax.org/details/books/anatomy-and-physiology

---

## Ingestion provenance

Raw PDFs are **not committed** to this repository. They are fetched and verified
by `vidyarag.ingest.download`, which records SHA-256 checksums, source URLs,
retrieval timestamps, and page counts to `data/raw/manifest.json` so any ingest
run is reproducible and auditable.

Checksums and page counts are populated by the Phase 1 ingestion run; see
[docs/EVALUATION.md](docs/EVALUATION.md) for corpus statistics.

---

## Disclaimer

VidyaRAG is an independent student project. It is **not affiliated with,
endorsed by, or sponsored by OpenStax or Rice University.** Generated answers are
produced by a language model and may be incorrect; they are not a substitute for
the source textbooks.
