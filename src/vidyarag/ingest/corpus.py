"""The indexed corpus, declared once.

This module is the single source of truth for which books VidyaRAG indexes and
under what licence. ATTRIBUTION.md documents the same facts for humans; if the
two ever disagree, this file is what actually ran.

Every field here ends up somewhere load-bearing:

* ``license_name`` / ``license_url`` / ``source_url`` are copied onto every
  chunk's payload, so attribution survives retrieval instead of being bolted on
  at the UI layer.
* ``citation_title`` is what a reader sees in ``Biology, S7.3, p.214``.
* ``book_uuid`` and ``print_isbn_13`` pin *which edition* was ingested, which
  matters more than usual here -- see the licence note below.

Licence, verified per title *and per edition* against OpenStax's content API::

    curl "https://openstax.org/apps/cms/api/v2/pages/?type=books.Book&fields=*&slug=biology"

OpenStax relicensed much of its catalogue at the second edition, so the licence
cannot be inferred from the title. ``Biology 2e``, ``Anatomy and Physiology
2e``, ``Microbiology`` and ``Concepts of Biology`` are all CC BY-NC-SA 4.0. Only
the *first* editions below remain CC BY 4.0, which makes them the complete
usable biology corpus rather than a preference among many. Any title added
later must be re-checked against that API by edition.
"""

from __future__ import annotations

from dataclasses import dataclass

CC_BY_4_0 = "CC BY 4.0"
CC_BY_4_0_URL = "https://creativecommons.org/licenses/by/4.0/"


@dataclass(frozen=True, slots=True)
class BookSpec:
    """Immutable description of one source textbook."""

    slug: str
    """Stable identifier. Used for filenames, chunk ids, and manifest keys."""

    title: str
    edition: str
    published: str
    """ISO date of the edition OpenStax lists, for provenance."""

    source_url: str
    """Human-facing book page. Rendered in citations and attribution."""

    pdf_url: str
    """Direct asset URL for the high-resolution PDF."""

    book_uuid: str
    print_isbn_13: str

    license_name: str = CC_BY_4_0
    license_url: str = CC_BY_4_0_URL
    publisher: str = "OpenStax, Rice University"

    @property
    def citation_title(self) -> str:
        """Short title as it appears in a rendered citation."""
        return self.title

    @property
    def filename(self) -> str:
        """Local filename for the downloaded PDF."""
        return f"{self.slug}.pdf"

    def attribution(self) -> str:
        """One-line CC BY attribution string, as the licence requires."""
        return (
            f"{self.title} ({self.edition} edition) by {self.publisher}, "
            f"licensed {self.license_name}. Free at {self.source_url}"
        )


BIOLOGY = BookSpec(
    slug="biology",
    title="Biology",
    edition="1st",
    published="2016-10-21",
    source_url="https://openstax.org/details/books/biology",
    pdf_url="https://assets.openstax.org/oscms-prodcms/media/documents/Biology-OP_xQoZM8Z.pdf",
    book_uuid="185cbf87-c72e-48f5-b51e-f14f21b5eabd",
    print_isbn_13="978-1-938168-09-3",
)

ANATOMY_AND_PHYSIOLOGY = BookSpec(
    slug="anatomy-and-physiology",
    title="Anatomy and Physiology",
    edition="1st",
    published="2013-04-25",
    source_url="https://openstax.org/details/books/anatomy-and-physiology",
    pdf_url="https://assets.openstax.org/oscms-prodcms/media/documents/AnatomyandPhysiology-OP.pdf",
    book_uuid="14fb4ad7-39a1-4eee-ab6e-3ef2482e3e22",
    print_isbn_13="978-1-938168-13-0",
)

CORPUS: tuple[BookSpec, ...] = (BIOLOGY, ANATOMY_AND_PHYSIOLOGY)
"""The indexed corpus, in a stable order.

Two titles rather than three: these are the only CC BY 4.0 biology books
OpenStax publishes. They are also complementary -- general biology against human
organ systems -- with overlap confined to Biology's animal-systems unit. That
shallow seam is what makes genuine cross-title multi-hop questions possible,
while avoiding the redundancy that would make context-precision ambiguous.
"""


def get_book(slug: str) -> BookSpec:
    """Look up a book by slug.

    Args:
        slug: Stable book identifier, e.g. ``"biology"``.

    Returns:
        The matching :class:`BookSpec`.

    Raises:
        KeyError: If no book has that slug, listing the ones that do.
    """
    for book in CORPUS:
        if book.slug == slug:
            return book
    available = ", ".join(b.slug for b in CORPUS)
    raise KeyError(f"Unknown book {slug!r}. Available: {available}")
