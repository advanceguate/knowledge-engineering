import pytest

from ke.contracts import Evidence
from ke.grounding import GroundingError, ground_evidence, quote_sha256, verify_evidence


def evidence(quote: str = "The map is not the territory.") -> Evidence:
    return Evidence(
        source_id="source-1",
        chunk_id="chunk-1",
        heading_path=["Introduction"],
        page_start=1,
        page_end=1,
        quote=quote,
        quote_sha256=quote_sha256(quote),
    )


def test_verifies_exact_quote_and_hash() -> None:
    result = verify_evidence(evidence(), "Before. The map is not the territory. After.")

    assert result.verified is True
    assert result.quote_sha256 == quote_sha256(result.quote)


def test_rejects_quote_not_in_attributed_chunk() -> None:
    with pytest.raises(GroundingError, match="was not found"):
        verify_evidence(evidence(), "A different statement.")


def test_rejects_mismatched_quote_hash() -> None:
    item = evidence().model_copy(update={"quote_sha256": "not-the-quote-hash"})

    with pytest.raises(GroundingError, match="quote_sha256"):
        verify_evidence(item, item.quote)


def test_non_strict_grounding_omits_unsupported_evidence() -> None:
    items = [evidence(), evidence("Unsupported")]

    grounded = ground_evidence(items, {"chunk-1": "The map is not the territory."}, strict=False)

    assert [item.quote for item in grounded] == ["The map is not the territory."]
