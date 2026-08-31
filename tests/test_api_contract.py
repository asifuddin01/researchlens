"""The request shapes the API promises.

No server is started here. What these guard is the part that can silently
inverte: the strings a client sends and what they mean once inside the engine.
`live="never"` reaching the engine as `None` would turn "search only the
indexed papers" into "decide for yourself", and nothing downstream would
notice — the answer would simply contain evidence the reader asked not to have.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from researchlens.api.main import AskRequest, SearchRequest, _SESSION_OK

#: The mapping the endpoints apply. Kept here rather than imported so that
#: changing it in main.py without meaning to fails a test rather than a reader.
LIVE = {"auto": None, "always": True, "never": False}


def test_live_defaults_to_auto():
    """Auto decides from the shape of the question, which is the only default
    the engine can justify."""
    assert AskRequest(question="what is scGPT").live == "auto"
    assert LIVE[AskRequest(question="what is scGPT").live] is None


@pytest.mark.parametrize(
    "value,expected", [("auto", None), ("always", True), ("never", False)]
)
def test_live_maps_to_the_engine_flag(value, expected):
    """False and None are different instructions and must not collapse: one
    says do not search, the other says decide."""
    req = AskRequest(question="what is current in scRNA-seq", live=value)
    assert LIVE[req.live] is expected


def test_an_unknown_live_value_is_refused():
    with pytest.raises(ValidationError):
        AskRequest(question="what is scGPT", live="sometimes")


def test_a_question_must_be_long_enough_to_retrieve_against():
    with pytest.raises(ValidationError):
        AskRequest(question="hi")


def test_an_unknown_provider_is_refused_before_the_engine_sees_it():
    with pytest.raises(ValidationError):
        AskRequest(question="what is scGPT", provider="gpt-5")


def test_search_carries_the_same_subset_argument_as_ask():
    """A reader who confines one to chosen papers expects the other to obey the
    same words."""
    s = SearchRequest(query="scGPT", doc_ids=["abc123"], session="a" * 12)
    a = AskRequest(question="what is scGPT", doc_ids=["abc123"], session="a" * 12)
    assert s.doc_ids == a.doc_ids
    assert s.session == a.session


# ---- session ids -----------------------------------------------------------

def test_a_session_id_is_checked_for_shape_not_trusted():
    """It is the only thing separating one reader's uploaded papers from
    another's, so a caller who sends a path or a wildcard gets a 422."""
    for bad in ("../../etc/passwd", "*", "a b c d", "short", "x" * 65):
        with pytest.raises(ValidationError):
            AskRequest(question="what is scGPT", session=bad)


def test_the_path_and_body_agree_on_what_a_session_looks_like():
    """`/ingest/{session}` checks with a regex because pydantic is not in that
    path; the two must not disagree about what is acceptable."""
    good = "aB3-_xyz90ab"
    assert _SESSION_OK.match(good)
    assert AskRequest(question="what is scGPT", session=good).session == good
    for bad in ("../etc", "short", "has space", "x" * 65):
        assert not _SESSION_OK.match(bad)
