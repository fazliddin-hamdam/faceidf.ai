import pytest
from pydantic import ValidationError

import main


def test_clean_query_extracts_photo_line() -> None:
    raw = "Here is the query:\na photo of a bearded man"
    assert main.clean_clip_query(raw) == "a photo of a bearded man"


def test_clean_query_strips_code_fence() -> None:
    raw = "```\na photo of a smiling woman\n```"
    assert main.clean_clip_query(raw) == "a photo of a smiling woman"


def test_clean_query_empty() -> None:
    assert main.clean_clip_query("   ") == ""


def test_normalize_cache_key() -> None:
    assert main.normalize_cache_key("  Esmer,   SAKALLI  ") == "esmer, sakallı"
    assert main.normalize_cache_key("ESMER, SAKALLI") == main.normalize_cache_key("esmer, sakallı")


def test_search_request_strips_whitespace() -> None:
    request = main.SearchRequest(text="  esmer  ")
    assert request.text == "esmer"


def test_search_request_rejects_whitespace_only() -> None:
    with pytest.raises(ValidationError):
        main.SearchRequest(text="   ")


def test_search_request_rejects_bad_top_k() -> None:
    with pytest.raises(ValidationError):
        main.SearchRequest(text="esmer", top_k=99)
