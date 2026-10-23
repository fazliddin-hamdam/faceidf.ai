from types import SimpleNamespace

import main


def _hit(person: str, score: float, path: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        id=path or person,
        score=score,
        payload={"person": person, "path": path or f"{person}.jpg"},
    )


def test_dedupe_keeps_best_hit_per_person() -> None:
    hits = [
        _hit("Ayse", 0.34, "a1.jpg"),
        _hit("Ayse", 0.31, "a2.jpg"),
        _hit("Mehmet", 0.33, "m1.jpg"),
    ]
    result = main.dedupe_hits(hits, top_k=5, min_score=0.2)
    assert [item.payload["path"] for item in result] == ["a1.jpg", "m1.jpg"]


def test_dedupe_filters_low_scores() -> None:
    hits = [_hit("Ayse", 0.19), _hit("Mehmet", 0.25)]
    result = main.dedupe_hits(hits, top_k=5, min_score=0.2)
    assert [item.payload["person"] for item in result] == ["Mehmet"]


def test_dedupe_respects_top_k() -> None:
    hits = [_hit(f"Kisi{i}", 0.3 - i * 0.01) for i in range(5)]
    result = main.dedupe_hits(hits, top_k=2, min_score=0.0)
    assert len(result) == 2


def test_dedupe_handles_missing_payload() -> None:
    hit = SimpleNamespace(id="x", score=0.3, payload=None)
    result = main.dedupe_hits([hit], top_k=1, min_score=0.2)
    assert len(result) == 1
