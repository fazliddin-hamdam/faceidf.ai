import json

from translation_cache import TranslationCache


def test_set_get_roundtrip(tmp_path) -> None:
    cache = TranslationCache(tmp_path / "cache.json", ttl_seconds=60)
    cache.set("esmer", "a photo of a dark haired man")
    assert cache.get("esmer") == "a photo of a dark haired man"
    assert cache.get("yok") is None


def test_ttl_expires_items(tmp_path) -> None:
    now = [1000.0]
    cache = TranslationCache(tmp_path / "cache.json", ttl_seconds=10, clock=lambda: now[0])
    cache.set("esmer", "query")
    now[0] += 11
    assert cache.get("esmer") is None


def test_lru_evicts_oldest(tmp_path) -> None:
    cache = TranslationCache(tmp_path / "cache.json", max_size=2, ttl_seconds=60)
    cache.set("a", "1")
    cache.set("b", "2")
    cache.set("c", "3")
    assert cache.get("a") is None
    assert cache.get("b") == "2"


def test_persists_and_loads(tmp_path) -> None:
    path = tmp_path / "cache.json"
    cache = TranslationCache(path, ttl_seconds=60)
    cache.set("esmer", "query")
    cache.save()

    loaded = TranslationCache(path, ttl_seconds=60)
    loaded.load()
    assert loaded.get("esmer") == "query"


def test_corrupt_file_is_ignored(tmp_path) -> None:
    path = tmp_path / "cache.json"
    path.write_text("{bozuk", encoding="utf-8")
    cache = TranslationCache(path)
    cache.load()
    assert len(cache) == 0


def test_load_skips_expired_entries(tmp_path) -> None:
    path = tmp_path / "cache.json"
    path.write_text(
        json.dumps({"esmer": {"value": "query", "expires_at": 1}}), encoding="utf-8"
    )
    cache = TranslationCache(path, ttl_seconds=60)
    cache.load()
    assert len(cache) == 0
