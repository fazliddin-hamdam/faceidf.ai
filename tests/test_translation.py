import asyncio
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

import main
from translation_cache import TranslationCache


def _fake_gemini(models: object) -> SimpleNamespace:
    return SimpleNamespace(aio=SimpleNamespace(models=models))


def test_translation_is_cached(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    cache = TranslationCache(tmp_path / "cache.json", ttl_seconds=60)
    monkeypatch.setattr(main.app.state, "translation_cache", cache, raising=False)

    class Models:
        calls = 0

        async def generate_content(self, **kwargs):
            self.calls += 1
            return SimpleNamespace(text="```\na photo of a bearded man\n```")

    models = Models()
    monkeypatch.setattr(main.app.state, "gemini", _fake_gemini(models), raising=False)

    first = asyncio.run(main.translate_to_clip_query("esmer, sakallı"))
    second = asyncio.run(main.translate_to_clip_query("  Esmer,   SAKALLI "))

    assert first == "a photo of a bearded man"
    assert second == first
    assert models.calls == 1


def test_translation_timeout_returns_504(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    cache = TranslationCache(tmp_path / "cache.json", ttl_seconds=60)
    monkeypatch.setattr(main.app.state, "translation_cache", cache, raising=False)
    monkeypatch.setattr(main, "TRANSLATION_TIMEOUT_SECONDS", 0.05)

    class SlowModels:
        async def generate_content(self, **kwargs):
            await asyncio.sleep(1)

    monkeypatch.setattr(main.app.state, "gemini", _fake_gemini(SlowModels()), raising=False)

    with pytest.raises(HTTPException) as exc:
        asyncio.run(main.translate_to_clip_query("yeni bir tarif"))

    assert exc.value.status_code == 504
