import hashlib
import hmac
import time
from urllib.parse import urlencode

import pytest
from fastapi import HTTPException

import main


def _signed_init_data(fields: dict[str, str], token: str = "123456:TEST_TOKEN") -> str:
    data_check_string = "\n".join(f"{key}={value}" for key, value in sorted(fields.items()))
    secret = hmac.new(b"WebAppData", token.encode(), hashlib.sha256).digest()
    digest = hmac.new(secret, data_check_string.encode(), hashlib.sha256).hexdigest()
    return urlencode({**fields, "hash": digest})


@pytest.fixture(autouse=True)
def _bot_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main, "TELEGRAM_BOT_TOKEN", "123456:TEST_TOKEN")


def test_valid_init_data_is_accepted() -> None:
    init_data = _signed_init_data(
        {"auth_date": str(int(time.time())), "user": '{"id":1}'}
    )
    main.verify_telegram_init_data(init_data)


def test_old_init_data_is_rejected() -> None:
    init_data = _signed_init_data(
        {"auth_date": str(int(time.time()) - main.INIT_DATA_MAX_AGE - 60), "user": '{"id":1}'}
    )
    with pytest.raises(HTTPException) as exc:
        main.verify_telegram_init_data(init_data)
    assert exc.value.status_code == 401


def test_malformed_auth_date_is_rejected_without_500() -> None:
    init_data = _signed_init_data({"auth_date": "abc", "user": '{"id":1}'})
    with pytest.raises(HTTPException) as exc:
        main.verify_telegram_init_data(init_data)
    assert exc.value.status_code == 401


def test_future_init_data_is_rejected() -> None:
    init_data = _signed_init_data(
        {"auth_date": str(int(time.time()) + 86400), "user": '{"id":1}'}
    )
    with pytest.raises(HTTPException) as exc:
        main.verify_telegram_init_data(init_data)
    assert exc.value.status_code == 401


def test_bad_signature_is_rejected() -> None:
    init_data = "auth_date=1&user=%7B%7D&hash=deadbeef"
    with pytest.raises(HTTPException) as exc:
        main.verify_telegram_init_data(init_data)
    assert exc.value.status_code == 401
