from pathlib import Path

import process_faces


def test_point_id_is_deterministic() -> None:
    first = process_faces.point_id("Ahmet/Ahmet_0001.jpg")
    second = process_faces.point_id("Ahmet/Ahmet_0001.jpg")
    assert first == second
    assert first != process_faces.point_id("Ahmet/Ahmet_0002.jpg")


def test_find_images_filters_extensions(tmp_path: Path) -> None:
    (tmp_path / "a.jpg").write_bytes(b"x")
    (tmp_path / "b.PNG").write_bytes(b"x")
    (tmp_path / "not-image.txt").write_text("x", encoding="utf-8")
    found = process_faces.find_images(tmp_path)
    assert [path.name for path in found] == ["a.jpg", "b.PNG"]
