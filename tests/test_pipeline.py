from pathlib import Path

from PIL import Image

from vision_pipeline.pipeline import save_image_atomic


def test_save_image_atomic_preserves_image_extension(tmp_path: Path) -> None:
    destination = tmp_path / "latest.jpg"
    image = Image.new("RGB", (8, 8), color="white")

    save_image_atomic(image, destination, quality=82)

    assert destination.exists()
    assert not (tmp_path / ".latest.tmp.jpg").exists()
    with Image.open(destination) as saved_image:
        assert saved_image.format == "JPEG"