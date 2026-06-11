from PIL import Image

from vision_pipeline.models import Detection
from vision_pipeline.vlm import TemplateDescriber, build_prompt, extract_generated_text


def test_template_describer_includes_image_size_and_confidence() -> None:
    image = Image.new("RGB", (1280, 720), color="white")
    detections = [Detection("person", 0.91, (10.0, 20.0, 50.0, 80.0))]

    description = TemplateDescriber().describe(image, detections)

    assert "1280x720" in description
    assert "person" in description
    assert "91%" in description


def test_build_prompt_includes_detector_details() -> None:
    image = Image.new("RGB", (640, 480), color="white")
    detections = [Detection("person", 0.86, (1.0, 2.0, 30.0, 40.0))]

    prompt = build_prompt(image, detections, "person")

    assert "640x480" in prompt
    assert "Detector summary: person" in prompt
    assert "person 86% bbox=(1,2,30,40)" in prompt


def test_extract_generated_text_from_chat_response() -> None:
    result = [{"generated_text": [{"role": "assistant", "content": "A person is visible."}]}]

    assert extract_generated_text(result) == "A person is visible."