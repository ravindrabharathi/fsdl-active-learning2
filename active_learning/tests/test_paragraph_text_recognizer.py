"""Test for paragraph_active_learning module."""
import json
import os
import time
from pathlib import Path

import editdistance

from active_learning.paragraph_active_learning import ParagraphTextRecognizer

os.environ["CUDA_VISIBLE_DEVICES"] = ""


_FILE_DIRNAME = Path(__file__).parents[0].resolve()
_SUPPORT_DIRNAME = _FILE_DIRNAME / "support" / "paragraphs"

# restricting number of samples to prevent CirleCI running out of time
_NUM_MAX_SAMPLES = 1 if os.environ.get("CIRCLECI", False) else 100


def test_paragraph_active_learning():
    """Test ParagraphTextRecognizer."""
    support_filenames = list(_SUPPORT_DIRNAME.glob("*.png"))
    with open(_SUPPORT_DIRNAME / "data_by_file_id.json", "r") as f:
        support_data_by_file_id = json.load(f)

    for i, support_filename in enumerate(support_filenames):
        if i >= _NUM_MAX_SAMPLES:
            break
        expected_text = support_data_by_file_id[support_filename.stem]["predicted_text"]
        start_time = time.time()
        predicted_text = _test_paragraph_active_learning(support_filename, expected_text)
        end_time = time.time()
        time_taken = round(end_time - start_time, 2)

        cer = _character_error_rate(support_data_by_file_id[support_filename.stem]["ground_truth_text"], predicted_text)
        print(f"Character error rate is {round(cer, 3)} for file {support_filename.name} (time taken: {time_taken}s)")


def _test_paragraph_active_learning(image_filename: Path, expected_text: str):
    """Test ParagraphTextRecognizer on 1 image."""
    active_learning = ParagraphTextRecognizer()
    predicted_text = active_learning.predict(image_filename)
    assert predicted_text == expected_text, f"predicted text does not match expected for {image_filename.name}"
    return predicted_text


def _character_error_rate(str_a: str, str_b: str) -> float:
    """Return character error rate."""
    return editdistance.eval(str_a, str_b) / max(len(str_a), len(str_b))
