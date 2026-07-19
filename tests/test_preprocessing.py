from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sentiment_system.preprocessing import clean_text  # noqa: E402


def test_clean_text_keeps_chinese_and_supported_punctuation() -> None:
    assert clean_text("  服务很好！值得推荐。123 ABC ") == "服务很好！值得推荐。"


def test_clean_text_handles_none_and_removes_whitespace() -> None:
    assert clean_text(None) == ""
    assert clean_text("物流\n 很 快") == "物流很快"
