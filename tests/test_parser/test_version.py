"""Tests for version detection."""

import pytest


@pytest.mark.unit
class TestVersionDetection:
    def test_detect_version_from_meta(self):
        from stellaris_advisor.parser.version import detect_version

        assert detect_version({"version": "4.3.0.0"}) == "4.3"

    def test_detect_version_strips_patch(self):
        from stellaris_advisor.parser.version import detect_version

        assert detect_version({"version": "4.3.2.1"}) == "4.3"

    def test_unsupported_version_raises(self):
        from stellaris_advisor.parser.version import UnsupportedVersionError, detect_version

        with pytest.raises(UnsupportedVersionError, match="3.12"):
            detect_version({"version": "3.12.0.0"}, supported=("4.3",))

    def test_missing_version_key_raises(self):
        from stellaris_advisor.parser.version import detect_version

        with pytest.raises(KeyError):
            detect_version({})

    def test_two_part_version_string(self):
        from stellaris_advisor.parser.version import detect_version

        assert detect_version({"version": "4.3"}) == "4.3"

    def test_unsupported_version_error_attrs(self):
        from stellaris_advisor.parser.version import UnsupportedVersionError, detect_version

        with pytest.raises(UnsupportedVersionError) as exc_info:
            detect_version({"version": "3.0.0.0"}, supported=("4.3",))
        assert exc_info.value.detected == "3.0"
        assert exc_info.value.supported == ("4.3",)
