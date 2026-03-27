"""Tests for Clausewitz format tokenizer and parser."""

import pytest
from hypothesis import given, strategies as st


@pytest.mark.unit
class TestTokenizer:
    def test_simple_key_value(self):
        from stellaris_advisor.parser.clausewitz import TokenType, tokenize

        tokens = list(tokenize('name="Earth"'))
        assert len(tokens) == 4  # IDENTIFIER, EQUALS, STRING, EOF
        assert tokens[0].type == TokenType.IDENTIFIER
        assert tokens[0].value == "name"
        assert tokens[1].type == TokenType.EQUALS
        assert tokens[2].type == TokenType.STRING
        assert tokens[2].value == "Earth"
        assert tokens[3].type == TokenType.EOF

    def test_integer_value(self):
        from stellaris_advisor.parser.clausewitz import TokenType, tokenize

        tokens = list(tokenize("size=25"))
        assert tokens[0].value == "size"
        assert tokens[2].type == TokenType.NUMBER
        assert tokens[2].value == 25

    def test_float_value(self):
        from stellaris_advisor.parser.clausewitz import TokenType, tokenize

        tokens = list(tokenize("energy=142.50"))
        assert tokens[2].type == TokenType.NUMBER
        assert tokens[2].value == pytest.approx(142.50)

    def test_negative_number(self):
        from stellaris_advisor.parser.clausewitz import TokenType, tokenize

        tokens = list(tokenize("modifier=-0.25"))
        assert tokens[2].type == TokenType.NUMBER
        assert tokens[2].value == pytest.approx(-0.25)

    def test_boolean_yes(self):
        from stellaris_advisor.parser.clausewitz import TokenType, tokenize

        tokens = list(tokenize("is_active=yes"))
        assert tokens[2].type == TokenType.BOOL
        assert tokens[2].value is True

    def test_boolean_no(self):
        from stellaris_advisor.parser.clausewitz import TokenType, tokenize

        tokens = list(tokenize("is_active=no"))
        assert tokens[2].type == TokenType.BOOL
        assert tokens[2].value is False

    def test_braces(self):
        from stellaris_advisor.parser.clausewitz import TokenType, tokenize

        tokens = list(tokenize("block={ }"))
        types = [t.type for t in tokens]
        assert TokenType.LBRACE in types
        assert TokenType.RBRACE in types

    def test_comment_skipped(self):
        from stellaris_advisor.parser.clausewitz import tokenize

        tokens = list(tokenize("# this is a comment\nname=\"Earth\""))
        non_eof = [t for t in tokens if t.type.name != "EOF"]
        assert len(non_eof) == 3

    def test_inline_comment_skipped(self):
        from stellaris_advisor.parser.clausewitz import tokenize

        tokens = list(tokenize("name=\"Earth\" # inline comment"))
        non_eof = [t for t in tokens if t.type.name != "EOF"]
        assert len(non_eof) == 3

    def test_multiline(self):
        from stellaris_advisor.parser.clausewitz import tokenize

        text = 'name="Earth"\nsize=25\nis_capital=yes'
        tokens = list(tokenize(text))
        non_eof = [t for t in tokens if t.type.name != "EOF"]
        assert len(non_eof) == 9

    def test_integer_key(self):
        from stellaris_advisor.parser.clausewitz import TokenType, tokenize

        tokens = list(tokenize("0={ }"))
        assert tokens[0].type == TokenType.IDENTIFIER
        assert tokens[0].value == "0"

    def test_empty_string(self):
        from stellaris_advisor.parser.clausewitz import TokenType, tokenize

        tokens = list(tokenize('name=""'))
        assert tokens[2].type == TokenType.STRING
        assert tokens[2].value == ""

    def test_string_with_spaces(self):
        from stellaris_advisor.parser.clausewitz import tokenize

        tokens = list(tokenize('name="United Nations of Earth"'))
        assert tokens[2].value == "United Nations of Earth"

    def test_unquoted_identifier_value(self):
        from stellaris_advisor.parser.clausewitz import TokenType, tokenize

        tokens = list(tokenize("type=gov_democratic"))
        assert tokens[2].type == TokenType.IDENTIFIER
        assert tokens[2].value == "gov_democratic"

    def test_line_tracking(self):
        from stellaris_advisor.parser.clausewitz import tokenize

        text = "a=1\nb=2\nc=3"
        tokens = list(tokenize(text))
        lines = [t.line for t in tokens if t.value in ("a", "b", "c")]
        assert lines == [1, 2, 3]

    def test_empty_input(self):
        from stellaris_advisor.parser.clausewitz import TokenType, tokenize

        tokens = list(tokenize(""))
        assert len(tokens) == 1
        assert tokens[0].type == TokenType.EOF

    def test_tabs_and_mixed_whitespace(self):
        from stellaris_advisor.parser.clausewitz import tokenize

        text = '\t name = \t "Earth" \t'
        tokens = list(tokenize(text))
        non_eof = [t for t in tokens if t.type.name != "EOF"]
        assert len(non_eof) == 3
