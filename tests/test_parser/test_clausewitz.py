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

        tokens = list(tokenize('# this is a comment\nname="Earth"'))
        non_eof = [t for t in tokens if t.type.name != "EOF"]
        assert len(non_eof) == 3

    def test_inline_comment_skipped(self):
        from stellaris_advisor.parser.clausewitz import tokenize

        tokens = list(tokenize('name="Earth" # inline comment'))
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


@pytest.mark.unit
class TestParser:
    def test_simple_key_value(self):
        from stellaris_advisor.parser.clausewitz import parse

        result = parse('name="Earth"')
        assert result == {"name": "Earth"}

    def test_integer_value(self):
        from stellaris_advisor.parser.clausewitz import parse

        result = parse("size=25")
        assert result == {"size": 25}

    def test_float_value(self):
        from stellaris_advisor.parser.clausewitz import parse

        result = parse("energy=142.50")
        assert result["energy"] == pytest.approx(142.50)

    def test_boolean_value(self):
        from stellaris_advisor.parser.clausewitz import parse

        result = parse("is_active=yes\nis_blocked=no")
        assert result == {"is_active": True, "is_blocked": False}

    def test_nested_block(self):
        from stellaris_advisor.parser.clausewitz import parse

        text = 'government={\n    type="gov_democratic"\n    authority="auth_democratic"\n}'
        result = parse(text)
        assert result["government"]["type"] == "gov_democratic"
        assert result["government"]["authority"] == "auth_democratic"

    def test_deeply_nested(self):
        from stellaris_advisor.parser.clausewitz import parse

        text = "a={\n    b={\n        c={\n            d=42\n        }\n    }\n}"
        result = parse(text)
        assert result["a"]["b"]["c"]["d"] == 42

    def test_bare_value_list_strings(self):
        from stellaris_advisor.parser.clausewitz import parse

        text = 'civics={ "civic_beacon_of_liberty" "civic_idealistic_foundation" }'
        result = parse(text)
        assert result["civics"] == [
            "civic_beacon_of_liberty",
            "civic_idealistic_foundation",
        ]

    def test_bare_value_list_integers(self):
        from stellaris_advisor.parser.clausewitz import parse

        text = "owned_planets={ 0 4 12 27 }"
        result = parse(text)
        assert result["owned_planets"] == [0, 4, 12, 27]

    def test_empty_block(self):
        from stellaris_advisor.parser.clausewitz import parse

        result = parse("empty={ }")
        assert result["empty"] == {}

    def test_duplicate_keys_become_list(self):
        from stellaris_advisor.parser.clausewitz import parse

        text = (
            "weight_modifier={\n    factor=1.5\n}\nweight_modifier={\n    factor=2.0\n}"
        )
        result = parse(text)
        assert isinstance(result["weight_modifier"], list)
        assert len(result["weight_modifier"]) == 2
        assert result["weight_modifier"][0]["factor"] == pytest.approx(1.5)
        assert result["weight_modifier"][1]["factor"] == pytest.approx(2.0)

    def test_integer_keys(self):
        from stellaris_advisor.parser.clausewitz import parse

        text = (
            "technology={\n"
            '    0={\n        technology="tech_corvettes"\n        level=1\n    }\n'
            '    1={\n        technology="tech_destroyers"\n        level=1\n    }\n'
            "}"
        )
        result = parse(text)
        assert result["technology"]["0"]["technology"] == "tech_corvettes"
        assert result["technology"]["1"]["technology"] == "tech_destroyers"

    def test_multiple_top_level_keys(self):
        from stellaris_advisor.parser.clausewitz import parse

        text = 'name="Earth"\nsize=25\nplanet_class="pc_continental"\nis_capital=yes'
        result = parse(text)
        assert result["name"] == "Earth"
        assert result["size"] == 25
        assert result["planet_class"] == "pc_continental"
        assert result["is_capital"] is True

    def test_identifier_value_unquoted(self):
        from stellaris_advisor.parser.clausewitz import parse

        result = parse("authority=auth_democratic")
        assert result["authority"] == "auth_democratic"

    def test_mixed_content(self):
        from stellaris_advisor.parser.clausewitz import parse

        text = """country={
    0={
        name="United Nations of Earth"
        government={
            type="gov_democratic"
            civics={ "civic_beacon_of_liberty" "civic_idealistic_foundation" }
        }
        owned_planets={ 0 4 12 }
        fleet_power=8200.50
        is_player=yes
    }
}"""
        result = parse(text)
        empire = result["country"]["0"]
        assert empire["name"] == "United Nations of Earth"
        assert empire["government"]["type"] == "gov_democratic"
        assert empire["government"]["civics"] == [
            "civic_beacon_of_liberty",
            "civic_idealistic_foundation",
        ]
        assert empire["owned_planets"] == [0, 4, 12]
        assert empire["fleet_power"] == pytest.approx(8200.50)
        assert empire["is_player"] is True

    def test_empty_input(self):
        from stellaris_advisor.parser.clausewitz import parse

        result = parse("")
        assert result == {}

    def test_comment_only(self):
        from stellaris_advisor.parser.clausewitz import parse

        result = parse("# just a comment\n# another comment")
        assert result == {}

    def test_bare_value_list_with_identifiers(self):
        from stellaris_advisor.parser.clausewitz import parse

        text = "ethics={ ethic_fanatic_egalitarian ethic_xenophile }"
        result = parse(text)
        assert result["ethics"] == ["ethic_fanatic_egalitarian", "ethic_xenophile"]

    def test_single_element_list(self):
        from stellaris_advisor.parser.clausewitz import parse

        text = "planets={ 7 }"
        result = parse(text)
        assert result["planets"] == [7]

    def test_triple_duplicate_keys(self):
        from stellaris_advisor.parser.clausewitz import parse

        text = "a=1\na=2\na=3"
        result = parse(text)
        assert result["a"] == [1, 2, 3]


@pytest.mark.unit
class TestParserProperties:
    @given(
        st.text(
            alphabet=st.sampled_from(
                "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
            ),
            min_size=1,
            max_size=20,
        )
    )
    def test_identifier_roundtrip(self, ident: str):
        from stellaris_advisor.parser.clausewitz import parse

        if not (ident[0].isalpha() or ident[0] == "_"):
            return
        if ident in ("yes", "no"):
            return
        text = f'{ident}="value"'
        result = parse(text)
        assert ident in result
        assert result[ident] == "value"

    @given(st.integers(min_value=-999999, max_value=999999))
    def test_integer_roundtrip(self, n: int):
        from stellaris_advisor.parser.clausewitz import parse

        text = f"key={n}"
        result = parse(text)
        assert result["key"] == n

    @given(st.lists(st.integers(min_value=0, max_value=9999), min_size=0, max_size=20))
    def test_integer_list_roundtrip(self, nums: list[int]):
        from stellaris_advisor.parser.clausewitz import parse

        inner = " ".join(str(n) for n in nums)
        text = f"items={{ {inner} }}"
        result = parse(text)
        if len(nums) == 0:
            assert result["items"] == {}
        else:
            assert result["items"] == nums
