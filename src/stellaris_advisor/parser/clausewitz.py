"""Clausewitz format tokenizer and recursive descent parser for Stellaris save files."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any


class TokenType(Enum):
    """Token types produced by the Clausewitz tokenizer."""

    IDENTIFIER = auto()
    STRING = auto()
    NUMBER = auto()
    BOOL = auto()
    EQUALS = auto()
    LBRACE = auto()
    RBRACE = auto()
    EOF = auto()


@dataclass(frozen=True)
class Token:
    """A single token from the Clausewitz tokenizer."""

    type: TokenType
    value: str | int | float | bool
    line: int
    col: int


class TokenizeError(Exception):
    """Raised when the tokenizer encounters invalid input."""

    def __init__(self, message: str, line: int, col: int) -> None:
        self.line = line
        self.col = col
        super().__init__(f"Line {line}, col {col}: {message}")


# Characters valid inside an unquoted identifier (after the first character).
_IDENT_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    "_.:@'-"
)

_ESCAPE_MAP: dict[str, str] = {
    "\\": "\\",
    '"': '"',
    "n": "\n",
    "t": "\t",
}


def tokenize(text: str) -> Iterator[Token]:
    """Tokenize Clausewitz-format text into a stream of tokens.

    Yields Token objects for each meaningful element in the input.
    The final token is always EOF.
    """
    pos = 0
    line = 1
    col = 1
    length = len(text)

    while pos < length:
        ch = text[pos]

        # --- Whitespace ---
        if ch in (" ", "\t", "\r"):
            pos += 1
            col += 1
            continue

        # --- Newline ---
        if ch == "\n":
            pos += 1
            line += 1
            col = 1
            continue

        # --- Comment ---
        if ch == "#":
            while pos < length and text[pos] != "\n":
                pos += 1
            continue

        # --- Single-char tokens ---
        if ch == "=":
            yield Token(TokenType.EQUALS, "=", line, col)
            pos += 1
            col += 1
            continue

        if ch == "{":
            yield Token(TokenType.LBRACE, "{", line, col)
            pos += 1
            col += 1
            continue

        if ch == "}":
            yield Token(TokenType.RBRACE, "}", line, col)
            pos += 1
            col += 1
            continue

        # --- Quoted string ---
        if ch == '"':
            start_col = col
            pos += 1
            col += 1
            parts: list[str] = []
            while pos < length and text[pos] != '"':
                if text[pos] == "\\":
                    pos += 1
                    col += 1
                    if pos >= length:
                        raise TokenizeError("Unterminated escape sequence", line, col)
                    escaped = _ESCAPE_MAP.get(text[pos])
                    if escaped is None:
                        raise TokenizeError(
                            f"Unknown escape sequence: \\{text[pos]}", line, col
                        )
                    parts.append(escaped)
                elif text[pos] == "\n":
                    parts.append("\n")
                    line += 1
                    col = 0  # will be incremented below
                else:
                    parts.append(text[pos])
                pos += 1
                col += 1
            if pos >= length:
                raise TokenizeError("Unterminated string", line, start_col)
            pos += 1  # skip closing quote
            col += 1
            yield Token(TokenType.STRING, "".join(parts), line, start_col)
            continue

        # --- Number (starts with digit or '-' followed by digit) ---
        if ch.isdigit() or (ch == "-" and pos + 1 < length and text[pos + 1].isdigit()):
            start_col = col
            start_pos = pos
            is_negative = ch == "-"
            if is_negative:
                pos += 1
                col += 1
            while pos < length and text[pos].isdigit():
                pos += 1
                col += 1
            has_dot = False
            if pos < length and text[pos] == ".":
                has_dot = True
                pos += 1
                col += 1
                while pos < length and text[pos].isdigit():
                    pos += 1
                    col += 1

            # Check if followed by identifier chars — makes this a version-like
            # string (e.g. "4.3.0") which should be treated as an IDENTIFIER.
            if pos < length and text[pos] in _IDENT_CHARS and not text[pos].isdigit():
                while pos < length and text[pos] in _IDENT_CHARS:
                    pos += 1
                    col += 1
                raw = text[start_pos:pos]
                yield Token(TokenType.IDENTIFIER, raw, line, start_col)
                continue

            # Also treat as identifier if there are additional dots (e.g. "1.2.3")
            if has_dot and pos < length and text[pos] == ".":
                while pos < length and text[pos] in _IDENT_CHARS:
                    pos += 1
                    col += 1
                raw = text[start_pos:pos]
                yield Token(TokenType.IDENTIFIER, raw, line, start_col)
                continue

            raw = text[start_pos:pos]

            # Non-negative pure integers used as keys (followed by =) → IDENTIFIER
            if not is_negative and not has_dot:
                peek = pos
                while peek < length and text[peek] in (" ", "\t"):
                    peek += 1
                if peek < length and text[peek] == "=":
                    yield Token(TokenType.IDENTIFIER, raw, line, start_col)
                    continue
                yield Token(TokenType.NUMBER, int(raw), line, start_col)
            elif has_dot:
                yield Token(TokenType.NUMBER, float(raw), line, start_col)
            else:
                yield Token(TokenType.NUMBER, int(raw), line, start_col)
            continue

        # --- Identifier / bool / numeric key ---
        if ch in _IDENT_CHARS:
            start_col = col
            start_pos = pos
            while pos < length and text[pos] in _IDENT_CHARS:
                pos += 1
                col += 1
            raw = text[start_pos:pos]
            if raw == "yes":
                yield Token(TokenType.BOOL, True, line, start_col)
            elif raw == "no":
                yield Token(TokenType.BOOL, False, line, start_col)
            else:
                yield Token(TokenType.IDENTIFIER, raw, line, start_col)
            continue

        raise TokenizeError(f"Unexpected character: {ch!r}", line, col)

    yield Token(TokenType.EOF, "", line, col)


# ---------------------------------------------------------------------------
# Recursive descent parser
# ---------------------------------------------------------------------------


class ParseError(Exception):
    """Raised when the parser encounters an unexpected token sequence."""

    def __init__(self, message: str, line: int, col: int) -> None:
        self.line = line
        self.col = col
        super().__init__(f"Line {line}, col {col}: {message}")


class _DuplicateList(list):  # type: ignore[type-arg]
    """Marker subclass to distinguish duplicate-key lists from bare value lists."""


class _Parser:
    """Token-list based recursive descent parser for Clausewitz format."""

    __slots__ = ("_tokens", "_pos")

    def __init__(self, tokens: list[Token]) -> None:
        self._tokens = tokens
        self._pos = 0

    # -- helpers --

    def _peek(self) -> Token:
        return self._tokens[self._pos]

    def _advance(self) -> Token:
        tok = self._tokens[self._pos]
        self._pos += 1
        return tok

    def _expect(self, tt: TokenType) -> Token:
        tok = self._advance()
        if tok.type != tt:
            raise ParseError(
                f"Expected {tt.name}, got {tok.type.name} ({tok.value!r})",
                tok.line,
                tok.col,
            )
        return tok

    # -- grammar --

    def parse_top_level(self) -> dict[str, Any]:
        """Parse the entire token stream into a dict."""
        return self._parse_pairs(TokenType.EOF)

    def _parse_pairs(self, until: TokenType) -> dict[str, Any]:
        """Parse key=value pairs until *until* token type is seen."""
        result: dict[str, Any] = {}
        while self._peek().type != until:
            key_tok = self._advance()
            if key_tok.type not in (TokenType.IDENTIFIER, TokenType.STRING):
                raise ParseError(
                    f"Expected key, got {key_tok.type.name} ({key_tok.value!r})",
                    key_tok.line,
                    key_tok.col,
                )
            key = str(key_tok.value)
            self._expect(TokenType.EQUALS)
            value = self._parse_value()
            if key in result:
                existing = result[key]
                if isinstance(existing, _DuplicateList):
                    existing.append(value)
                else:
                    result[key] = _DuplicateList([existing, value])
            else:
                result[key] = value
        return result

    def _parse_value(self) -> Any:
        """Parse a single value (scalar, block, or list)."""
        tok = self._peek()
        if tok.type == TokenType.LBRACE:
            return self._parse_block_or_list()
        return self._advance().value

    def _parse_block_or_list(self) -> Any:
        """Parse { ... } as either a dict or a bare value list."""
        self._advance()  # consume LBRACE

        # Empty block
        if self._peek().type == TokenType.RBRACE:
            self._advance()
            return {}

        # Peek: if second token is EQUALS, this is a dict block.
        if self._pos + 1 < len(self._tokens) and self._tokens[self._pos + 1].type == TokenType.EQUALS:
            pairs = self._parse_pairs(TokenType.RBRACE)
            self._advance()  # consume RBRACE
            return pairs

        # Otherwise, bare value list.
        return self._parse_bare_list()

    def _parse_bare_list(self) -> list[Any]:
        """Parse bare values until RBRACE."""
        items: list[Any] = []
        while self._peek().type != TokenType.RBRACE:
            items.append(self._advance().value)
        self._advance()  # consume RBRACE
        return items


def parse(text: str) -> dict[str, Any]:
    """Parse Clausewitz-format text into a nested dict.

    Handles key=value pairs, nested blocks, bare value lists,
    and duplicate keys (collapsed into lists).

    Raises:
        ParseError: If the token stream has unexpected structure.
        TokenizeError: If the text cannot be tokenized.
    """
    tokens = list(tokenize(text))
    parser = _Parser(tokens)
    result = parser.parse_top_level()
    return _normalize(result)


def _normalize(obj: Any) -> Any:
    """Convert _DuplicateList markers back to plain lists, recursively."""
    if isinstance(obj, dict):
        return {k: _normalize(v) for k, v in obj.items()}
    if isinstance(obj, _DuplicateList):
        return [_normalize(item) for item in obj]
    if isinstance(obj, list):
        return [_normalize(item) for item in obj]
    return obj
