"""A small, pure-Python compatibility layer for OpenAI Whisper's tokenizer.

It implements the limited ``tiktoken.Encoding`` interface used by the
vendored OpenAI Whisper package.  Vocabulary ranks come from Whisper's own
``*.tiktoken`` files, so no compiled tiktoken wheel or network access is
required at inference time.
"""

from __future__ import annotations

import re
from typing import Iterable, Mapping


# Equivalent for Whisper's use to tiktoken's ``\p{L}`` / ``\p{N}`` pattern,
# using only Python's standard-library Unicode-aware regex engine.
_PATTERN = re.compile(
    r"'s|'t|'re|'ve|'m|'ll|'d| ?[^\W\d_]+| ?\d+| ?[^\s\w]+|\s+(?!\S)|\s+"
)


class Encoding:
    """Byte-pair encoder compatible with the API OpenAI Whisper calls."""

    def __init__(
        self,
        *,
        name: str,
        pat_str: str,
        mergeable_ranks: Mapping[bytes, int],
        special_tokens: Mapping[str, int],
        explicit_n_vocab: int | None = None,
    ) -> None:
        self.name = name
        self._ranks = dict(mergeable_ranks)
        self._special_tokens = dict(special_tokens)
        self.special_tokens_set = frozenset(self._special_tokens)
        self._id_to_bytes = {rank: token for token, rank in self._ranks.items()}
        self._id_to_special = {rank: token for token, rank in self._special_tokens.items()}
        self.eot_token = self._special_tokens["<|endoftext|>"]
        if explicit_n_vocab is not None and explicit_n_vocab != len(self._ranks) + len(self._special_tokens):
            raise ValueError("Vocabulary size does not match mergeable and special tokens")

    def encode_single_token(self, token: str | bytes) -> int:
        if isinstance(token, str) and token in self._special_tokens:
            return self._special_tokens[token]
        value = token.encode("utf-8") if isinstance(token, str) else token
        try:
            return self._ranks[value]
        except KeyError as error:
            raise KeyError(f"Token is not present in the Whisper vocabulary: {token!r}") from error

    def _encode_piece(self, piece: bytes) -> list[int]:
        """Perform rank-ordered byte-pair merges for one pre-tokenized span."""
        if not piece:
            return []
        parts = [bytes((byte,)) for byte in piece]
        while len(parts) > 1:
            candidate_rank = None
            candidate_index = None
            for index in range(len(parts) - 1):
                rank = self._ranks.get(parts[index] + parts[index + 1])
                if rank is not None and (candidate_rank is None or rank < candidate_rank):
                    candidate_rank, candidate_index = rank, index
            if candidate_index is None:
                break
            parts[candidate_index : candidate_index + 2] = [
                parts[candidate_index] + parts[candidate_index + 1]
            ]
        return [self._ranks[part] for part in parts]

    def encode(
        self,
        text: str,
        *,
        allowed_special: Iterable[str] | str = (),
        disallowed_special: Iterable[str] | str = "all",
    ) -> list[int]:
        # Whisper only encodes ordinary text and punctuation. Supporting special
        # tokens here keeps the shim safe for prompts too, without importing
        # tiktoken's Rust extension.
        if allowed_special == "all":
            allowed = self.special_tokens_set
        else:
            allowed = set(allowed_special)
        special_pattern = None
        if allowed:
            special_pattern = re.compile("(" + "|".join(re.escape(token) for token in sorted(allowed, key=len, reverse=True)) + ")")
        pieces = [text] if special_pattern is None else special_pattern.split(text)
        result: list[int] = []
        for piece in pieces:
            if piece in allowed:
                result.append(self._special_tokens[piece])
                continue
            for match in _PATTERN.finditer(piece):
                result.extend(self._encode_piece(match.group().encode("utf-8")))
        return result

    def decode(self, tokens: Iterable[int], *, errors: str = "replace") -> str:
        return b"".join(self.decode_single_token_bytes(token) for token in tokens).decode("utf-8", errors=errors)

    def decode_single_token_bytes(self, token: int) -> bytes:
        if token in self._id_to_bytes:
            return self._id_to_bytes[token]
        if token in self._id_to_special:
            return self._id_to_special[token].encode("utf-8")
        raise KeyError(f"Unknown token ID: {token}")
