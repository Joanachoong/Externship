"""Token chunking shared by semantic theme models.

This helper uses the caller's tokenizer because the same text can have different
token lengths for the NLI classifier and sentence encoder. It does not clean text,
split on sentence boundaries, or add overlap between consecutive chunks.
"""


def _chunks(text, tokenizer, token_budget):
    """Cover the whole text rather than silently truncating long reviews.

    ``token_budget`` must be a positive count of content tokens after reserving
    space for the model's special tokens. Return decoded strings so callers can
    pass them to high-level inference APIs. Empty input returns ``[text]`` to
    preserve a caller's one-input/one-owner bookkeeping.
    """
    # Count subword tokens, not words/characters; omit special tokens because
    # each inference API will add its own boundaries when encoding the chunk.
    tokens = tokenizer.encode(text, add_special_tokens=False, verbose=False)
    pieces, start = [], 0
    while start < len(tokens):
        end = min(start + token_budget, len(tokens))
        piece = tokenizer.decode(tokens[start:end], skip_special_tokens=True)
        # Decoding a partial subword can slightly change re-tokenized length.
        # Shrink only this chunk until the string also fits when encoded again.
        while len(tokenizer.encode(piece, add_special_tokens=False, verbose=False)) > token_budget:
            end -= 1
            if end <= start:
                raise ValueError("A token cannot be represented within the chunk budget")
            piece = tokenizer.decode(tokens[start:end], skip_special_tokens=True)
        pieces.append(piece)
        # Continue exactly where this chunk stopped: no skipped source tokens
        # and no overlap that would repeatedly count the same passage.
        start = end
    return pieces or [text]
