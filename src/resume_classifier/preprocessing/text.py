"""Text normalisation, in two deliberately separate levels.

``ingestion_normalize`` is conservative and its output is what gets *stored* in
the corpus: Unicode repair, control-character removal, whitespace collapsing.
Case and punctuation survive, so downstream decisions stay open.

``feature_normalize`` is aggressive and its output is *never* stored. It is the
exact transformation applied both when training a model and when serving a
prediction. Because both paths call this one function, the train/serve skew that
silently degrades text classifiers cannot occur.

PII redaction lives at the feature level for two reasons. Privacy is the obvious
one. The subtler one is leakage: a candidate's name or e-mail address is a
fingerprint, and an unredacted fingerprint lets a model recognise a person who
appears in more than one split rather than learning the category.
"""

from __future__ import annotations

import re
import unicodedata

# -- control and formatting characters -----------------------------------
# Zero-width marks and line/paragraph separators. Listed by codepoint because
# the characters themselves are invisible in source and easy to corrupt.
_ZERO_WIDTH_CODEPOINTS = (
    0x200B,  # zero width space
    0x200C,  # zero width non-joiner
    0x200D,  # zero width joiner
    0x200E,  # left-to-right mark
    0x200F,  # right-to-left mark
    0x2028,  # line separator
    0x2029,  # paragraph separator
    0xFEFF,  # byte order mark
)
_CONTROL_CODEPOINTS = (*range(0x00, 0x09), 0x0B, 0x0C, *range(0x0E, 0x20), 0x7F)

_ZERO_WIDTH = re.compile("[" + "".join(chr(cp) for cp in _ZERO_WIDTH_CODEPOINTS) + "]")
_CONTROL = re.compile("[" + "".join(chr(cp) for cp in _CONTROL_CODEPOINTS) + "]")
_WHITESPACE_RUN = re.compile(r"[^\S\n]+")
_BLANK_LINES = re.compile(r"\n{3,}")

# -- personally identifying patterns -------------------------------------
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_URL = re.compile(r"\b(?:https?://|www\.)\S+", re.IGNORECASE)
_PHONE = re.compile(
    r"(?<!\w)(?:\+?\d{1,3}[\s.-]?)?(?:\(\d{2,4}\)[\s.-]?)?"
    r"\d{3,4}[\s.-]?\d{3,4}(?:[\s.-]?\d{2,4})?(?!\w)"
)
_LONG_DIGITS = re.compile(r"\b\d[\d,.]{3,}\b")
_STANDALONE_DIGITS = re.compile(r"\b\d+\b")

EMAIL_TOKEN = " emailaddr "
URL_TOKEN = " urladdr "
PHONE_TOKEN = " phonenum "
NUMBER_TOKEN = " numtoken "

# Keep +, #, . and - so that "c++", "c#", "node.js" and "back-end" survive
# long enough for the skill matcher to see them.
_NON_WORD = re.compile(r"[^a-z0-9+#.\s_-]+")

# Both rules are anchored to token boundaries. A one-character lookbehind is not
# enough: inside a run like "c++" it succeeds at the second "+" (whose neighbour
# is punctuation, not a letter) and silently eats it.
#
# Leading punctuation is always noise ("---", "..."). Trailing punctuation is
# noise for . _ and - ("years.", "skills-"), but NOT for + and #, which carry
# meaning in language names ("c++", "c#", "f#").
_LEAD_PUNCT = re.compile(r"(?:^|(?<=\s))[.+#_-]+")
_TRAIL_PUNCT = re.compile(r"[._-]+(?=\s|$)")
_SPACE_RUN = re.compile(r"\s+")


def ingestion_normalize(text: str) -> str:
    """Repair and tidy raw text without discarding information.

    Applies Unicode NFKC, strips control and zero-width characters, normalises
    line endings, collapses runs of horizontal whitespace, and limits blank-line
    runs. Case and punctuation are preserved.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _ZERO_WIDTH.sub("", text)
    text = _CONTROL.sub(" ", text)
    text = _WHITESPACE_RUN.sub(" ", text)
    text = _BLANK_LINES.sub("\n\n", text)
    return "\n".join(line.strip() for line in text.split("\n")).strip()


def redact_pii(text: str) -> str:
    """Replace direct identifiers with stable placeholder tokens.

    Order matters: e-mail addresses contain the ``@`` and dots that the URL and
    telephone patterns would otherwise partially consume.
    """
    text = _EMAIL.sub(EMAIL_TOKEN, text)
    text = _URL.sub(URL_TOKEN, text)
    text = _PHONE.sub(PHONE_TOKEN, text)
    return _LONG_DIGITS.sub(NUMBER_TOKEN, text)


def feature_normalize(text: str) -> str:
    """Produce the exact text the model sees, at training and at inference.

    This is the single shared transformation. Changing it changes both sides at
    once, by construction.
    """
    if not text:
        return ""
    text = ingestion_normalize(text)
    text = redact_pii(text)
    text = text.casefold()
    text = _STANDALONE_DIGITS.sub(NUMBER_TOKEN, text)
    text = _NON_WORD.sub(" ", text)
    text = _LEAD_PUNCT.sub(" ", text)
    text = _TRAIL_PUNCT.sub(" ", text)
    return _SPACE_RUN.sub(" ", text).strip()


def word_count(text: str) -> int:
    """Whitespace-delimited word count."""
    return len(text.split())
