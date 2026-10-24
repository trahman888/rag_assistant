"""
Reversible PII masking: the LLM never sees private details, the user still does.

HOW IT WORKS
    1. Before a prompt is sent, mask() swaps each private detail for a placeholder:
           "Email sara.ahmed@gmail.com or call +1 604-555-0199"
        -> "Email <EMAIL_1> or call <PHONE_1>"
       The same value always gets the same placeholder within one request, so
       the LLM can still tell "two different emails" from "the same email twice".
    2. The LLM answers using the placeholders ("Sara's email is <EMAIL_1>").
    3. unmask() / unmask_stream() swap the real values back in before the user
       sees the answer. The real values only ever exist inside this app.

    One Masker is created per LLM request and thrown away afterwards, so the
    mapping (placeholder -> real value) is never stored or sent anywhere.

WHAT IS DETECTED (pattern rules: fast, free, no extra packages)
    EMAIL        sara.ahmed@gmail.com
    CARD         4111 1111 1111 1111   (must pass the Luhn checksum, so random long numbers aren't hidden)
    SSN / SIN    123-45-6789 / 123 456 789
    IP           192.168.1.20
    PHONE        +1 604-555-0199, (604) 555 0199, 604.555.0199, +44 20 7946 0958
                 (needs 9–15 digits, so years, dates and amounts are left alone)

    Names and street addresses need a language-aware detector. Microsoft
    Presidio (open-source) can be added later inside find_pii().

LIMITS
    Pattern rules miss unusual formats. Masking lowers the risk of private data
    reaching the LLM provider; it isn't a guarantee.
"""
import re
from collections.abc import Iterable, Iterator

ENABLED = True          # set to False to turn masking off everywhere

_EMAIL = re.compile(r"(?<![\w.+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}(?![\w-])")
_CARD = re.compile(r"(?<![\d-])(?:\d[ -]?){12,18}\d(?![\d-])")
_SSN = re.compile(r"(?<![\d-])\d{3}-\d{2}-\d{4}(?![\d-])")
_SIN = re.compile(r"(?<![\d-])\d{3}[ -]\d{3}[ -]\d{3}(?![\d-])")
_IP = re.compile(r"(?<![\d.])(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)(?![\d.])")
_PHONE = re.compile(r"(?<![\w+])(?:\+\d{1,3}[\s.-]?)?(?:\(\d{1,4}\)[\s.-]?)?\d{2,4}(?:[\s.-]?\d{2,4}){1,4}(?![\w])")
_PLACEHOLDER = re.compile(r"<([A-Z]+)_(\d+)>")


def _luhn_ok(number: str) -> bool:
    digits = [int(d) for d in re.sub(r"\D", "", number)][::-1]
    total = sum(d if i % 2 == 0 else (d * 2 - 9 if d * 2 > 9 else d * 2) for i, d in enumerate(digits))
    return len(digits) >= 13 and total % 10 == 0


def find_pii(text: str) -> list[tuple[int, int, str]]:
    """Return (start, end, kind) for every private detail, without overlaps.
    Earlier rules win, so an email's digits are never also treated as a phone."""
    found: list[tuple[int, int, str]] = []

    def add(pattern, kind, check=None):
        for m in pattern.finditer(text):
            s, e = m.span()
            if check and not check(m.group()):
                continue
            if any(s < fe and e > fs for fs, fe, _ in found):
                continue
            found.append((s, e, kind))

    add(_EMAIL, "EMAIL")
    add(_CARD, "CARD", _luhn_ok)
    add(_SSN, "SSN")
    add(_SIN, "SIN")
    add(_IP, "IP")
    add(_PHONE, "PHONE", lambda v: 9 <= len(re.sub(r"\D", "", v)) <= 15
        and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", v.strip()))           # not a date like 2026-10-23
    return sorted(found)


class Masker:
    """Holds the placeholder <-> real value mapping for ONE request."""

    def __init__(self, enabled: bool = ENABLED):
        self.enabled = enabled
        self.to_placeholder: dict[str, str] = {}
        self.to_value: dict[str, str] = {}
        self.counts: dict[str, int] = {}

    @property
    def hidden(self) -> int:
        """How many distinct private details were hidden from the LLM."""
        return len(self.to_value)

    def _placeholder(self, value: str, kind: str) -> str:
        key = value.lower() if kind == "EMAIL" else re.sub(r"\D", "", value) if kind != "IP" else value
        if key not in self.to_placeholder:
            self.counts[kind] = self.counts.get(kind, 0) + 1
            ph = f"<{kind}_{self.counts[kind]}>"
            self.to_placeholder[key] = ph
            self.to_value[ph] = value
        return self.to_placeholder[key]

    def mask(self, text: str) -> str:
        if not self.enabled or not text:
            return text
        out, last = [], 0
        for s, e, kind in find_pii(text):
            out.append(text[last:s])
            out.append(self._placeholder(text[s:e], kind))
            last = e
        out.append(text[last:])
        return "".join(out)

    def mask_messages(self, messages: list[dict]) -> list[dict]:
        return [{**m, "content": self.mask(m["content"])} for m in messages]

    def unmask(self, text: str) -> str:
        if not self.to_value:
            return text
        return _PLACEHOLDER.sub(lambda m: self.to_value.get(m.group(0), m.group(0)), text)

    def unmask_stream(self, pieces: Iterable[str]) -> Iterator[str]:
        """Unmask a streamed reply. A placeholder can arrive split across pieces
        ("<EMA" + "IL_1>"), so a possibly-unfinished tail is held back until complete."""
        buffer = ""
        for piece in pieces:
            buffer += piece or ""
            cut = buffer.rfind("<")
            if cut != -1 and re.fullmatch(r"<[A-Z]*_?\d*", buffer[cut:]):
                ready, buffer = buffer[:cut], buffer[cut:]
            else:
                ready, buffer = buffer, ""
            if ready:
                yield self.unmask(ready)
        if buffer:
            yield self.unmask(buffer)


PROMPT_NOTE = ("Some private details in this conversation were replaced with placeholders such as "
               "<EMAIL_1> or <PHONE_2>. Treat each placeholder as the real value. If the answer needs "
               "one, copy the placeholder exactly as written; never guess or invent the real value.")
