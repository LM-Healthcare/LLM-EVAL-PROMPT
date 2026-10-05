"""Extraction of the machine-readable block at the end of each response.

The parser is deliberately strict about *what* it accepts (an explicit
`ANSWER:` line with one letter A-E; no guessing from free text, which would
bias the comparison between verbose and terse conditions) and tolerant about
*formatting* (markdown emphasis, parentheses, trailing text). When a key
appears more than once, the last occurrence wins.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

_DECOR = r"[\s*_`#>]*"
ANSWER_RE = re.compile(
    rf"(?im)^{_DECOR}(?:final\s+)?(?:ANSWER|RISPOSTA){_DECOR}[:：]{_DECOR}\(?([A-E])(?![A-Za-z])"
)
CONF_RE = re.compile(
    rf"(?im)^{_DECOR}(?:CONFIDENCE|CONFIDENZA){_DECOR}[:：]{_DECOR}(\d{{1,3}}(?:[.,]\d+)?)\s*%?"
)
URG_RE = re.compile(rf"(?im)^{_DECOR}(?:URGENCY|URGENZA){_DECOR}[:：]{_DECOR}([^\n]+)")
NEXT_RE = re.compile(rf"(?im)^{_DECOR}(?:NEXT_STEP|NEXT STEP|PASSO SUCCESSIVO){_DECOR}[:：]{_DECOR}([^\n]+)")
REF_HEADER_RE = re.compile(rf"(?im)^{_DECOR}(?:REFERENCES|RIFERIMENTI){_DECOR}[:：]?\s*$")

URGENCY_LEVELS = ["non-urgent", "urgent", "emergency"]  # ordinal, low -> high


def normalise_urgency(raw: str) -> str | None:
    s = raw.strip().lower().strip("*_`.<> ")
    s = s.replace("–", "-").replace("—", "-")
    if re.match(r"^(non[\s-]?urgent|not[\s-]urgent|non[\s-]?urgente)", s):
        return "non-urgent"
    if re.match(r"^(emergen|emergenza)", s):
        return "emergency"
    if re.match(r"^(urgent|urgente)", s):
        return "urgent"
    return None


@dataclass
class Parsed:
    answer: str | None = None
    confidence: float | None = None
    status: str = "ok"  # ok | no_answer | no_confidence | invalid_confidence
    truncated: bool = False
    urgency: str | None = None
    urgency_raw: str | None = None
    next_step: str | None = None
    references: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["parse_status"] = d.pop("status")
        d["references"] = " || ".join(self.references)
        return d


def parse_response(text: str, finish_reason: str = "") -> Parsed:
    text = text or ""
    p = Parsed(truncated=finish_reason in ("max_tokens", "length"))
    answers = ANSWER_RE.findall(text)
    if answers:
        p.answer = answers[-1].upper()
    confs = CONF_RE.findall(text)
    if confs:
        val = float(confs[-1].replace(",", "."))
        if 0 <= val <= 100:
            p.confidence = val
    if p.answer is None:
        p.status = "no_answer"
    elif p.confidence is None:
        p.status = "invalid_confidence" if confs else "no_confidence"

    urg = URG_RE.findall(text)
    if urg:
        p.urgency_raw = urg[-1].strip()
        p.urgency = normalise_urgency(p.urgency_raw)
    nxt = NEXT_RE.findall(text)
    if nxt:
        p.next_step = nxt[-1].strip()

    m = None
    for m in REF_HEADER_RE.finditer(text):
        pass
    if m is not None:
        for line in text[m.end():].splitlines():
            line = line.strip()
            if not line:
                if p.references:
                    break
                continue
            if ANSWER_RE.match(line) or CONF_RE.match(line) or URG_RE.match(line):
                break
            if line.startswith(("-", "*", "•")) or re.match(r"^\d+[.)]", line):
                ref = re.sub(r"^(?:[-*•]|\d+[.)])\s*", "", line).strip()
                if ref:
                    p.references.append(ref)
            else:
                break
    return p
