"""Copy-snapping for tool-call arguments.

Measured on Songgot 12L (2026-09-12): of 500 benchmark items, 203 pick the right tool and fill the right keys but
get a value wrong, and most of those are near-miss copies of a span of the user's Korean query (a kept particle,
a corrupted syllable). A tiny model garbles copies; the query is right there, so snap each string value to the
closest span of the query when the two are close enough.

Candidates are every word-boundary span of the query up to a few words, plus each span with a trailing Korean
particle removed. The best candidate by character F1 wins if it beats the model's own string and clears a floor.
"""
import re

PARTICLES = ("으로부터", "에서부터", "이라고", "라고", "에서", "에게", "한테", "까지", "부터", "으로", "로", "의", "에", "은", "는", "이", "가", "을", "를", "와", "과", "랑", "이랑", "도", "만", "까진", "께")
HANGUL = re.compile(r"[가-힣]")


def _f1(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    prev = [0] * (len(b) + 1)
    for ch in a:
        cur = [0]
        for j, bj in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if ch == bj else max(prev[j], cur[j - 1]))
        prev = cur
    l = prev[-1]
    return 0.0 if not l else 2 * (l / len(a)) * (l / len(b)) / ((l / len(a)) + (l / len(b)))


def strip_particle(s: str) -> str:
    for p in PARTICLES:
        if s.endswith(p) and len(s) > len(p) + 1 and HANGUL.search(s[: -len(p)]):
            return s[: -len(p)]
    return s


def spans(query: str, max_words: int = 6):
    words = query.split()
    out = set()
    for i in range(len(words)):
        for j in range(i + 1, min(i + max_words, len(words)) + 1):
            s = " ".join(words[i:j]).strip(" .,!?~\"'()[]{}")
            if s:
                out.add(s); out.add(strip_particle(s))
    return [s for s in out if s]


def snap_value(v, query: str, floor: float = 0.5, margin: float = 0.02):
    """Return the query span that best matches v, if it is a better match than v itself."""
    if not isinstance(v, str) or len(v) < 2 or not HANGUL.search(query):
        return v
    if v in query:
        st = strip_particle(v)
        return st if st != v and st in query else v
    best, score = v, _f1(v, v) * 0.0 + floor
    for s in spans(query):
        f = _f1(v, s)
        if f > score + margin:
            best, score = s, f
    return best


def snap_call(call: dict, query: str) -> dict:
    if not isinstance(call, dict) or not isinstance(call.get("arguments"), dict):
        return call
    return dict(call, arguments={k: snap_value(v, query) for k, v in call["arguments"].items()})
