import re


# ----------------------------------------------------------------------------- vote (ReCon candidates)
def _last_boxed(s):
    """Last \\boxed{...} / \\fbox{...} with balanced braces (rest of the string if unclosed; s if none)."""
    if "\\boxed " in s:
        return "\\boxed " + s.split("\\boxed ")[-1].split("$")[0]
    idx = s.rfind("\\boxed")
    if idx < 0:
        idx = s.rfind("\\fbox")
        if idx < 0:
            return s
    depth = 0
    for i in range(idx, len(s)):
        depth += (s[i] == "{") - (s[i] == "}")
        if s[i] == "}" and depth == 0:
            return s[idx:i + 1]
    return s[idx:]


def _unbox(s):
    if s.startswith("\\boxed ") and "\\boxed " in s:
        return s[len("\\boxed "):]
    if s.startswith("\\boxed{") and s.endswith("}"):
        return s[len("\\boxed{"):-1]
    return s


def _clean(t):
    t = str(t).strip().split("<|", 1)[0]
    t = re.sub(r"(\$|\\\)|\\\]|\\end\{.*?\})\s*$", "", t).strip()
    return re.sub(r"[.,;:]+$", "", t).strip()


def _last_number(s):
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s or "")
    return nums[-1] if nums else None


def _boxed_loose(s, strip_braces):
    """boxed{...} without backslash, boxedboxed{...}, 'boxed: 123', ... (the last occurrence)."""
    matches = list(re.finditer(r"(\\?\s*(?:boxed|fbox)+)\b", s, flags=re.IGNORECASE))
    if not matches:
        return None
    tail = s[matches[-1].start():]
    b = tail.find("{")
    if b != -1:
        depth = 0
        for j in range(b, len(tail)):
            depth += (tail[j] == "{") - (tail[j] == "}")
            if tail[j] == "}" and depth == 0:
                return _clean(tail[b + 1:j]) or None
    m = re.search(r"(?:boxed|fbox)+\s*[:\-]?\s*([^\n\r$<]+)", tail, flags=re.IGNORECASE)
    if m:
        t = _clean(m.group(1))
        if strip_braces:
            t = t.lstrip("{").rstrip("}").strip()
        return t or None
    return _last_number(tail)


def parse_answer(text, dream=False):
    """Answer string of one ReCon candidate (None if there is none). Tries the last \\boxed{}, a loose 'boxed'
    match, <answer>...</answer>, then the last number. dream=True is the Dream variant used in the paper: the strict
    step only runs when '\\boxed' / '\\fbox' is followed by '{' or a space, and the loose match drops stray braces."""
    s = (text or "").replace("\x00", "").replace("<|endoftext|>", "")
    ans = None
    if not dream or re.search(r"\\(?:boxed|fbox)(?:\s|\{)", s):
        c = _unbox(_last_boxed(s)).strip()
        ans = _clean(c) if c else None
    if not ans and s:
        ans = _boxed_loose(s, strip_braces=dream)
    if not ans:
        m = re.search(r"<answer>(.*?)</answer>", s, re.DOTALL | re.IGNORECASE)
        ans = (_clean(m.group(1).strip()) or None) if m else None
    if not ans:
        ans = _last_number(s)
    return None if ans == "\\boxed{...}" else ans


# ----------------------------------------------------------------------------- scoring (final outputs)
def _scored_boxed(raw):
    """Content of the last closed \\boxed{...}, else of <answer>...</answer>."""
    parsed = None
    if "\\boxed " in raw:
        parsed = raw.split("\\boxed ")[-1].split("$")[0]
    else:
        idx = raw.rfind("\\boxed")
        idx = idx if idx >= 0 else raw.rfind("\\fbox")
        if idx < 0:
            parsed = raw
        else:
            depth = 0
            for i in range(idx, len(raw)):
                depth += (raw[i] == "{") - (raw[i] == "}")
                if raw[i] == "}" and depth == 0:
                    box = raw[idx:i + 1]
                    parsed = box[len("\\boxed{"):-1] if box.startswith("\\boxed{") else box
                    break
    if not parsed:
        m = re.search(r"<answer>(.*?)</answer>", raw, re.DOTALL)
        if m:
            parsed = m.group(1).strip()
    return parsed


def _numeric_equiv(a, b):
    na = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(a))
    nb = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(b))
    return len(na) == 1 and len(nb) == 1 and abs(float(na[0]) - float(nb[0])) < 1e-6


def is_equiv(a, b):
    if isinstance(a, (float, int)) or isinstance(b, (float, int)):
        return _numeric_equiv(a, b)
    if a is None or b is None:
        return a is None and b is None

    def norm(s):
        s = str(s).lower().strip()
        for pat, rep in [(r"^\((.*?)\)$", r"\1"), (r"\.\s*$", ""), (r"\s+", " "), (r"\s+and\s+", ","),
                         (r"^\$([^\$]*)\$$", r"\1"), (r"^\\\((.*?)\\\)$", r"\1"), (r"^\\\[(.*?)\\\]$", r"\1")]:
            s = re.sub(pat, rep, s)
        return {"no.": "no", "yes.": "yes"}.get(s, s)
    a, b = norm(a), norm(b)
    if _numeric_equiv(a, b):
        return True
    if "," in a and "," in b:
        pa, pb = [p.strip() for p in a.split(",")], [p.strip() for p in b.split(",")]
        if len(pa) == len(pb) and all(is_equiv(x, y) for x, y in zip(pa, pb)):
            return True
    return a == b


def _gsm_number(txt, last=False):
    txt = re.sub(r"(?<=\d),(?=\d{3}\b)", "", txt).replace("$", "")   # 118,000 -> 118000, $20 -> 20
    try:
        return float(txt)
    except ValueError:
        nums = re.findall(r"-?\d+\.?\d*", txt)
        return float(nums[-1] if last else nums[0]) if nums else None


def score_gsm8k(raw, gt):
    """First number of the first non-empty \\boxed{} (else the last number of <answer>) == reference."""
    for boxed in re.findall(r"\\boxed{(.*?)}", raw):
        boxed = boxed.strip()
        if boxed and not re.match(r"^\.+$", boxed):
            parsed = _gsm_number(boxed)
            if parsed is not None:
                return parsed == gt
    m = re.search(r"<answer>(.*?)</answer>", raw, re.DOTALL)
    parsed = _gsm_number(m.group(1).strip(), last=True) if m and m.group(1).strip() else None
    return parsed is not None and parsed == gt


def score_boxed(raw, gt):
    """MATH, ARC-C, TruthfulQA (option letter): last \\boxed{} (else <answer>) equivalent to the reference."""
    parsed = _scored_boxed(raw)
    return parsed is not None and is_equiv(parsed, gt)


SCORERS = {"gsm8k": score_gsm8k, "math": score_boxed, "arcc": score_boxed, "truthfulqa": score_boxed}
