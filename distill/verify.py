"""Rule checks for closed-ended prompts: extract the final answer and compare it with the dataset's gold answers."""
import math
import re
from fractions import Fraction

FINAL = re.compile(r"final answer\s*[:：]\s*(.+)", re.I)
NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?(?:/\d+)?")
LATEX_WRAPPER = re.compile(r"\\(?:boxed|text|textbf|mathrm)\{([^{}]*)\}")
STEP = re.compile(r"^\s*(?:step\s*)?\d+\s*[.):]", re.I)


def extract_boxed(text):
    """Content of the last \\boxed{...}, with nested braces; None if there is none."""
    start = text.rfind("\\boxed{")
    if start < 0:
        return None
    depth, begin = 1, start + len("\\boxed{")
    for i in range(begin, len(text)):
        depth += {"{": 1, "}": -1}.get(text[i], 0)
        if depth == 0:
            return text[begin:i]
    return None


def extract_final(answer):
    """The text after the last 'Final answer:' marker, else the last \\boxed{} content; None if neither is present."""
    matches = FINAL.findall(answer)
    return matches[-1].strip() if matches else extract_boxed(answer)


def to_number(s):
    s = LATEX_WRAPPER.sub(r"\1", s).strip().strip("*$ ").rstrip(".").replace(",", "")
    try:
        return float(Fraction(s))
    except (ValueError, ZeroDivisionError):
        return None


def normalize(s):
    """Lowercase words without punctuation, markdown or articles (articles are kept if nothing else is left)."""
    s = LATEX_WRAPPER.sub(r"\1", s).lower()
    words = "".join(ch if ch.isalnum() else " " for ch in s).split()
    return " ".join([w for w in words if w not in ("a", "an", "the")] or words)


def is_correct(answer, gold):
    """Whether the response's final answer matches any gold answer: numerically when the gold is a number (comparing
    the last number in the final answer), otherwise after normalization. No final answer counts as wrong."""
    pred = extract_final(answer or "")
    if pred is None:
        return False
    for g in gold:
        target = to_number(g)
        if target is not None:
            nums = NUMBER.findall(pred)
            value = to_number(nums[-1]) if nums else None
            if value is not None and math.isclose(value, target, rel_tol=1e-9, abs_tol=1e-9):
                return True
        elif normalize(pred) == normalize(g):
            return True
    return False


def split_steps(rationale):
    """Steps of a rationale: numbered items (a line without a number continues the item before it), or its non-empty
    lines when the steps are not numbered."""
    lines = [line.strip() for line in (rationale or "").splitlines() if line.strip()]
    if not any(STEP.match(line) for line in lines):
        return lines
    steps = []
    for line in lines:
        if STEP.match(line) or not steps:
            steps.append(line)
        else:
            steps[-1] += " " + line
    return steps


def format_steps(rationale):
    """The stored rationale format: one step per paragraph, steps separated by a blank line."""
    return "\n\n".join(split_steps(rationale))


def count_steps(rationale):
    return len(split_steps(rationale))
