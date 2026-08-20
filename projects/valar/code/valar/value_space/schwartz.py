from __future__ import annotations

SCHWARTZ_10 = [
    "Self-Direction",
    "Stimulation",
    "Hedonism",
    "Achievement",
    "Power",
    "Security",
    "Conformity",
    "Tradition",
    "Benevolence",
    "Universalism",
]

CIRCUMPLEX_ORDER = [
    "Self-Direction",
    "Stimulation",
    "Hedonism",
    "Achievement",
    "Power",
    "Security",
    "Conformity",
    "Tradition",
    "Benevolence",
    "Universalism",
]

OPENNESS_TO_CHANGE = {"Self-Direction", "Stimulation"}
SELF_ENHANCEMENT = {"Hedonism", "Achievement", "Power"}
CONSERVATION = {"Security", "Conformity", "Tradition"}
SELF_TRANSCENDENCE = {"Benevolence", "Universalism"}


def circumplex_distance(a: str, b: str) -> int:
    n = len(CIRCUMPLEX_ORDER)
    i, j = CIRCUMPLEX_ORDER.index(a), CIRCUMPLEX_ORDER.index(b)
    return min(abs(i - j), n - abs(i - j))
