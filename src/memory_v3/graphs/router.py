"""
memory-v3 -- Query Router
Analyzes query intent to determine which graph layer(s) should be searched.
"""

from __future__ import annotations

TEMPORAL_SIGNALS = [
    "when", "after", "before", "timeline", "history", "sequence",
    "happened", "date", "time", "first", "last", "recent",
]
CAUSAL_SIGNALS = [
    "why", "because", "caused", "led to", "result", "consequence",
    "reason", "fault", "due to", "root cause",
]
ENTITY_SIGNALS = [
    "who", "which team", "person", "built", "works on", "responsible",
    "owns", "maintains", "created by",
]
SEMANTIC_SIGNALS = [
    "what is", "how does", "concept", "means", "related to",
    "about", "explain", "define",
]

# Map from signal list to layer name
_SIGNAL_MAP = [
    (TEMPORAL_SIGNALS, "temporal"),
    (CAUSAL_SIGNALS, "causal"),
    (ENTITY_SIGNALS, "entity"),
    (SEMANTIC_SIGNALS, "semantic"),
]

ALL_LAYERS = ["semantic", "temporal", "causal", "entity"]


def route_query(query: str) -> list[str]:
    """
    Analyze a query string to determine which graph layer(s) to search.

    Checks for signal words/phrases in the query. Multi-word signals are checked
    as substrings; single-word signals are checked as whole-word matches to reduce
    false positives.

    Returns a list of matching layer names. Defaults to ["semantic"] if no
    signals match.
    """
    query_lower = query.lower()
    query_words = set(query_lower.split())
    matched_layers = []

    for signals, layer in _SIGNAL_MAP:
        for signal in signals:
            if " " in signal:
                # Multi-word signal: check as substring
                if signal in query_lower:
                    if layer not in matched_layers:
                        matched_layers.append(layer)
                    break
            else:
                # Single-word signal: check as whole word
                if signal in query_words:
                    if layer not in matched_layers:
                        matched_layers.append(layer)
                    break

    # Default to semantic if nothing matched
    if not matched_layers:
        matched_layers = ["semantic"]

    return matched_layers
