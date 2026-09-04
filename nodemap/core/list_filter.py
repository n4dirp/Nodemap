"""Provide pure filtering logic for the minimap type list."""

from __future__ import annotations


def normalize_query(query: str) -> str:
    """Strip surrounding whitespace and lowercase the search query."""
    return query.strip().lower()


def matches(query: str, text: str) -> bool:
    """Return True when *text* contains *query* case-insensitively.

    *query* is expected to be pre-normalized (see ``normalize_query``);
    an empty query never matches.
    """
    if not query:
        return False
    return query in text.lower()


def match_span(query: str, text: str) -> int:
    """Return the start index of the first *query* match in *text*, or -1.

    *query* is expected to be pre-normalized; an empty query never matches.
    The returned index is into the original *text* (lowercasing preserves
    positions for case-insensitive matching).
    """
    if not query:
        return -1
    return text.lower().find(query)


def _child_text(name: str, search_texts: dict[str, str] | None) -> str:
    """Return the searchable text for a child node name.

    Prefers the caller-provided *search_texts* entry (node name plus its
    custom label) and falls back to the bare name when absent.
    """
    if search_texts is not None:
        text = search_texts.get(name)
        if text is not None:
            return text
    return name


def filter_matching_nodes(
    type_stats: dict[str, int],
    children: dict[str, list[str]],
    query: str,
    search_texts: dict[str, str] | None = None,
) -> frozenset[str] | None:
    """Return the node names to keep under *query*, or None when not filtering.

    A node is kept when its type label matches the query or when its own name
    (or, with *search_texts*, its combined name + custom label text) matches.
    This mirrors the row rules of :func:`filter_type_list`: every child of a
    label-matching type stays, plus any child that matches by name. An empty
    query returns None (draw everything); otherwise the result is a frozenset
    of node names to keep visible.
    """
    norm = normalize_query(query)
    if not norm:
        return None
    keep: set[str] = set()
    for label, _count in type_stats.items():
        names = children.get(label, ())
        if matches(norm, label):
            keep.update(names)
            continue
        for name in names:
            if matches(norm, _child_text(name, search_texts)):
                keep.add(name)
    return frozenset(keep)


def filter_type_list(
    type_stats: dict[str, int],
    children: dict[str, list[str]],
    expanded: set[str],
    query: str,
    search_texts: dict[str, str] | None = None,
) -> tuple[list[tuple[str, int]], set[str], dict[str, list[str]]]:
    """Filter node-type rows by *query* and resolve the effective expansion.

    A type stays visible when its label matches, when the single node of a
    one-node type matches (such types list no child rows), or when at least
    one child node matches. A child "matches" on its node name or, when
    *search_texts* is provided, on the combined name + custom label text.
    The returned display count is the full type count for label matches and
    the number of matching children otherwise.

    Types whose only matches are children are added to the returned
    expansion set so the matching nodes become visible; the caller's
    *expanded* set is never mutated. With an empty *query* the input is
    returned unchanged (counts and expansion preserved).

    The returned child map maps every type label to the child rows that stay
    listed under it: only the nodes matching *query* while one is active, or
    all children passed in when the list is unfiltered.
    """
    norm = normalize_query(query)
    if not norm:
        return [(label, count) for label, count in type_stats.items()], set(expanded), dict(children)

    def _child_match(name: str) -> bool:
        return matches(norm, _child_text(name, search_texts))

    def _label_rank(label: str) -> int:
        normalized_label = label.lower()
        if normalized_label.startswith(norm):
            return 0
        if norm in normalized_label:
            return 1
        return 2

    visible: list[tuple[int, str, int]] = []
    effective = set(expanded)
    filtered_children: dict[str, list[str]] = {}
    for label, count in type_stats.items():
        names = children.get(label, [])
        matched = [name for name in names if _child_match(name)]
        filtered_children[label] = matched
        if matches(norm, label):
            visible.append((_label_rank(label), label, count))
            continue
        if not names:
            continue
        if len(names) == 1:
            if _child_match(names[0]):
                visible.append((2, label, count))
        else:
            if matched:
                visible.append((2, label, len(matched)))
                effective.add(label)
    # Relevance order: label starts-with, then label contains, then child
    # matches; stable within each rank. The caller keeps this order while a
    # query is active instead of applying its name/count sort.
    visible.sort(key=lambda item: (item[0], item[1].lower()))
    return [(label, count) for _rank, label, count in visible], effective, filtered_children
