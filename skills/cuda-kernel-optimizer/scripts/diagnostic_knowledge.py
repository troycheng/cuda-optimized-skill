#!/usr/bin/env python3
"""Route a bounded set of offline diagnostic cards into active diagnosis."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from pathlib import Path


CARDS_PATH = Path(__file__).resolve().parents[1] / "references" / "diagnostic_cards.json"


def _load_cards() -> list[dict]:
    value = json.loads(CARDS_PATH.read_text(encoding="utf-8"))
    if value.get("schema_version") != "cuda-optimizer/diagnostic-cards-v1":
        raise ValueError("diagnostic card registry schema is unsupported")
    cards = value.get("cards")
    if type(cards) is not list or not cards:
        raise ValueError("diagnostic card registry is empty")
    return cards


def route_cards(
    diagnosis: Mapping[str, object],
    execution_map: Mapping[str, object],
    *,
    limit: int = 3,
) -> dict:
    if type(limit) is not int or isinstance(limit, bool) or not 1 <= limit <= 5:
        raise ValueError("limit must be between 1 and 5")
    primary = diagnosis.get("primary_category")
    categories = []
    if isinstance(primary, str) and primary != "mixed":
        categories.append(primary)
    ranked = diagnosis.get("ranked_categories", [])
    if type(ranked) is list:
        categories.extend(
            item.get("category")
            for item in ranked
            if type(item) is dict and isinstance(item.get("category"), str)
        )
    if not categories:
        categories = ["unknown"]
    elif primary == "mixed":
        categories.insert(0, "mixed")
    categories = list(dict.fromkeys(categories))
    labels = " ".join(
        str(item.get("label", "")).lower()
        for item in execution_map.get("nodes", [])
        if type(item) is dict
    )
    ranked_cards = []
    for card in _load_cards():
        category_rank = min(
            (
                categories.index(category)
                for category in card["categories"]
                if category in categories
            ),
            default=99,
        )
        if category_rank == 99:
            continue
        term_match = any(term in labels for term in card["match_terms"])
        ranked_cards.append(
            ((category_rank, 0 if term_match else 1, card["priority"], card["id"]), card)
        )
    if not ranked_cards:
        fallback = next(
            card for card in _load_cards() if card["id"] == "diagnostic.cross-layer.triage"
        )
        ranked_cards = [((0, 0, 0, fallback["id"]), fallback)]
    ranked_cards.sort(key=lambda item: item[0])
    return {
        "schema_version": "cuda-optimizer/diagnostic-knowledge-context-v1",
        "categories": categories,
        "cards": [copy.deepcopy(card) for _, card in ranked_cards[:limit]],
        "promotion_authority": "none",
    }
