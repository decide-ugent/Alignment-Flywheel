"""MedicalBatchDecider — accepts up to max_patches medical corrections."""

from typing import Any, Dict, Optional, Set

from flywheel.protocols.ooda.decide_step import DecideStep


class MedicalBatchDecider(DecideStep):
    """Decide: cap medical refinement batch to max_patches items.

    Uses tiered selection to produce gradual convergence:
      1. Different category from last iteration + novel keyword
      2. Any category with a novel keyword
      3. Different category from last (for coverage updates)
      4. Any remaining candidate

    This alternates between flaw categories across iterations,
    ensuring hard-block patches and audit coverage updates are
    spread out rather than applied all at once.
    """

    def __init__(self):
        self._seen_keywords: Set[str] = set()
        self._last_category: Optional[str] = None

    def _has_novel_kw(self, item):
        kws = {c.payload.get("keyword", "")
               for c in item.get("corrections", [])
               if hasattr(c, "payload") and c.payload.get("keyword")}
        return not kws or not kws.issubset(self._seen_keywords)

    def decide(self, oriented: Dict[str, Any]) -> Dict[str, Any]:
        planned = oriented["planned"]
        max_patches = oriented["max_patches"]
        cat_filter = oriented.get("category_filter")
        if cat_filter:
            planned = [p for p in planned if p["category"] in cat_filter]

        # Tier 1: different category + novel keyword
        tier1 = [p for p in planned
                 if p["category"] != self._last_category
                 and self._has_novel_kw(p)]
        # Tier 2: different category, any keyword (for coverage rotation)
        tier2 = [p for p in planned
                 if p["category"] != self._last_category]
        # Tier 3: any category with novel keyword
        tier3 = [p for p in planned if self._has_novel_kw(p)]

        candidates = tier1 or tier2 or tier3 or planned
        accepted = candidates[:max_patches]

        for p in accepted:
            self._last_category = p["category"]
            for c in p.get("corrections", []):
                if hasattr(c, "payload"):
                    kw = c.payload.get("keyword")
                    if kw:
                        self._seen_keywords.add(kw)

        return {
            "accepted": accepted,
            "rejected": max(0, len(planned) - max_patches),
            "shrinks": 0,
            "oracle_version": oriented["oracle_version"],
        }
