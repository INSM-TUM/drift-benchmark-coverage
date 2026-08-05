"""Weighted dependency discovery — compute W = N_D / T for every dependency type.

For each activity pair (A, B) and each dependency type D:
  - N_D = number of traces that do NOT violate the rule for D
  - T   = total traces in event log
  - W   = N_D / T  (1.0 = fully consistent, 0.0 = always violated)

See .planning/quick/008-weighted-discovery-violation-rules for the full
violation rule table for both existential and temporal types.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.armature_light.models import Trace


@dataclass(frozen=True)
class ExistentialWeights:
    """Per-pair existential dependency weights.
    """
    count_both: int
    count_only_a: int
    count_only_b: int
    count_neither: int
    total: int


@dataclass(frozen=True)
class TemporalWeights:
    """Per-pair temporal dependency weights.
    """

    direct: float
    direct_backward: float
    true_eventual: float
    true_eventual_backward: float
    total: int


@dataclass(frozen=True)
class PairWeights:
    """All dependency weights for a single activity pair (source, target)."""

    source: str
    target: str
    existential: ExistentialWeights
    temporal: TemporalWeights


def compute_weights(traces: list[Trace]) -> dict[tuple[str, str], PairWeights]:
    """Compute dependency weights for all activity pairs.

    Args:
        traces: List of process traces parsed from an XES event log.

    Returns:
        Dict mapping (source, target) -> PairWeights for every pair.
    """
    from collections import defaultdict

    T = len(traces)
    if T == 0:
        return {}

    # Global counts for temporal probabilities
    direct_counts = defaultdict(int)
    direct_total = defaultdict(int)
    eventual_counts = defaultdict(int)
    eventual_total = defaultdict(int)

    # Existential counts
    act_trace_counts = defaultdict(int)
    pair_trace_counts = defaultdict(int)

    all_activities: set[str] = set()

    for trace in traces:
        acts = [e.activity for e in trace.events]
        n_acts = len(acts)
        
        act_set = set(acts)
        all_activities.update(act_set)
        
        for act in act_set:
            act_trace_counts[act] += 1
            for act2 in act_set:
                pair_trace_counts[(act, act2)] += 1
        
        # Temporal: Direct Follows
        for i in range(n_acts - 1):
            act_i = acts[i]
            act_j = acts[i + 1]
            direct_counts[(act_i, act_j)] += 1
            direct_total[act_i] += 1
            
        # Temporal: Eventual Follows
        last_pos = {}
        for i, act in enumerate(acts):
            last_pos[act] = i
            
        for i, act in enumerate(acts):
            eventual_total[act] += 1
            for b, pos in last_pos.items():
                if pos > i:
                    eventual_counts[(act, b)] += 1

    activities = sorted(all_activities)
    result: dict[tuple[str, str], PairWeights] = {}

    for act_a in activities:
        for act_b in activities:
            # ── existential counters ──────────────────────────────────────
            count_both = pair_trace_counts[(act_a, act_b)]
            count_only_a = act_trace_counts[act_a] - count_both
            count_only_b = act_trace_counts[act_b] - count_both
            count_neither = T - count_both - count_only_a - count_only_b

            # ── temporal probabilities ────────────────────────────────────
            if direct_total[act_a] > 0:
                direct_prob = direct_counts[(act_a, act_b)] / direct_total[act_a]
            else:
                direct_prob = 0.0
                
            if direct_total[act_b] > 0:
                direct_bk_prob = direct_counts[(act_b, act_a)] / direct_total[act_b]
            else:
                direct_bk_prob = 0.0
                
            if eventual_total[act_a] > 0:
                eventual_prob = eventual_counts[(act_a, act_b)] / eventual_total[act_a]
            else:
                eventual_prob = 0.0
                
            if eventual_total[act_b] > 0:
                eventual_bk_prob = eventual_counts[(act_b, act_a)] / eventual_total[act_b]
            else:
                eventual_bk_prob = 0.0

            # ── build weight objects ──────────────────────────────────────
            ew = ExistentialWeights(
                count_both=count_both / T,
                count_only_a=count_only_a / T,
                count_only_b=count_only_b / T,
                count_neither=count_neither / T,
                total=T,
            )

            tw = TemporalWeights(
                direct=direct_prob,
                direct_backward=direct_bk_prob,
                true_eventual=eventual_prob,
                true_eventual_backward=eventual_bk_prob,
                total=T,
            )

            result[(act_a, act_b)] = PairWeights(
                source=act_a,
                target=act_b,
                existential=ew,
                temporal=tw,
            )

    return result