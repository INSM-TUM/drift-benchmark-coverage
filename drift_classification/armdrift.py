"""Discrete ARM per log segment and drift classification (paper Defs. 1-6, Tables 1-3, Prop. 1).

Everything here is computed on activity labels from trace-level counts.

Temporal dependency of an ordered pair (A, B), A != B
-----------------------------------------------------
Relevant traces: the traces in which both A and B occur (n_AB).
  x  (no ordering)   n_AB / n_A <= 1 - tau and n_AB / n_B <= 1 - tau   ("never co-occur")
  orientation        a trace is oriented A->B if the first A precedes the first B
                     A ≺* B if >= tau of the co-occurrence traces are oriented A->B,
                     B ≺* A (= A ≻* B) symmetrically, otherwise  -  (independence)
  directness         among the oriented traces, the share with an A directly followed by B:
                     >= tau -> ≺d ; <= 1 - tau -> ≺t ; otherwise ≺ (eventual = ≺d ∪ ≺t)

Existential dependency of (A, B)
--------------------------------
  A ⇒ B    n_AB / n_A >= tau           (relevant traces: those containing A)
  A ⇐ B    n_AB / n_B >= tau
  ⇔        both implications hold
  otherwise, if "never co-occur" (same test as for x):
      ⇎    if (n_onlyA + n_onlyB) / N >= tau   (exactly one in every trace)
      ⊼    otherwise
  otherwise
      ∨    if n_{A∨B} / N >= tau               (at least one in every trace)
      −    otherwise

With tau = 1 these are exactly the noise-free definitions (and armature's existential rules).

Self pairs (A, A) (diagonal)
----------------------------
Relevant traces: those containing A; A repeats in a trace if it occurs more than once.
  no repetition (<= 1 - tau of the relevant traces)   (×, −)
  otherwise                                           (≺?, ⇐)  (a later occurrence implies an earlier one)
     with the directness among the repeating traces: share with A directly followed by A
     >= tau -> ≺d ; <= 1 - tau -> ≺t ; otherwise ≺
E.g. e,f (sequence) -> e,f,e (loop) changes (e,e) from (×,−) to (≺t,⇐): T18 + E27, i.e. a mixed transition.
Diagonal cells take part in the drift classification like every other pair (they are not checked
against Table 1, since (×,−) is the neutral diagonal and not a relation between two activities).
"""

from __future__ import annotations

import gzip
import io
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from lxml import etree

START, END = "▶start", "■end"

# --- dependency encoding -------------------------------------------------------------
# temporal: (base, dir) base in d,e,t,x,i ; dir +1 = A before B, -1 = B before A, 0 symmetric
# existential: (base, dir) base in imp,eq,xor,or,nand,ind ; dir +1 = A⇒B, -1 = A⇐B
T_SYM = {("d", 1): "≺d", ("d", -1): "≻d", ("e", 1): "≺", ("e", -1): "≻", ("t", 1): "≺t",
         ("t", -1): "≻t", ("x", 0): "×", ("i", 0): "−"}
E_SYM = {("imp", 1): "⇒", ("imp", -1): "⇐", ("eq", 0): "⇔", ("xor", 0): "⇎", ("or", 0): "∨",
         ("nand", 0): "⊼", ("ind", 0): "−"}

T_ORDER = ["d", "e", "t", "x", "i"]
E_ORDER = ["imp", "eq", "xor", "or", "nand", "ind"]


def _build_ids(order, prefix, directional):
    ids, k = {}, 1
    for r in order:
        for c in order:
            if r == c and r not in directional:
                continue
            k_id = f"{prefix}{k}"
            ids[(r, c)] = k_id
            k += 1
    return ids


T_ID = _build_ids(T_ORDER, "T", {"d", "e", "t"})   # 23 temporal transitions (Table 2)
E_ID = _build_ids(E_ORDER, "E", {"imp"})           # 31 existential transitions (Table 3)
assert len(T_ID) == 23 and len(E_ID) == 31
assert T_ID[("x", "i")] == "T19" and T_ID[("i", "x")] == "T23" and E_ID[("ind", "nand")] == "E31"

# Prop. 1: transitions between {⇔,∨,−} and {⇎,⊼} are not admissible as existential transitions
_G1, _G2 = {"eq", "or", "ind"}, {"xor", "nand"}
E_INADMISSIBLE = {E_ID[(a, b)] for a in _G1 for b in _G2} | {E_ID[(a, b)] for a in _G2 for b in _G1}
E_ADMISSIBLE = set(E_ID.values()) - E_INADMISSIBLE
assert len(E_ADMISSIBLE) == 19

# Table 1 validity set V (existential implication consolidated)
VALID = {
    "d": {"imp", "eq", "or", "ind"}, "e": {"imp", "eq", "or", "ind"},
    "t": {"imp", "eq", "or", "ind"}, "i": {"imp", "eq", "or", "ind"},
    "x": {"imp", "xor", "nand"},
}


def cell_str(c):
    if c is None:
        return "⊥"
    t, e = c
    return f"({T_SYM[t]},{E_SYM[e]})"


def is_valid(c):
    (tb, _), (eb, _) = c
    return eb in VALID[tb]


# --- log reading -----------------------------------------------------------------------
def _local(tag):
    return tag.rsplit("}", 1)[-1]


@dataclass
class LogData:
    traces: list            # list[tuple[str,...]]
    lifecycle_values: Counter = field(default_factory=Counter)
    lifecycle_filtered: bool = False
    first_ts: list = field(default_factory=list)
    versions: list = field(default_factory=list)   # model:version trace attribute if present
    n_empty: int = 0


def read_xes(source) -> LogData:
    """source: path (.xes / .xes.gz) or bytes."""
    if isinstance(source, (bytes, bytearray)):
        fh = io.BytesIO(source)
    elif str(source).endswith(".gz"):
        fh = gzip.open(source, "rb")
    else:
        fh = open(source, "rb")
    raw, lcs, tss, vers = [], Counter(), [], []
    ctx = etree.iterparse(fh, events=("end",), huge_tree=True)
    for _, el in ctx:
        if _local(el.tag) != "trace":
            continue
        evs, ver = [], None
        for ch in el:
            ln = _local(ch.tag)
            if ln == "event":
                name = lc = ts = None
                for a in ch:
                    k = a.get("key")
                    if k == "concept:name":
                        name = a.get("value")
                    elif k == "lifecycle:transition":
                        lc = a.get("value")
                    elif k == "time:timestamp":
                        ts = a.get("value")
                evs.append((name, lc, ts))
                if lc is not None:
                    lcs[lc.lower()] += 1
            elif ch.get("key") == "model:version":
                ver = ch.get("value")
        raw.append(evs)
        vers.append(ver)
        el.clear()
        while el.getprevious() is not None:
            del el.getparent()[0]
    fh.close()
    filt = any(k != "complete" for k in lcs)
    traces, n_empty = [], 0
    for evs in raw:
        if filt:
            evs = [e for e in evs if e[1] is None or e[1].lower() == "complete"]
        tss.append(evs[0][2] if evs else None)
        if not evs:
            n_empty += 1
        traces.append(tuple(e[0] for e in evs))
    return LogData(traces, lcs, filt, tss, vers, n_empty)


def read_xes_from_zip(zpath, member) -> LogData:
    with zipfile.ZipFile(zpath) as z:
        return read_xes(z.read(member))


# --- ARM computation ---------------------------------------------------------------------
@dataclass
class SegmentStats:
    N: int
    occ: dict
    both: dict
    fwd: dict
    fwd_direct: dict
    rep: dict
    rep_direct: dict
    first_pos: dict      # traces where activity is the first event
    last_pos: dict       # traces where activity is the last event


def segment_stats(traces, add_start_end=False) -> SegmentStats:
    var = Counter(traces)
    occ, both, fwd, fwd_d = Counter(), Counter(), Counter(), Counter()
    rep, rep_d, fst, lst = Counter(), Counter(), Counter(), Counter()
    N = 0
    for v, c in var.items():
        if add_start_end:
            v = (START,) + v + (END,)
        N += c
        if not v:
            continue
        fst[v[0]] += c
        lst[v[-1]] += c
        first = {}
        cnt = Counter(v)
        for i, a in enumerate(v):
            if a not in first:
                first[a] = i
        dsucc = set(zip(v, v[1:]))
        acts = list(first)
        for a in acts:
            occ[a] += c
            if cnt[a] > 1:
                rep[a] += c
                if (a, a) in dsucc:
                    rep_d[a] += c
        for a in acts:
            fa = first[a]
            for b in acts:
                if a == b:
                    continue
                both[(a, b)] += c
                if fa < first[b]:
                    fwd[(a, b)] += c
                    if (a, b) in dsucc:
                        fwd_d[(a, b)] += c
    return SegmentStats(N, dict(occ), dict(both), dict(fwd), dict(fwd_d), dict(rep), dict(rep_d),
                        dict(fst), dict(lst))


def pair_cell(s: SegmentStats, a, b, tau):
    eps = 1e-12
    N, na, nb = s.N, s.occ[a], s.occ[b]
    nab = s.both.get((a, b), 0)
    union = na + nb - nab
    only_a, only_b = na - nab, nb - nab
    # "never co-occur": co-occurrence is within the noise share from the viewpoint of each activity
    # (relevant traces: those containing A, resp. B). Symmetric counterpart of the implication test,
    # so that a rare activity that always co-occurs with a frequent one is not mistaken for ×.
    never_both = nab / na <= 1 - tau + eps and nab / nb <= 1 - tau + eps
    # existential
    imp_ab = nab / na >= tau - eps
    imp_ba = nab / nb >= tau - eps
    if imp_ab and imp_ba:
        e = ("eq", 0)
    elif imp_ab:
        e = ("imp", 1)
    elif imp_ba:
        e = ("imp", -1)
    elif never_both:
        e = ("xor", 0) if (only_a + only_b) / N >= tau - eps else ("nand", 0)
    elif union / N >= tau - eps:
        e = ("or", 0)
    else:
        e = ("ind", 0)
    # temporal
    if never_both:
        t = ("x", 0)
    else:
        f_ab, f_ba = s.fwd.get((a, b), 0), s.fwd.get((b, a), 0)
        if f_ab / nab >= tau - eps:
            d, k = 1, s.fwd_direct.get((a, b), 0) / f_ab
        elif f_ba / nab >= tau - eps:
            d, k = -1, s.fwd_direct.get((b, a), 0) / f_ba
        else:
            d = 0
        if d == 0:
            t = ("i", 0)
        elif k >= tau - eps:
            t = ("d", d)
        elif k <= 1 - tau + eps:
            t = ("t", d)
        else:
            t = ("e", d)
    return (t, e)


DIAG_NONE = (("x", 0), ("ind", 0))   # activity does not repeat: (×, −)


def diag_cell(s: SegmentStats, a, tau):
    """Self pair: (×,−) if the activity does not repeat beyond the noise share (1 - tau),
    otherwise (≺d/≺/≺t, ⇐) with the directness of the repetitions."""
    eps = 1e-12
    r = s.rep.get(a, 0)
    if r / s.occ[a] <= 1 - tau + eps:
        return DIAG_NONE
    k = s.rep_direct.get(a, 0) / r
    base = "d" if k >= tau - eps else ("t" if k <= 1 - tau + eps else "e")
    return ((base, 1), ("imp", -1))


@dataclass
class ARM:
    sigma: set
    cells: dict        # (a,b) with a<b -> cell
    diag: dict         # a -> cell (DIAG_NONE if a does not repeat)
    stats: SegmentStats


def compute_arm(traces, tau, add_start_end=False, min_support=1) -> ARM:
    return arm_from_stats(segment_stats(traces, add_start_end), tau, min_support)


def arm_from_stats(s: SegmentStats, tau, min_support=1) -> ARM:
    sigma = {a for a, n in s.occ.items() if n >= min_support}
    acts = sorted(sigma)
    cells = {}
    for i, a in enumerate(acts):
        for b in acts[i + 1:]:
            cells[(a, b)] = pair_cell(s, a, b, tau)
    diag = {a: diag_cell(s, a, tau) for a in acts}
    return ARM(sigma, cells, diag, s)


# --- comparison --------------------------------------------------------------------------
def transition(c0, c1):
    """Return (kind, ids, detail) for a changed pair (both defined)."""
    (t0, e0), (t1, e1) = c0, c1
    ids = []
    t_ch, e_ch = t0 != t1, e0 != e1
    if t_ch:
        ids.append(T_ID[(t0[0], t1[0])])
    if e_ch:
        ids.append(E_ID[(e0[0], e1[0])])
    kind = "mixed" if (t_ch and e_ch) else ("temporal" if t_ch else "existential")
    return kind, ids


@dataclass
class DriftResult:
    drift_class: str
    sigma_added: list
    sigma_removed: list
    n_temporal: int
    n_existential: int
    n_mixed: int
    n_existence: int
    ids: Counter
    pairs: list          # (a, b, before_str, after_str, kind, ids), diagonal pairs have a == b
    diag_changes: list   # (a, before_str, after_str)
    inadmissible_E: list
    invalid_cells: list


def compare(arm0: ARM, arm1: ARM) -> DriftResult:
    s0, s1 = arm0.sigma, arm1.sigma
    added, removed = sorted(s1 - s0), sorted(s0 - s1)
    keys = set(arm0.cells) | set(arm1.cells)
    nt = ne = nm = nx = 0
    ids, pairs, inad, invalid = Counter(), [], [], []
    for arm in (arm0, arm1):
        for k, c in arm.cells.items():
            if not is_valid(c):
                invalid.append((k, cell_str(c)))
    for k in sorted(keys):
        c0, c1 = arm0.cells.get(k), arm1.cells.get(k)
        if c0 == c1:
            continue
        if c0 is None or c1 is None:
            nx += 1
            pairs.append((k[0], k[1], cell_str(c0), cell_str(c1), "existence", []))
            continue
        kind, tid = transition(c0, c1)
        nt += kind == "temporal"
        ne += kind == "existential"
        nm += kind == "mixed"
        for i in tid:
            ids[i] += 1
        if kind == "existential" and tid[0] in E_INADMISSIBLE:
            inad.append((k, tid[0]))
        pairs.append((k[0], k[1], cell_str(c0), cell_str(c1), kind, tid))
    # diagonal: (A, A) for activities present on both sides, classified like any other pair
    diag_ch = []
    for a in sorted(s0 & s1):
        c0, c1 = arm0.diag[a], arm1.diag[a]
        if c0 == c1:
            continue
        kind, tid = transition(c0, c1)
        nt += kind == "temporal"
        ne += kind == "existential"
        nm += kind == "mixed"
        for i in tid:
            ids[i] += 1
        diag_ch.append((a, cell_str(c0), cell_str(c1)))
        pairs.append((a, a, cell_str(c0), cell_str(c1), kind, tid))
    if added or removed:
        cls = "existence"
    elif nt + ne + nm == 0:
        cls = "no_drift"
    elif ne == 0 and nm == 0:
        cls = "temporal-only"
    elif nt == 0 and nm == 0:
        cls = "existential-only"
    else:
        cls = "mixed"
    return DriftResult(cls, added, removed, nt, ne, nm, nx, ids, pairs, diag_ch, inad, invalid)


# --- segmentation ------------------------------------------------------------------------
def make_segments(n, changes, margin_frac=0.05, max_margin_share=0.25):
    """changes: sorted list of (start, end) trace indices (0-based, `start` = first trace that may
    belong to the new behaviour, `end` = first trace that surely does; sudden: start == end).
    Returns list of (lo, hi) half-open stable segments, len(changes)+1 of them.
    Margin per side = min(margin_frac * n, max_margin_share * untrimmed segment length)."""
    bounds = [0] + [x for c in changes for x in c] + [n]
    segs = []
    m = margin_frac * n
    for k in range(len(changes) + 1):
        lo, hi = bounds[2 * k], bounds[2 * k + 1]
        L = max(hi - lo, 0)
        mm = min(m, max_margin_share * L)
        lo2 = lo + (0 if k == 0 else int(round(mm)))
        hi2 = hi - (0 if k == len(changes) else int(round(mm)))
        segs.append((lo2, max(hi2, lo2)))
    return segs
