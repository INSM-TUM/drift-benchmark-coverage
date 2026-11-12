"""Sensitivity run: classify every drift with armature's own extractors (exact, no tau, no filter).

armature's temporal extractor is DFG/SCC based (loop-aware); armdrift orients pairs by first occurrence.
Both agree on all existential labels and on all temporal labels outside loop contexts
Output: out/<bench>_armature_drifts.csv
"""
import os
import sys
import zipfile
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if os.environ.get("ARMATURE_REPO"):  # otherwise armature must be pip-installed
    sys.path.insert(0, os.path.expanduser(os.environ["ARMATURE_REPO"]))
import run_verification as rv  # noqa: E402
from armdrift import ARM, compare, make_segments, read_xes, START, END  # noqa: E402
from armature.discovery.models import Trace, Event  # noqa: E402
from armature.discovery.existential import extract_existential_dependencies  # noqa: E402
from armature.discovery.temporal import extract_temporal_dependencies  # noqa: E402
from armature.discovery.dfg import build_dfg  # noqa: E402
from armature.discovery.scc import find_sccs  # noqa: E402

TM = {"direct": ("d", 1), "direct_backward": ("d", -1), "eventual": ("e", 1), "eventual_backward": ("e", -1),
      "true_eventual": ("t", 1), "true_eventual_backward": ("t", -1), "independence": ("i", 0),
      "no_ordering": ("x", 0)}
EM = {"implication": ("imp", 1), "implication_backward": ("imp", -1), "equivalence": ("eq", 0),
      "negated_equivalence": ("xor", 0), "or": ("or", 0), "nand": ("nand", 0), "independence": ("ind", 0)}


def armature_arm(traces, se):
    var = set(traces)
    if se:
        var = {(START,) + v + (END,) for v in var}
    T = [Trace(case_id=str(i), events=[Event(activity=a) for a in v]) for i, v in enumerate(var)]
    dfg = build_dfg(T)
    tp = extract_temporal_dependencies(T, dfg, find_sccs(dfg))
    ex = extract_existential_dependencies(T)
    acts = sorted({a for v in var for a in v})
    cells = {(a, b): (TM[str(tp[(a, b)])], EM[str(ex[(a, b)])]) for i, a in enumerate(acts) for b in acts[i + 1:]}
    diag = {a: ((TM[str(tp[(a, a)])], EM[str(ex[(a, a)])]) if (a, a) in tp and (a, a) in ex else None)
            for a in acts}
    return ARM(set(acts), cells, diag, None)


def work(job):
    if job["src"][0] == "file":
        L = read_xes(job["src"][1])
    else:
        with zipfile.ZipFile(job["src"][1]) as zf:
            L = read_xes(zf.read(job["src"][2]))
    traces = L.traces
    tsk = [rv._ts_key(x) for x in L.first_ts]
    if not all(a <= b for a, b in zip(tsk, tsk[1:]) if a is not None and b is not None) and None not in tsk:
        traces = [traces[i] for i in sorted(range(len(traces)), key=lambda i: tsk[i])]
    segs = make_segments(len(traces), [(c["start"], c["end"]) for c in job["changes"]])
    rows = []
    for se in (False, True):
        arms = [armature_arm(traces[lo:hi], se) for lo, hi in segs]
        for k in range(len(job["changes"])):
            r = compare(arms[k], arms[k + 1])
            rows.append(dict(benchmark=job["benchmark"], log=job["log"], op=job["op"], noise=job["noise"],
                             drift_index=k + 1, start_end="yes" if se else "no", drift_class=r.drift_class, class_incl_self_pairs=r.class_with_diag,
                             transition_ids=rv.fmt_ids(r.ids),
                             temporal_pairs="|".join(f"{a},{b}:{b0}->{b1}" for a, b, b0, b1, kd, _ in r.pairs
                                                     if kd in ("temporal", "mixed"))))
    return rows


if __name__ == "__main__":
    which = sys.argv[1]
    jobs = rv.own_jobs() if which == "own" else rv.cdrift_jobs() if which == "cdrift" else rv.cdlg_jobs(1000)[0]
    out = []
    with Pool(os.cpu_count()) as p:
        for r in p.imap_unordered(work, jobs):
            out += r
    out.sort(key=lambda r: (r["log"], r["drift_index"], r["start_end"]))
    rv.write_csv(os.path.join(rv.OUT, f"{which}_armature_drifts.csv"), out)
    print(which, len(out))
