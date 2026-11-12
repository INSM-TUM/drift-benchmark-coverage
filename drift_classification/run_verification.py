"""Classify every drift of the existential dataset, CDRIFT and a CDLG sample (Defs. 3-5).

Usage:  python run_verification.py [own|cdrift|cdlg|all] [--cdlg-n 1000] [--procs 16]
Raw outputs go to out/; make_tables.py turns them into results/*.csv.
"""

from __future__ import annotations

import argparse
import ast
import csv
import glob
import os
import re
import sys
import time
from datetime import datetime, timezone
import zipfile
from collections import Counter, defaultdict
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from armdrift import (E_ADMISSIBLE, E_ID, START, END, arm_from_stats, compare, make_segments,  # noqa: E402
                      read_xes, segment_stats)

HERE = os.path.dirname(os.path.abspath(__file__))
# data locations (override with environment variables if the data lives elsewhere)
OWN_ROOT = os.environ.get("DRIFT_OWN_ROOT", os.path.join(HERE, "../existence_only"))
CDRIFT_ROOT = os.environ.get("DRIFT_CDRIFT_ROOT", os.path.join(HERE, "data/cdrift/inscope"))
CDLG_INFO = os.environ.get("DRIFT_CDLG_INFO", os.path.join(HERE, "data/cdlg/test_drift_info.csv"))
CDLG_BATCHES = os.environ.get("DRIFT_CDLG_BATCHES", os.path.join(HERE, "data/cdlg/batches"))
OUT = os.path.join(HERE, "out")

TAUS = (0.95, 0.90)
SE_VARIANTS = (False, True)
MARGIN = 0.05

OSTOVAR_OP = {"ConditionalMove": "cm", "ConditionalRemoval": "cre", "ConditionalToSequence": "cf",
              "Frequency": "fr", "Loop": "lp", "ParallelMove": "pm", "ParallelRemoval": "pre",
              "ParallelToSequence": "pl", "SerialMove": "sm", "SerialRemoval": "sre", "Skip": "cb",
              "Substitute": "rp", "Swap": "sw"}


# ----------------------------------------------------------------------------------------- jobs
def own_jobs():
    jobs = []
    for info in sorted(glob.glob(os.path.join(OWN_ROOT, "*", "drift_info.csv"))):
        folder = os.path.basename(os.path.dirname(info))
        per = defaultdict(dict)
        with open(info) as f:
            for r in csv.DictReader(f, delimiter=";"):
                per[r["log_name"]][r["drift_sub_attribute"] if r["drift_attribute"].startswith("change_info")
                                   else r["drift_attribute"]] = r["value"]
        for name, d in sorted(per.items()):
            s = re.search(r"\[(\d+)\]", d.get("change_start", ""))
            e = re.search(r"\[(\d+)\]", d.get("change_end", "") or "")
            s = int(s.group(1))
            e = int(e.group(1)) if e else s
            jobs.append(dict(benchmark="own", dataset="existential", log=f"{folder}/{name}", op=folder, noise="",
                             src=("file", os.path.join(OWN_ROOT, folder, name + ".gz")),
                             changes=[dict(start=s, end=e, label="drift_1", type=d.get("drift_type", ""),
                                           change_type="gradual" if e != s else "sudden")]))
    return jobs


def cdrift_jobs():
    jobs = []
    for p in sorted(glob.glob(os.path.join(CDRIFT_ROOT, "*.xes"))):
        n = os.path.basename(p)
        if n == "bose_log.xes":
            cps, op, noise, ds = [1199, 2399, 3599, 4799], "bose", "", "Bose"
        elif n.startswith("sudden_trace"):
            _, _, nz, size, op = n[:-4].split("_")
            cps, noise, ds = [int(size) // 2 - 1], nz.replace("noise", ""), "Ceravolo"
        else:
            m = re.match(r"Atomic_(\w+?)_output_\w+?(?:_(\d))?\.xes", n)
            op, noise, ds = OSTOVAR_OP[m.group(1)], (m.group(2) or "0"), "Ostovar"
            cps = [999, 1999]
        # cdrift-evaluation: cp = index of the last trace of the old version -> first new = cp + 1
        jobs.append(dict(benchmark="cdrift", log=n, op=op, noise=noise, dataset=ds, src=("file", p),
                         changes=[dict(start=c + 1, end=c + 1, label=f"cp{i+1}", type="sudden",
                                       change_type="sudden") for i, c in enumerate(cps)]))
    return jobs


def parse_list(s):
    s = (s or "").strip()
    if s in ("", "[]"):
        return []
    return [x.strip() for x in s.strip("[]").split(",") if x.strip()]


def load_cdlg_info():
    logs = defaultdict(lambda: dict(noise=False, drifts=defaultdict(dict)))
    with open(CDLG_INFO) as f:
        for r in csv.DictReader(f, delimiter=";"):
            L = logs[r["log_name"]]
            did = r["drift_or_noise_id"]
            if did.startswith("noise"):
                L["noise"] = True
                continue
            att, sub, val = r["drift_attribute"], r["drift_sub_attribute"], r["value"]
            if att == "drift_type":
                L["drifts"][did]["type"] = val
            elif att.startswith("change_info"):
                L["drifts"][did].setdefault("changes", defaultdict(dict))[att][sub] = val
    out = {}
    for name, L in logs.items():
        ch = []
        for did, d in L["drifts"].items():
            if not d.get("type"):
                continue
            for ci, c in sorted(d.get("changes", {}).items(), key=lambda kv: int(kv[0].split("_")[-1])):
                idx = ast.literal_eval(c["change_trace_index"])
                # CDLG: change_trace_index = len(log_so_far) + 1 (1-based first trace of the new part)
                s = idx[0] - 1
                e = (idx[1] - 1) if len(idx) > 1 else s
                ch.append(dict(start=s, end=e, label=f"{did}/{ci}", type=d["type"], change_type=c["change_type"],
                               tree_before=c["process_tree_before"], tree_after=c["process_tree_after"],
                               added=parse_list(c["activities_added"]), deleted=parse_list(c["activities_deleted"]),
                               moved=parse_list(c["activities_moved"])))
        ch.sort(key=lambda c: c["start"])
        out[name] = dict(noise=L["noise"], changes=ch)
    return out


def cdlg_jobs(n_target):
    info = load_cdlg_info()
    jobs = []
    for z in sorted(glob.glob(os.path.join(CDLG_BATCHES, "xes_batch_*.zip"))):
        with zipfile.ZipFile(z) as zf:
            members = sorted(zf.namelist())
        for m in members:
            name = os.path.basename(m)
            L = info.get(name)
            if L is None or L["noise"] or not L["changes"]:
                continue
            jobs.append(dict(benchmark="cdlg", dataset="CDLG-test", log=name, op="cdlg", noise="0", src=("zip", z, m),
                             changes=L["changes"]))
            if len(jobs) >= n_target:
                return jobs, info
    return jobs, info


# ----------------------------------------------------------------------------------------- worker
def fmt_ids(c: Counter):
    return ";".join(f"{k}x{v}" for k, v in sorted(c.items(), key=lambda kv: (kv[0][0], int(kv[0][1:]))))


def presence_changes(s0, s1, tau):
    """Activities whose mandatory status (present in >= tau of the traces) flips between segments."""
    out = []
    for a in sorted(set(s0.occ) & set(s1.occ)):
        p0, p1 = s0.occ[a] / s0.N, s1.occ[a] / s1.N
        if (p0 >= tau) != (p1 >= tau):
            ref = s0 if p0 >= tau else s1
            out.append(f"{a}:{p0:.3f}->{p1:.3f}[first={ref.first_pos.get(a, 0) / ref.N:.2f},"
                       f"last={ref.last_pos.get(a, 0) / ref.N:.2f}]")
    return out


def _ts_key(s):
    if not s:
        return None
    try:
        d = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if d.tzinfo is None:
            d = d.replace(tzinfo=timezone.utc)
        return d.timestamp()
    except ValueError:
        return None


TRACE_ORDER = "time"


def process(job):
    t0 = time.time()
    try:
        if job["src"][0] == "file":
            L = read_xes(job["src"][1])
        else:
            with zipfile.ZipFile(job["src"][1]) as zf:
                L = read_xes(zf.read(job["src"][2]))
    except Exception as ex:  # noqa: BLE001
        return [], [], dict(log=job["log"], benchmark=job["benchmark"], error=repr(ex))
    traces = L.traces
    n = len(traces)
    changes = job["changes"]
    tsk = [_ts_key(x) for x in L.first_ts]
    ts_sorted = all(a <= b for a, b in zip(tsk, tsk[1:]) if a is not None and b is not None)
    order = "file"
    if TRACE_ORDER == "time" and not ts_sorted and all(k is not None for k in tsk):
        idx = sorted(range(n), key=lambda i: tsk[i])  # stable sort on first-event timestamp
        traces = [traces[i] for i in idx]
        order = "time(sorted)"
    meta = dict(benchmark=job["benchmark"], log=job["log"], n_traces=n, n_empty=L.n_empty,
                lifecycle=dict(L.lifecycle_values), lifecycle_filtered=L.lifecycle_filtered,
                ts_sorted_in_file=ts_sorted, trace_order=order,
                change_out_of_range=any(c["end"] >= n for c in changes))
    segs = make_segments(n, [(c["start"], c["end"]) for c in changes], MARGIN)
    meta["segments"] = segs
    stats = {se: [segment_stats(traces[lo:hi], se) for lo, hi in segs] for se in SE_VARIANTS}
    rows, pairs = [], []
    for k, c in enumerate(changes):
        (lo0, hi0), (lo1, hi1) = segs[k], segs[k + 1]
        for tau in TAUS:
            pres = presence_changes(stats[False][k], stats[False][k + 1], tau) if hi0 > lo0 and hi1 > lo1 else []
            for se in SE_VARIANTS:
                s0, s1 = stats[se][k], stats[se][k + 1]
                base = dict(benchmark=job["benchmark"], dataset=job.get("dataset", ""), log=job["log"],
                            op=job["op"], noise=job["noise"],
                            drift_index=k + 1, drift_label=c["label"], drift_type=c["type"],
                            change_type=c["change_type"],
                            change_point=c["start"] if c["start"] == c["end"] else f"{c['start']}-{c['end']}",
                            seg_before=f"{lo0}-{hi0}", n_before=hi0 - lo0, seg_after=f"{lo1}-{hi1}",
                            n_after=hi1 - lo1, tau=tau, start_end="yes" if se else "no", trace_order=order)
                if s0.N == 0 or s1.N == 0:
                    rows.append(base | dict(drift_class="ERROR_empty_segment"))
                    continue
                r = compare(arm_from_stats(s0, tau), arm_from_stats(s1, tau))
                rows.append(base | dict(
                    drift_class=r.drift_class, class_incl_self_pairs=r.class_with_diag,
                    n_temporal=r.n_temporal, n_existential=r.n_existential, n_mixed=r.n_mixed,
                    n_existence_pairs=r.n_existence, transition_ids=fmt_ids(r.ids),
                    sigma_added="|".join(r.sigma_added), sigma_removed="|".join(r.sigma_removed),
                    self_pair_changes="|".join(f"{a}:{b0}->{b1}" for a, b0, b1 in r.diag_changes),
                    inadmissible_existential="|".join(f"{a},{b}:{i}" for (a, b), i in r.inadmissible_E),
                    invalid_cells="|".join(f"{a},{b}:{x}" for (a, b), x in r.invalid_cells[:5]),
                    presence_changes="|".join(pres)))
                keep = job["benchmark"] in ("own", "cdrift") or r.drift_class == "existential-only"
                if keep:
                    for a, b, b0, b1, kind, ids in r.pairs:
                        pairs.append(dict(benchmark=job["benchmark"], log=job["log"], op=job["op"],
                                          drift_index=k + 1, tau=tau, start_end="yes" if se else "no",
                                          drift_class=r.drift_class, A=a, B=b, before=b0, after=b1, kind=kind,
                                          ids="+".join(ids)))
    meta["seconds"] = round(time.time() - t0, 2)
    return rows, pairs, meta


# ----------------------------------------------------------------------------------------- main
def write_csv(path, rows, fields=None):
    if not rows:
        open(path, "w").close()
        return
    fields = fields or list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8-sig") as f:  # BOM so Excel shows ⇒ ≺ etc.
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _set_taus(t, order):
    global TAUS, TRACE_ORDER
    TAUS = t
    TRACE_ORDER = order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which", nargs="?", default="all")
    ap.add_argument("--cdlg-n", type=int, default=1000)
    ap.add_argument("--procs", type=int, default=os.cpu_count())
    ap.add_argument("--taus", default="0.95,0.90")
    ap.add_argument("--tag", default="")
    ap.add_argument("--order", default="time", choices=["time", "file"])
    a = ap.parse_args()
    global TAUS
    TAUS = tuple(float(x) for x in a.taus.split(","))
    os.makedirs(OUT, exist_ok=True)
    targets = ["own", "cdrift", "cdlg"] if a.which == "all" else [a.which]
    for b in targets:
        if b == "own":
            jobs = own_jobs()
        elif b == "cdrift":
            jobs = cdrift_jobs()
        else:
            jobs, _ = cdlg_jobs(a.cdlg_n)
        print(f"[{b}] {len(jobs)} logs", flush=True)
        rows, pairs, metas = [], [], []
        jobs.sort(key=lambda j: -os.path.getsize(j["src"][1]) if j["src"][0] == "file" else 0)
        with Pool(a.procs, initializer=_set_taus, initargs=(TAUS, a.order)) as pool:
            for i, (r, p, m) in enumerate(pool.imap_unordered(process, jobs, chunksize=1)):
                rows += r
                pairs += p
                metas.append(m)
                if (i + 1) % 50 == 0:
                    print(f"  {i+1}/{len(jobs)}", flush=True)
        rows.sort(key=lambda r: (r["log"], r["drift_index"], -r["tau"], r["start_end"]))
        pairs.sort(key=lambda r: (r["log"], r["drift_index"], -r["tau"], r["start_end"], r["A"], r["B"]))
        write_csv(os.path.join(OUT, f"{b}{a.tag}_drifts.csv"), rows)
        write_csv(os.path.join(OUT, f"{b}{a.tag}_pairs.csv"), pairs)
        write_csv(os.path.join(OUT, f"{b}{a.tag}_logs.csv"), sorted(metas, key=lambda m: m["log"]))
        print(f"[{b}] done: {len(rows)} drift rows", flush=True)


if __name__ == "__main__":
    main()
