"""Build results/{own,cdrift,cdlg}_drift_classes.csv from the raw outputs in out/.

One row per drift (CDLG: per change event), tau = 0.95, no artificial start/end activities.
Run run_verification.py (and optionally armature_variant.py) first.
"""
import csv
import os
import re

import run_verification as rv
from armdrift import E_ID

TAU, SE = "0.95", "no"
NAME2E = {"IMP": "imp", "AND": "eq", "OR": "or", "INDEPENDENCE": "ind", "NAND": "nand", "XOR": "xor"}
ESYM = {"imp": "⇒", "eq": "⇔", "or": "∨", "ind": "−", "nand": "⊼", "xor": "⇎"}


def read(name):
    p = os.path.join(rv.OUT, name)
    return list(csv.DictReader(open(p, encoding="utf-8-sig"))) if os.path.exists(p) and os.path.getsize(p) else []


def split_ids(s, prefix):
    return ";".join(x for x in s.split(";") if x.startswith(prefix))


def expected(log):
    m = re.search(r"log_([A-Z]+)_([A-Z]+)\.xes", log)
    if not m:
        return ""
    a, b = NAME2E[m.group(1)], NAME2E[m.group(2)]
    return f"{ESYM[a]}→{ESYM[b]} ({E_ID[(a, b)]})"


# Manual corrections: drifts the first-occurrence ordering misclassifies; class_dfg_based gives the evidence.
MANUAL = {
    ("cdlg", "log_168_1726735314.xes", "drift_1/change_info_1"): (
        "mixed",
        "manually set to mixed (script: existential-only): the redo part of a loop is swapped with the subsequent "
        "sequence, which necessarily changes the order of the swapped activities; first-occurrence ordering misses it "
        "because all activities lie in an outer loop, the DFG-based ARM shows (e,l) ≻d→−, (b,g) ≻d→−, (b,l) −→≻d"),
}


def table(bench, head, extra=lambda r: {}):
    dfg = {(r["log"], r["drift_index"]): r["drift_class"] for r in read(f"{bench}_armature_drifts.csv")
           if r["start_end"] == SE}
    out = []
    for r in read(f"{bench}_drifts.csv"):
        if r["tau"] != TAU or r["start_end"] != SE:
            continue
        cls, note = MANUAL.get((bench, r["log"], r.get("drift_label", "")), (r["drift_class"], ""))
        out.append({k: r[k] for k in head} | extra(r) | dict(
            drift_class=cls,
            class_dfg_based=dfg.get((r["log"], r["drift_index"]), ""),
            temporal_transitions=split_ids(r["transition_ids"], "T"),
            existential_transitions=split_ids(r["transition_ids"], "E"),
            n_temporal_pairs=r["n_temporal"], n_existential_pairs=r["n_existential"], n_mixed_pairs=r["n_mixed"],
            activities_added=r["sigma_added"], activities_removed=r["sigma_removed"]) | (
            dict(note=note) if bench == "cdlg" else {}))
    os.makedirs("results", exist_ok=True)
    rv.write_csv(os.path.join("results", f"{bench}_drift_classes.csv"), out)
    print(bench, len(out))


def cdlg_extra():
    info = rv.load_cdlg_info()
    ch = {(name, c["label"]): c for name, L in info.items() for c in L["changes"]}

    def f(r):
        c = ch[(r["log"], r["drift_label"])]
        return dict(tree_before=c["tree_before"], tree_after=c["tree_after"], documented_added="|".join(c["added"]),
                    documented_deleted="|".join(c["deleted"]), documented_moved="|".join(c["moved"]))
    return f


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    table("own", ["log", "change_point"], lambda r: dict(expected_from_name=expected(r["log"])))
    table("cdrift", ["dataset", "log", "op", "noise", "drift_index", "change_point"])
    table("cdlg", ["log", "drift_index", "drift_label", "drift_type", "change_type", "change_point"], cdlg_extra())
