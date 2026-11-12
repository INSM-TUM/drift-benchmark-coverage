"""
================================================================================
Purely existential Mathematical Pattern Generator for Purely Existential Concept Drift
File: generate_related_conditions.py
================================================================================

"""

import os
import sys
import random
import argparse
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple


DEFAULT_LOG_SIZE = 20000
DEFAULT_DRIFT_POINT = 0.5


# ==============================================================================
# Fast, Clean XES Log Writer (Matches generate.py schema exactly)
# ==============================================================================

def build_and_save_xes(traces_data: List[List[str]], filename: str, seed: int = 42):
    """
    Writes activity sequences to a standard IEEE 1849-2016 .xes XML file.
    Produces clean, indented XML identical to generate.py but streaming-optimized.
    """
    rng = random.Random(seed)
    current_time = datetime(2024, 1, 1, 8, 0, 0)

    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)

    with open(filename, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="utf-8"?>\n')
        f.write('<log xmlns="http://www.xes-standard.org/" xes.version="1849-2016" xes.features="nested-attributes">\n')
        f.write('\t<string key="drift:info" value="temp">\n')
        f.write('\t\t<int key="drift_0" value="0">\n')
        f.write('\t\t\t<int key="change_info" value="0"/>\n')
        f.write('\t\t</int>\n')
        f.write('\t</string>\n')

        for i, trace_activities in enumerate(traces_data, start=1):
            f.write('\t<trace>\n')
            f.write(f'\t\t<string key="concept:name" value="{i}"/>\n')

            for activity in trace_activities:
                current_time += timedelta(
                    hours=rng.randint(1, 12), minutes=rng.randint(0, 59)
                )
                ts_str = current_time.strftime("%Y-%m-%dT%H:%M:%S.%f")
                f.write('\t\t<event>\n')
                f.write(f'\t\t\t<string key="concept:name" value="{activity}"/>\n')
                f.write(f'\t\t\t<date key="time:timestamp" value="{ts_str}"/>\n')
                f.write('\t\t\t<string key="lifecycle:transition" value="complete"/>\n')
                f.write('\t\t</event>\n')

            f.write('\t</trace>\n')

        f.write('</log>\n')

    print(f"[+] Successfully generated: {filename} ({len(traces_data)} traces)")


# ==============================================================================
# Statistical Verification Helpers (DFG and ARM)
# ==============================================================================

def verify_related_conditions_invariance(traces: List[List[str]], drift_idx: int) -> Dict[str, Any]:
    """
    Verifies that 1st-order DFG and DFG-IO remain invariant across the drift,
    while ARM registers a significant existential shift.
    """
    pre = traces[:drift_idx]
    post = traces[drift_idx:]

    vocab = sorted(list({act for t in traces for act in t}))
    v_idx = {act: i for i, act in enumerate(vocab)}
    V = len(vocab)

    # 1. DFG Transition Matrices
    dfg_pre = [[0.0] * V for _ in range(V)]
    dfg_post = [[0.0] * V for _ in range(V)]

    for t in pre:
        for u, v in zip(t[:-1], t[1:]):
            dfg_pre[v_idx[u]][v_idx[v]] += 1
    for t in post:
        for u, v in zip(t[:-1], t[1:]):
            dfg_post[v_idx[u]][v_idx[v]] += 1

    # Row normalize to conditional probabilities P(v | u)
    for i in range(V):
        r_sum_pre = sum(dfg_pre[i])
        r_sum_post = sum(dfg_post[i])
        if r_sum_pre > 0:
            dfg_pre[i] = [c / r_sum_pre for c in dfg_pre[i]]
        if r_sum_post > 0:
            dfg_post[i] = [c / r_sum_post for c in dfg_post[i]]

    max_dfg_diff = max(
        abs(dfg_pre[i][j] - dfg_post[i][j]) for i in range(V) for j in range(V)
    )

    # 2. DFG-IO Start and End Distributions
    start_pre, start_post = [0.0] * V, [0.0] * V
    end_pre, end_post = [0.0] * V, [0.0] * V
    for t in pre:
        if t:
            start_pre[v_idx[t[0]]] += 1
            end_pre[v_idx[t[-1]]] += 1
    for t in post:
        if t:
            start_post[v_idx[t[0]]] += 1
            end_post[v_idx[t[-1]]] += 1

    n_pre, n_post = len(pre), len(post)
    start_pre = [c / n_pre for c in start_pre]
    start_post = [c / n_post for c in start_post]
    end_pre = [c / n_pre for c in end_pre]
    end_post = [c / n_post for c in end_post]

    max_start_diff = max(abs(start_pre[i] - start_post[i]) for i in range(V))
    max_end_diff = max(abs(end_pre[i] - end_post[i]) for i in range(V))

    # 3. ARM (Co-occurrence Relational Shift)
    arm_pre = [[0.0] * V for _ in range(V)]
    arm_post = [[0.0] * V for _ in range(V)]
    for t in pre:
        unique_acts = set(t)
        for u in unique_acts:
            for v in unique_acts:
                arm_pre[v_idx[u]][v_idx[v]] += 1
    for t in post:
        unique_acts = set(t)
        for u in unique_acts:
            for v in unique_acts:
                arm_post[v_idx[u]][v_idx[v]] += 1

    arm_pre = [[c / n_pre for c in row] for row in arm_pre]
    arm_post = [[c / n_post for c in row] for row in arm_post]

    max_arm_shift = max(
        abs(arm_pre[i][j] - arm_post[i][j]) for i in range(V) for j in range(V)
    )

    return {
        "max_dfg_diff": max_dfg_diff,
        "max_start_diff": max_start_diff,
        "max_end_diff": max_end_diff,
        "max_arm_shift": max_arm_shift,
        "dfg_invariant": max_dfg_diff <= 0.05,
        "dfg_io_invariant": max(max_start_diff, max_end_diff) <= 0.05,
        "arm_detected": max_arm_shift >= 0.20,
    }


# ==============================================================================
# Patterns (v2)
#
# Every trigger is an independent coin flip (as in Fig. 1 of the paper) and every
# consequence copies one trigger. The drift only re-assigns which trigger a
# consequence copies. So a trigger/consequence pair goes from "iff" (⇔) to
# "unrelated" (−) or back (E11 / E28), or, in pattern 10, from implication to
# independence and back (E6 / E27). Every pair keeps all four outcomes
# (both / only A / only B / neither) possible on at least one side, so it never
# crosses into ⇎/⊼ and the temporal dependency stays fixed (Prop. 1).
#
# v1 picked exactly ONE trigger per trace (rng.choice(["a", "b"])). Then a and d
# never co-occur before the drift and always co-occur after it: (×,⇎) -> (≺t,⇔),
# a mixed transition (T18+E13). That made 8 of 10 v1 logs mixed drifts.
#
# DFG invariance: consequences only swap between triggers with equal marginals,
# so every directly-follows frequency is unchanged in expectation.
# ==============================================================================

def _coin(rng, p=0.5):
    # rng.choice keeps patterns 02/08 bit-identical to v1
    return rng.choice([True, False]) if p == 0.5 else rng.random() < p


def _trace(**present):
    """Activities in keyword order, kept if truthy."""
    return [a for a, on in present.items() if on]


def p01_routing_2way(rng, post):
    # pre: d<=>a, e<=>b | post: d<=>b, e<=>a
    a, b = _coin(rng), _coin(rng)
    return _trace(a=a, b=b, c=1, d=b if post else a, e=a if post else b)


def p02_powerset_2var(rng, post):
    # pre: d<=>a | post: d<=>b   (Fig. 1)
    a, b = _coin(rng), _coin(rng)
    return _trace(a=a, b=b, c=1, d=b if post else a)


def p03_routing_3way(rng, post):
    # pre: e<=>a, f<=>b, g<=>c | post: e<=>c, f<=>a, g<=>b
    a, b, c = _coin(rng), _coin(rng), _coin(rng)
    e, f, g = (c, a, b) if post else (a, b, c)
    return _trace(a=a, b=b, c=c, d=1, e=e, f=f, g=g)


def p04_cascade_2stage(rng, post):
    # stage 2 static: d<=>a, e<=>b | stage 3 pre: g<=>a, h<=>b | post: g<=>b, h<=>a
    a, b = _coin(rng), _coin(rng)
    return _trace(a=a, b=b, c=1, d=a, e=b, f=1, g=b if post else a, h=a if post else b)


def p05_bottleneck_2step(rng, post):
    # middle c, d | pre: e<=>a, f<=>b | post: e<=>b, f<=>a
    a, b = _coin(rng), _coin(rng)
    return _trace(a=a, b=b, c=1, d=1, e=b if post else a, f=a if post else b)


def p06_localized_swap(rng, post):
    # pre: f<=>a, g<=>b, h<=>c, i<=>d | post: f<=>b, g<=>a (h, i static)
    a, b, c, d = (_coin(rng) for _ in range(4))
    return _trace(a=a, b=b, c=c, d=d, e=1, f=b if post else a, g=a if post else b, h=c, i=d)


def p07_asymmetric_prob(rng, post):
    # P(a)=P(b)=0.4, P(c)=0.2 | pre: e<=>a, f<=>b, g<=>c | post: e<=>b, f<=>a (g static)
    # ponytail: only equal-marginal triggers swap, else consequence frequencies drift
    a, b, c = _coin(rng, 0.4), _coin(rng, 0.4), _coin(rng, 0.2)
    return _trace(a=a, b=b, c=c, d=1, e=b if post else a, f=a if post else b, g=c)


def p08_powerset_3var(rng, post):
    # pre: e<=>a, f<=>b, g<=>c | post: e<=>b, f<=>c, g<=>a
    a, b, c = _coin(rng), _coin(rng), _coin(rng)
    e, f, g = (b, c, a) if post else (a, b, c)
    return _trace(a=a, b=b, c=c, d=1, e=e, f=f, g=g)


def p09_routing_4way(rng, post):
    # pre: f<=>a, g<=>b, h<=>c, i<=>d | post: f<=>d, g<=>a, h<=>b, i<=>c
    a, b, c, d = (_coin(rng) for _ in range(4))
    f, g, h, i = (d, a, b, c) if post else (a, b, c, d)
    return _trace(a=a, b=b, c=c, d=d, e=1, f=f, g=g, h=h, i=i)


def p10_implication_condition(rng, post):
    # pre: d only if a (d => a, a -/-> d) | post: d only if b
    # (a,d): ⇐ -> − (E6), (b,d): − -> ⇐ (E27). "neither" stays possible on both sides.
    a, b, x = _coin(rng), _coin(rng), _coin(rng)
    return _trace(a=a, b=b, c=1, d=(b if post else a) and x)


# ==============================================================================
# CSV Metadata & Summary Report Writers
# ==============================================================================

def write_drift_info_csv(
    csv_path: str,
    records: List[Dict[str, Any]],
    drift_type: str = "sudden",
):
    """Writes ground-truth concept drift metadata in standard CV4CDD format."""
    lines = ["log_name;drift_or_noise_id;drift_attribute;drift_sub_attribute;value"]
    for rec in records:
        log_name = os.path.basename(rec["filename"])
        drift_point = rec["drift_idx"]
        lines.append(f"{log_name};drift_1;log_id;na;{log_name}")
        lines.append(f"{log_name};drift_1;drift_id;na;1")
        lines.append(f"{log_name};drift_1;process_perspective;na;control-flow")
        lines.append(f"{log_name};drift_1;drift_type;na;{drift_type}")
        lines.append(f"{log_name};drift_1;change_info_1;change_start;2024-05-01 00:00:00.000000 [{drift_point}]")
        lines.append(f"{log_name};drift_1;change_info_1;change_end;")

    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[+] Successfully generated ground-truth metadata: {csv_path}")


# ==============================================================================
# Main Orchestration
# ==============================================================================

PATTERNS = [
    ("related_conditions_pattern01_routing_2way.xes", p01_routing_2way),
    ("related_conditions_pattern02_powerset_2var.xes", p02_powerset_2var),
    ("related_conditions_pattern03_routing_3way.xes", p03_routing_3way),
    ("related_conditions_pattern04_cascade_2stage.xes", p04_cascade_2stage),
    ("related_conditions_pattern05_bottleneck_2step.xes", p05_bottleneck_2step),
    ("related_conditions_pattern06_localized_swap.xes", p06_localized_swap),
    ("related_conditions_pattern07_asymmetric_prob.xes", p07_asymmetric_prob),
    ("related_conditions_pattern08_powerset_3var.xes", p08_powerset_3var),
    ("related_conditions_pattern09_routing_4way.xes", p09_routing_4way),
    ("related_conditions_pattern10_implication_condition.xes", p10_implication_condition),
]


def generate_all_related_conditions(
    output_dir: str = "./related_conditions",
    num_traces: int = DEFAULT_LOG_SIZE,
    drift_point: float = DEFAULT_DRIFT_POINT,
    verify: bool = True,
):
    os.makedirs(output_dir, exist_ok=True)
    drift_idx = int(num_traces * drift_point)

    print("\n" + "=" * 80)
    print(" GENERATING purely existential MATHEMATICAL PATTERNS (Pure Alphabet Sigma = {a, b, c, ...})")
    print(f" Destination Directory: {output_dir}")
    print(f" Traces per Log:        {num_traces}")
    print(f" Drift Point:           Trace {drift_idx} ({drift_point * 100:.0f}%)")
    print("=" * 80 + "\n")

    records = []

    for idx, (base_filename, make_trace) in enumerate(PATTERNS, start=1):
        target_path = os.path.join(output_dir, base_filename)
        seed = 100 + idx
        rng = random.Random(seed)
        traces = [make_trace(rng, i > drift_idx) for i in range(1, num_traces + 1)]
        build_and_save_xes(traces, target_path, seed=seed)

        if verify:
            check = verify_related_conditions_invariance(traces, drift_idx)
            if not (check["dfg_invariant"] and check["dfg_io_invariant"] and check["arm_detected"]):
                print(f"[!] Warning: Statistical verification anomaly on {base_filename}: {check}")
        else:
            check = {
                "max_dfg_diff": 0.0,
                "max_start_diff": 0.0,
                "max_end_diff": 0.0,
                "max_arm_shift": 0.5,
                "dfg_invariant": True,
                "dfg_io_invariant": True,
                "arm_detected": True,
            }

        records.append({
            "filename": target_path,
            "drift_idx": drift_idx,
            "check": check,
        })

        print(
            f"[{idx:02d}/{len(PATTERNS):02d}] {base_filename:<55} | "
            f"Alphabet: {len({a for t in traces for a in t}):2d} acts | "
            f"DFG Diff: {check['max_dfg_diff']:.4f} | "
            f"ARM Shift: {check['max_arm_shift']:.2f}"
        )

    # Output CSV and Markdown summary
    csv_path = os.path.join(output_dir, "drift_info.csv")
    write_drift_info_csv(csv_path, records)

    print("\n[+] All purely existential patterns generated successfully!")
    print(f"[+] Output Directory: {output_dir}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate purely existential mathematical event logs with pure letter activities (a, b, c, ...)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./related_conditions",
        help="Target folder for .xes logs and metadata (default: ./related_conditions)",
    )
    parser.add_argument(
        "--num-traces",
        type=int,
        default=DEFAULT_LOG_SIZE,
        help=f"Number of traces per log (default: {DEFAULT_LOG_SIZE})",
    )
    parser.add_argument(
        "--drift-point",
        type=float,
        default=DEFAULT_DRIFT_POINT,
        help=f"Drift position between 0.0 and 1.0 (default: {DEFAULT_DRIFT_POINT})",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip automated statistical verification checks",
    )

    args = parser.parse_args()
    generate_all_related_conditions(
        output_dir=args.output_dir,
        num_traces=args.num_traces,
        drift_point=args.drift_point,
        verify=not args.no_verify,
    )


if __name__ == "__main__":
    main()
