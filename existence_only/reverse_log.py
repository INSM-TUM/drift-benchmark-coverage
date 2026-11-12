"""Reverse a drift log: post-drift traces first, so the drift runs backwards (e.g. OR_AND -> AND_OR, E18 -> E9).

usage: python reverse_log.py <category_dir> <SRC> <DST>     e.g. isolated OR_AND AND_OR
Trace order is reversed; each trace is time-shifted onto the start time of the trace that held its
new position, so time order == new order. model:version is flipped (0 <-> 1). Change indices mirror:
[s, e] -> [N - e, N - s]. A row for DST is appended to <category_dir>/drift_info.csv.
"""
import re, sys, os
from datetime import datetime
from lxml import etree

NS = "{http://www.xes-standard.org/}"
FMT = "%Y-%m-%dT%H:%M:%S.%f"
cat, src, dst = sys.argv[1:4]
src_f, dst_f = f"log_{src}.xes", f"log_{dst}.xes"

tree = etree.parse(os.path.join(cat, src_f))
root = tree.getroot()
traces = root.findall(f"{NS}trace")
dates = lambda t: t.findall(f"{NS}event/{NS}date[@key='time:timestamp']")
first = lambda t: datetime.fromisoformat(dates(t)[0].get("value"))
starts = [first(t) for t in traces]

for t in traces:
    root.remove(t)
for i, t in enumerate(reversed(traces)):
    shift = starts[i] - first(t)
    for d in dates(t):
        d.set("value", (datetime.fromisoformat(d.get("value")) + shift).strftime(FMT))
    t.find(f"{NS}string[@key='concept:name']").set("value", str(i + 1))
    v = t.find(f"{NS}int[@key='model:version']")
    if v is not None:
        v.set("value", str(1 - int(v.get("value"))))
    root.append(t)
tree.write(os.path.join(cat, dst_f), xml_declaration=True, encoding="utf-8", pretty_print=True)

# drift_info: mirror change indices
n = len(traces)
info = os.path.join(cat, "drift_info.csv")
rows = [l.rstrip("\n") for l in open(info) if l.startswith(src_f + ";")]
idx = {k: int(re.search(r"\[(\d+)\]", l).group(1)) for l in rows
       for k in ("change_start", "change_end") if f";{k};" in l and "[" in l}
s, e = idx["change_start"], idx.get("change_end", idx["change_start"])
new = {"change_start": n - e, "change_end": n - s}
out = []
for l in rows:
    l = l.replace(src_f, dst_f)
    for k, v in new.items():
        if f";{k};" in l and "[" in l:
            ts = starts[v].strftime("%Y-%m-%d %H:%M:%S.%f")
            l = re.sub(r";[^;]*\[\d+\]$", f";{ts} [{v}]", l)
    out.append(l)
with open(info, "a") as f:
    f.write("\n".join(out) + "\n")
print(f"{cat}/{dst_f}: {n} traces, change {new['change_start']}-{new['change_end']}")
