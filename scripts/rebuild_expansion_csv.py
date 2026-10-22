"""Rebuild paper/data/expansion_runs_summary.csv from the frozen run dirs.

Covers H-sweep, PID-Lagrangian (ALL main-gain seeds plus alternative-gain
runs, gains read from each run's pid_protocol.json -- never from dir names),
and PCGrad surgery. One row per run: per-contract means aggregated from
joint_eval.json.
"""
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "data" / "expansion_runs_summary.csv"

rows = []


def add_run(path, label):
    eval_file = path / "joint_eval.json"
    if not eval_file.exists():
        print(f"  skip (no joint_eval.json): {path.name}")
        return
    recs = json.load(open(eval_file, encoding="utf-8"))
    if not recs:
        return
    viol = sum(r["violation"] for r in recs) / len(recs)
    cost = sum(r["cost"] for r in recs) / len(recs)
    fs = sum(bool(r["full_contract_met"]) for r in recs)
    fsv = sum(bool(r["valid_full_contract_met"]) for r in recs)
    rows.append({
        "method": label,
        "seed": recs[0]["seed"],
        "mean_violation": round(viol, 4),
        "mean_cost": round(cost, 4),
        "fullstrict_met": fs,
        "fullstrict_valid": fsv,
        "contracts_total": len(recs),
    })
    print(f"  {label:38s} s{recs[0]['seed']}  viol {viol:.4f}  "
          f"cost {cost:.4f}  {fsv}/{len(recs)}")


for H in (200, 400, 800):
    base = ROOT / "results" / f"hsweep_H{H}"
    print(f"[H-sweep H={H}]")
    for d in sorted(base.glob("lambda50_b1_s*")):
        add_run(d, f"H-sweep H={H}")

print("[PCGrad surgery]")
for d in sorted((ROOT / "results" / "gradsurgery").glob("pcgrad_*")):
    add_run(d, "PCGrad surgery")

print("[PID-Lagrangian]")
pid_root = ROOT / "results" / "pid_lagrangian"
for d in sorted(pid_root.iterdir()):
    if not (d / "joint_eval.json").exists():
        continue
    proto = json.load(open(d / "pid_protocol.json", encoding="utf-8"))
    if "smoke" in d.name:
        continue
    gains = (proto["kp"], proto["ki"], proto["kd"])
    main = gains == (20.0, 1.0, 4.0)
    label = ("PID-Lagrangian" if main else
             f"PID alt-gain kp={gains[0]:g},ki={gains[1]:g},kd={gains[2]:g}")
    print(f"  dir={d.name}  gains={gains}  -> {label}")
    add_run(d, label)

if OUT.exists():
    shutil.copy(OUT, OUT.with_suffix(".csv.bak"))
header = ("method,seed,mean_violation,mean_cost,fullstrict_met,"
          "fullstrict_valid,contracts_total\n")
with open(OUT, "w", encoding="utf-8", newline="") as f:
    f.write(header)
    for r in rows:
        f.write(f"{r['method']},{r['seed']},{r['mean_violation']},"
                f"{r['mean_cost']},{r['fullstrict_met']},"
                f"{r['fullstrict_valid']},{r['contracts_total']}\n")
print(f"\nwrote {OUT} ({len(rows)} rows; backup at .csv.bak)")

main_rows = [r for r in rows if r["method"] == "PID-Lagrangian"]
if len(main_rows) == 3:
    import statistics as st
    v = [r["mean_violation"] for r in main_rows]
    c = [r["mean_cost"] for r in main_rows]
    n = sum(r["fullstrict_valid"] for r in main_rows)
    print(f"PID main-gain 3-seed check: viol {st.mean(v):.3f}±"
          f"{st.pstdev(v):.3f}  cost {st.mean(c):.3f}±{st.pstdev(c):.3f}  "
          f"FullStrictValid {n}/12")
