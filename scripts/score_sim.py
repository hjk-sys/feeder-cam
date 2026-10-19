"""Score the pipeline's output on the simulated feeder against its answer key.

    python scripts/score_sim.py

Matches each logged visit to the true visit it overlaps most in time, then
reports how many visits were found, how many were split into several (a
tracking failure: one bird counted twice), and how many species were right.
Evaluation of the video.
"""

import argparse
import json
import sqlite3
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def overlap(a0, a1, b0, b1):
    return max(0.0, (min(a1, b1) - max(a0, b0)).total_seconds())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--key", type=Path, default=ROOT / "data" / "sim" / "feeder_sim.json")
    ap.add_argument("--db", type=Path, default=ROOT / "data" / "sim" / "feeder.db")
    args = ap.parse_args()

    key = json.loads(args.key.read_text())
    t0 = datetime.fromisoformat(key["start_time"])
    truth = [dict(v, t_start=t0 + timedelta(seconds=v["start_s"]), t_end=t0 + timedelta(seconds=v["end_s"]))
             for v in key["visits"]]

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    logged = conn.execute("SELECT * FROM labeled_visits WHERE device = 'sim'").fetchall()

    # Each logged visit belongs to the true visit it overlaps most. With two
    # birds on screen, also prefer the one whose species matches.
    matches = {i: [] for i in range(len(truth))}
    unmatched = 0
    for row in logged:
        s, e = datetime.fromisoformat(row["started_at"]), datetime.fromisoformat(row["ended_at"])
        scored = [(overlap(s, e, t["t_start"], t["t_end"]) + (0.5 if t["species"] == row["final_species"] else 0), i)
                  for i, t in enumerate(truth)]
        best, i = max(scored)
        if best <= 0.5:
            unmatched += 1
        else:
            matches[i].append(row)

    found = sum(1 for m in matches.values() if m)
    split = sum(1 for m in matches.values() if len(m) > 1)
    # Species is judged on the longest logged piece of each true visit.
    correct, confusions = 0, Counter()
    for i, rows in matches.items():
        if not rows:
            continue
        main_row = max(rows, key=lambda r: r["n_frames"])
        if main_row["final_species"] == truth[i]["species"]:
            correct += 1
        else:
            confusions[(truth[i]["species"], main_row["final_species"])] += 1

    n = len(truth)
    print(f"true visits:            {n}")
    print(f"logged visits:          {len(logged)}  (ideal: {n})")
    print(f"visits found:           {found}/{n}  ({found / n:.0%})")
    print(f"counted more than once: {split}  (tracker lost the bird and picked it up as new)")
    print(f"false visits:           {unmatched}  (logged when no bird was there)")
    print(f"species correct:        {correct}/{found}  ({correct / max(1, found):.0%} of found visits)")
    if confusions:
        print("mistakes (true -> predicted):")
        for (t, p), k in confusions.most_common():
            print(f"  {k}  {t} -> {p}")


if __name__ == "__main__":
    main()
