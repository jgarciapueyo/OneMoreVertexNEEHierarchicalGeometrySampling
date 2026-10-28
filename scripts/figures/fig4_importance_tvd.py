"""Total variation distance between the ground-truth and the approximated node
importance per BVH level (right panel of Fig. 4).

Input: bvh_debug.csv written by the SamplingBVH with outputDebugCSV=true, e.g.
    mitsuba scenes/0_importance_quads/importance_quads.xml -DoutputDebugCSV=true -Dspp=1
Usage:
    python3 scripts/figures/fig4_importance_tvd.py bvh_debug.csv [--plot tvd.pdf]

For every level (distance of a node to its deepest leaf, as in the CSV), the
weights of the nodes of that level are normalized into discrete distributions
and compared with TVD = 0.5 * sum_i |p_i - q_i|. Our approximation ignores
visibility, so it is compared with the ground truth with and without it.
"""
import argparse
import csv
from collections import defaultdict

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", help="bvh_debug.csv")
    ap.add_argument("--approx", default="imp_ours", help="column with the approximated weights")
    ap.add_argument("--plot", default=None, help="optional output figure (requires matplotlib)")
    args = ap.parse_args()

    cols = defaultdict(lambda: defaultdict(list))
    with open(args.csv) as f:
        for row in csv.DictReader(f):
            level = int(row["level"])
            for c in ("imp_gt_vis", "imp_gt_novis", args.approx):
                cols[level][c].append(float(row[c]))

    def tvd(p, q):
        p, q = np.asarray(p, float), np.asarray(q, float)
        if p.sum() <= 0 or q.sum() <= 0:
            return float("nan")
        return 0.5 * np.abs(p / p.sum() - q / q.sum()).sum()

    levels = sorted(cols)
    print(f"{'level':>5} {'nodes':>6} {'TVD vs GT (vis)':>16} {'TVD vs GT (no vis)':>19}")
    rows = []
    for lv in levels:
        c = cols[lv]
        a, b = tvd(c["imp_gt_vis"], c[args.approx]), tvd(c["imp_gt_novis"], c[args.approx])
        rows.append((lv, a, b))
        print(f"{lv:>5} {len(c[args.approx]):>6} {a:>16.4f} {b:>19.4f}")

    if args.plot:
        import matplotlib.pyplot as plt
        lv, a, b = zip(*rows)
        plt.figure(figsize=(4, 3))
        plt.plot(lv, a, "o-", label="GT with visibility")
        plt.plot(lv, b, "s--", label="GT without visibility")
        plt.xlabel("BVH level (distance to the deepest leaf)")
        plt.ylabel("total variation distance")
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.plot)
        print(f"Wrote {args.plot}")


if __name__ == "__main__":
    main()
