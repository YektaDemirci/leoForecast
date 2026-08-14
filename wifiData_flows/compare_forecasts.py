import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from deseason import deseasonalise_parts        # noqa: E402
from score_wifi import load_wifi                # noqa: E402

TRAIN_FRAC = 0.70
RESERVED = ("idx", "date", "truth", "season")

TOL = 1e-4


def read_dump(spec):
    """Parse a `[LABEL=]PATH` dump spec into (label, DataFrame)."""
    label, _, path = spec.rpartition("=")
    if not os.path.exists(path):
        raise SystemExit(f"no such dump: {path}")
    d = pd.read_csv(path)
    missing = [c for c in ("idx", "truth") if c not in d.columns]
    if missing:
        raise SystemExit(f"{path} is missing required column(s) {missing}; see "
                         f"the dump schema in this file's docstring")
    d = d.sort_values("idx").drop_duplicates("idx").reset_index(drop=True)
    methods = [c for c in d.columns if c not in RESERVED]
    if not methods:
        raise SystemExit(f"{path} carries no forecaster columns")
    if label:
        d = d.rename(columns={m: f"{m}@{label}" for m in methods})
    return path, label, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", action="append", required=True, metavar="[LABEL=]PATH",
                    help="per-forecast dump; repeat once per run")
    ap.add_argument("--csv", default=os.path.join(HERE, "wifiData.csv"))
    ap.add_argument("--horizon", type=int, default=1,
                    help="forecast horizon the dumps were produced at. Only "
                         "affects the `season` check: at h > 1 a forecast "
                         "targets the SUM over h buckets, so the component "
                         "added back is the sum of comp over the same window. "
                         "All dumps compared in one call must share it.")
    ap.add_argument("--out", default=None,
                    help="write the table here as well as printing it")
    ap.add_argument("--allow-unverified", action="store_true",
                    help="score dumps that carry no `season` column instead of "
                         "refusing them. The re-seasonalisation is then "
                         "unchecked -- a stale deseasonalised csv will not be "
                         "caught, so the deseasonalised rows may silently "
                         "describe a different transform than the analytic ones")
    args = ap.parse_args()

    df = load_wifi(args.csv)
    n = len(df)
    x = df["OT"].to_numpy(dtype=float)
    peak = np.abs(x).max()
    # The reference component, from the same causal transform score_wifi.py and
    # plot_deseasonalizeWifi.py use. Every dump's `season` is checked against
    # this, so all re-seasonalisations are verified against ONE definition
    # rather than against whichever csv happened to be sitting in each repo.
    _des, _x, comp = deseasonalise_parts(df, int(n * TRAIN_FRAC))
    _cc = np.concatenate(([0.0], np.cumsum(comp)))

    dumps = [read_dump(s) for s in args.dump]

    # Intersect on idx: the runs cover different windows (Dataset_Custom starts
    # its test split at n - num_test - seq_len, run_one at 0.8*n + T), and a
    # metric is only comparable across them on the rows they all target.
    common = None
    for _p, _l, d in dumps:
        s = set(d.idx.astype(int))
        common = s if common is None else (common & s)
    common = np.array(sorted(common), dtype=int)
    if common.size < 2:
        raise SystemExit("the dumps share fewer than 2 buckets -- nothing to score")

    # Same window sum raw_space_nmse scores against: at h = 1 this is x[common].
    _cx = np.concatenate(([0.0], np.cumsum(x)))
    truth = _cx[common + args.horizon] - _cx[common]
    scale = float(np.mean(np.abs(np.diff(truth))))

    print(f"{args.csv}: {n} rows")
    print(f"scored on the INTERSECTION of {len(dumps)} dump(s): "
          f"{common.size} buckets [{common[0]}, {common[-1]}]  "
          f"{df.date.iloc[common[0]]} .. {df.date.iloc[common[-1]]}")
    print(f"naive scale (mean |y_t - y_t-1| on that window) = {scale:.6e}\n")

    rows, unverified = [], []
    for path, _label, d in dumps:
        d = d[d.idx.astype(int).isin(common)].set_index("idx").loc[common]

        # Alignment: the dump's own truth must be the raw series at idx. This
        # is what catches an off-by-seq_len index, the one error a forecast
        # dump cannot otherwise reveal.
        terr = float(np.abs(d["truth"].to_numpy(float) - truth).max()) / peak
        if terr > TOL:
            raise SystemExit(
                f"{path}: `truth` does not match {args.csv} at the given idx "
                f"(relative {terr:.2g} > {TOL:g}). The dump's idx is off, or it "
                f"was written against a different csv.")

        if "season" in d.columns:
            got = d["season"].to_numpy(dtype=float)
            ref = _cc[common + args.horizon] - _cc[common]
            zero = float(np.abs(got).max()) / peak <= TOL

            off = float(np.mean(got - ref))
            serr = float(np.abs((got - ref) - off).max()) / peak
            if not zero and serr > TOL:
                raise SystemExit(
                    f"{path}: the `season` it added back is neither zero (a raw "
                    f"run) nor deseason.py's component -- after allowing for a "
                    f"constant offset it still differs in shape by "
                    f"{100 * serr:.2f}% of peak traffic. Regenerate the "
                    f"deseasonalised csv with plot_deseasonalizeWifi.py and "
                    f"re-run that forecaster.")
        else:
            unverified.append(path)

        for m in [c for c in d.columns if c not in RESERVED]:
            p = d[m].to_numpy(dtype=float)
            e = truth - p
            rows.append(dict(forecaster=m,
                             mse=float(np.mean(e ** 2)),
                             mase=float(np.mean(np.abs(e)) / scale)))

    if unverified and not args.allow_unverified:
        raise SystemExit(
            "these dumps carry no `season` column, so their re-seasonalisation "
            "cannot be verified:\n  " + "\n  ".join(unverified) +
            "\nAdd the column (0.0 for a raw-series run), or pass "
            "--allow-unverified to score them anyway.")

    out = pd.DataFrame(rows).sort_values("mase").reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))

    hdr = f"{'RANK':>4}  {'FORECASTER':<28} {'MSE':>13} {'MASE':>9}"
    lines = [f"Scored on {common.size} shared buckets [{common[0]}, {common[-1]}], "
             f"raw OT ground truth, naive scale {scale:.6e}",
             "MASE 1.0 is one-step persistence on this window.", "", hdr,
             "-" * len(hdr)]
    for r in out.itertuples():
        lines.append(f"{r.rank:>4}  {r.forecaster:<28} {r.mse:>13.4e} {r.mase:>9.4f}")
    if unverified:
        lines += ["", "UNVERIFIED (no `season` column, re-seasonalisation unchecked):"]
        lines += [f"  {p}" for p in unverified]
    text = "\n".join(lines)
    print(text)

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            fh.write(text + "\n")
        out.to_csv(os.path.splitext(args.out)[0] + ".csv", index=False)
        print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
