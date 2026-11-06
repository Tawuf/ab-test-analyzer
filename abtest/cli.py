"""Command-line interface: ``python -m abtest <command>``.

  analyze   analyse an experiment CSV and write a Markdown report + chart
  plan      sample size / test duration for a new experiment
  peeking   simulate how 'peeking' inflates false positives
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from . import stats as st  # noqa: E402
from .simulate import peeking_false_positive_rate  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def _pct(x: float, digits: int = 2) -> str:
    return f"{100 * x:.{digits}f} %"


def analyze(csv_path: Path, group: str, metric: str, revenue: str | None, segment: str | None,
            control: str, alpha: float, out_dir: Path) -> str:
    df = pd.read_csv(csv_path)
    groups = sorted(df[group].unique())
    if len(groups) != 2 or control not in groups:
        raise SystemExit(f"Need exactly two groups including control '{control}', found {groups}")
    treatment = next(g for g in groups if g != control)
    a, b = df[df[group] == control], df[df[group] == treatment]

    lines = [f"# A/B test report: `{csv_path.name}`", ""]

    srm_p = st.srm_check(len(a), len(b))
    lines += ["## 1. Sanity check: sample ratio", "",
              f"{control}: {len(a):,} users, {treatment}: {len(b):,} users. SRM chi-square p = {srm_p:.3f}. "
              + ("**Traffic split looks healthy.**" if srm_p >= 0.001 else
                 "**Warning: possible sample ratio mismatch. Check the assignment before trusting results.**"),
              ""]

    z = st.two_proportion_ztest(int(a[metric].sum()), len(a), int(b[metric].sum()), len(b), alpha)
    bayes = st.bayesian_proportions(int(a[metric].sum()), len(a), int(b[metric].sum()), len(b))
    lines += [f"## 2. Primary metric: `{metric}` rate", "",
              "| | Control | Treatment |", "|---|---:|---:|",
              f"| Group | {control} | {treatment} |",
              f"| Rate | {_pct(z.rate_a)} | {_pct(z.rate_b)} |", "",
              f"* Absolute difference: **{100 * z.abs_diff:+.2f} percentage points** "
              f"({100 * (1 - alpha):.0f} % CI {100 * z.ci_low:+.2f} to {100 * z.ci_high:+.2f})",
              f"* Relative lift: **{100 * z.rel_lift:+.1f} %**",
              f"* z = {z.z:.2f}, p = {z.p_value:.4f}, so the result is "
              + ("**statistically significant**" if z.significant else "**not statistically significant**")
              + f" at α = {alpha}",
              f"* Bayesian view: P({treatment} better than {control}) = **{_pct(bayes.prob_b_better, 1)}**; "
              f"95 % credible interval for relative lift {100 * bayes.lift_ci_low:+.1f} % to "
              f"{100 * bayes.lift_ci_high:+.1f} %; expected loss if we ship {treatment}: "
              f"{100 * bayes.expected_loss_choose_b:.4f} pp", ""]

    if revenue:
        t = st.welch_ttest(a[revenue], b[revenue], alpha)
        lines += [f"## 3. Secondary metric: `{revenue}` per user (Welch t-test)", "",
                  f"* Mean {control}: {t.mean_a:.2f}, mean {treatment}: {t.mean_b:.2f}",
                  f"* Difference {t.diff:+.2f} ({100 * (1 - alpha):.0f} % CI {t.ci_low:+.2f} to {t.ci_high:+.2f}), "
                  f"p = {t.p_value:.4f}", ""]

    if segment:
        lines += [f"## 4. Segments by `{segment}` (exploratory)", "",
                  "| Segment | Control rate | Treatment rate | Lift | p-value |", "|---|---:|---:|---:|---:|"]
        for seg, sub in df.groupby(segment):
            sa, sb = sub[sub[group] == control], sub[sub[group] == treatment]
            r = st.two_proportion_ztest(int(sa[metric].sum()), len(sa), int(sb[metric].sum()), len(sb), alpha)
            lines.append(f"| {seg} | {_pct(r.rate_a)} | {_pct(r.rate_b)} | {100 * r.rel_lift:+.1f} % | {r.p_value:.3f} |")
        lines += ["", "*Segment results are exploratory. Testing many segments raises the chance of a "
                  "false positive, so confirm any segment finding in a new experiment.*", ""]

    # posterior chart
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for samples, label, color in [(bayes.samples_a, control, "#7f8c8d"), (bayes.samples_b, treatment, "#2980b9")]:
        ax.hist(samples * 100, bins=150, density=True, alpha=0.55, color=color, label=f"Variant {label}")
    ax.set_xlabel(f"{metric} rate (%)")
    ax.set_yticks([])
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_title(f"Posterior distribution of the conversion rate: P({treatment} > {control}) = "
                 f"{_pct(bayes.prob_b_better, 1)}", loc="left", fontsize=11, fontweight="bold")
    ax.legend(frameon=False)
    fig.tight_layout()
    chart = out_dir / "posterior.png"
    fig.savefig(chart, dpi=150)
    plt.close(fig)
    lines += [f"![Posterior]({chart.name})", ""]

    decision = ("Ship the treatment." if z.significant and z.abs_diff > 0 else
                "Keep the control." if z.significant else
                "No clear winner. Keep the control or run the test longer if the expected effect is smaller than the test can detect.")
    lines += ["## Recommendation", "", f"**{decision}**", ""]

    report = "\n".join(lines)
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    return report


def plan(baseline: float, mde: float, relative: bool, alpha: float, power: float, daily: int | None) -> None:
    n = st.sample_size_proportions(baseline, mde, alpha, power, relative)
    target = baseline * (1 + mde) if relative else baseline + mde
    print(f"Baseline rate {_pct(baseline)} -> target {_pct(target)}  (alpha {alpha}, power {power:.0%})")
    print(f"Users needed per group: {n:,}   total: {2 * n:,}")
    if daily:
        print(f"At {daily:,} users/day across both groups: about {math.ceil(2 * n / daily)} days")


def peeking(out_dir: Path) -> None:
    res = peeking_false_positive_rate()
    print(f"False positive rate, one look at the end: {_pct(res['fixed_horizon_fpr'], 1)}")
    print(f"False positive rate, peeking daily ({res['days']} days): {_pct(res['peeking_fpr'], 1)}")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    days = range(1, res["days"] + 1)
    ax.plot(days, [100 * v for v in res["peeking_fpr_by_day"]], marker="o", color="#c0392b", lw=2,
            label="Stop as soon as p < 0.05 (peeking)")
    ax.axhline(100 * res["alpha"], color="#27ae60", lw=2, ls="--", label="Promised false positive rate (5 %)")
    ax.set_xticks(list(days))
    ax.set_xlabel("Days the experiment has run")
    ax.set_ylabel("Share of A/A tests declared 'significant' (%)")
    ax.set_title("Peeking at results inflates false positives", loc="left", fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "peeking.png", dpi=150)
    plt.close(fig)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="abtest", description="A/B test analysis toolkit")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("analyze", help="analyse an experiment CSV")
    p.add_argument("csv", type=Path)
    p.add_argument("--group", default="variant")
    p.add_argument("--metric", default="converted", help="0/1 column")
    p.add_argument("--revenue", help="optional continuous column, e.g. revenue_sek")
    p.add_argument("--segment", help="optional column for segment breakdown")
    p.add_argument("--control", default="A")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--out", type=Path, default=ROOT / "docs")

    p = sub.add_parser("plan", help="sample size for a new experiment")
    p.add_argument("--baseline", type=float, required=True, help="current rate, e.g. 0.05")
    p.add_argument("--mde", type=float, required=True, help="minimum detectable effect")
    p.add_argument("--relative", action="store_true", help="treat --mde as relative (0.1 = +10 %%)")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--power", type=float, default=0.80)
    p.add_argument("--daily", type=int, help="users per day, to estimate duration")

    p = sub.add_parser("peeking", help="simulate the peeking problem")
    p.add_argument("--out", type=Path, default=ROOT / "docs")

    args = ap.parse_args(argv)
    if args.cmd == "analyze":
        print(analyze(args.csv, args.group, args.metric, args.revenue, args.segment,
                      args.control, args.alpha, args.out))
    elif args.cmd == "plan":
        plan(args.baseline, args.mde, args.relative, args.alpha, args.power, args.daily)
    else:
        peeking(args.out)


if __name__ == "__main__":
    main()
