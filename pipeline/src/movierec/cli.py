"""Command line entry point: ``uv run movierec <command>``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from movierec import config, data, export, pipeline
from movierec.scoring import recommend

MODEL_LABELS = {
    "popularity": "Popularity (baseline)",
    "content": "Content (embeddings)",
    "collab": "Item-kNN (collaborative)",
    "hybrid": "Hybrid",
}
PROFILE_LABELS = {
    "recent-5": "Query = each user's 5 most recent ratings",
    "full": "Query = all of the user's training ratings",
}


def results_table(results: dict[str, Any]) -> str:
    """Markdown tables for the README, generated from eval/results.json."""
    k = results["k"]
    lines: list[str] = []
    for profile, by_model in results["profiles"].items():
        baseline = by_model["popularity"]
        lines += [
            f"**{PROFILE_LABELS.get(profile, profile)}** ({results['users']} users)",
            "",
            f"| Model | HitRate@{k} (95% CI) | NDCG@{k} | NDCG vs. popularity "
            f"| Catalog coverage | Avg. popularity of recs | Long-tail share |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for model, m in by_model.items():
            lift = m["ndcg"] / baseline["ndcg"] if baseline["ndcg"] else float("nan")
            low, high = m["hit_rate_ci"]
            lines.append(
                f"| {MODEL_LABELS.get(model, model)} | {m['hit_rate']:.3f} ({low:.3f}-{high:.3f}) "
                f"| {m['ndcg']:.3f} | {lift:.2f}x | {m['coverage']:.1%} "
                f"| {m['avg_popularity']:.0f} | {m['long_tail_share']:.1%} |"
            )
        lines.append("")
    cov = results["query_coverage"]
    lines.append(
        "Movies that return a full list when picked on their own: "
        + ", ".join(f"{MODEL_LABELS[m].split(' (')[0].lower()} {v:.1%}" for m, v in cov.items())
        + "."
    )
    return "\n".join(lines)


def _cmd_download(args: argparse.Namespace) -> int:
    path = data.ensure_dataset(args.data_dir)
    print(f"dataset verified and extracted to {path}")
    return 0


def _cmd_build(args: argparse.Namespace) -> int:
    ds = data.load(data.ensure_dataset(args.data_dir))
    results, full, _ = pipeline.run(ds, args.data_dir / "cache")
    for path in export.write_all(full.catalog, full.scorer, results):
        print(f"wrote {path.relative_to(config.REPO_ROOT)} ({path.stat().st_size / 1024:.0f} KiB)")
    print()
    print(results_table(results))
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    results = json.loads(args.results.read_text(encoding="utf-8"))
    print(results_table(results))
    return 0


def _cmd_recommend(args: argparse.Namespace) -> int:
    if not args.results.exists():
        print("run `movierec build` first; recommend reuses its tuned settings", file=sys.stderr)
        return 1
    ds = data.load(data.ensure_dataset(args.data_dir))
    settings = json.loads(args.results.read_text(encoding="utf-8"))["settings"]
    full = pipeline.fit_serving(ds, args.data_dir / "cache", settings)
    catalog = full.catalog
    liked = []
    for query in args.titles:
        hits = catalog.find(query)
        if not hits:
            print(f"no movie matches {query!r}", file=sys.stderr)
            return 1
        liked.append(hits[0])
    print("Liked: " + "; ".join(catalog.label(r) for r in liked))
    for rec in recommend(full.scorer, catalog, liked, args.model):
        why = ", ".join(rec.shared_genres + rec.shared_tags) or "co-ratings"
        print(
            f"  {catalog.label(rec.row):<55} {rec.score:6.3f}  "
            f"because you liked {catalog.label(rec.because)} [{why}]; "
            f"{rec.collab_share:.0%} collaborative"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="movierec", description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=config.DEFAULT_DATA_DIR)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("download", help="fetch and verify MovieLens ml-latest-small").set_defaults(
        func=_cmd_download
    )
    sub.add_parser(
        "build", help="tune, evaluate, refit on all data and export web/public/data"
    ).set_defaults(func=_cmd_build)

    report = sub.add_parser("report", help="print the README results table")
    report.add_argument("--results", type=Path, default=config.RESULTS_PATH)
    report.set_defaults(func=_cmd_report)

    rec = sub.add_parser("recommend", help="recommend from one or more liked titles")
    rec.add_argument("titles", nargs="+", help="title fragments, e.g. 'pulp fiction'")
    rec.add_argument("--model", choices=["content", "collab", "hybrid"], default="hybrid")
    rec.add_argument("--results", type=Path, default=config.RESULTS_PATH)
    rec.set_defaults(func=_cmd_recommend)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
