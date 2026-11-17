"""Check the published CatBoost 94/95 summary and per-horizon aggregates."""

import argparse
import csv
from collections import defaultdict
from math import isclose, isfinite
from pathlib import Path
from statistics import fmean


COHORTS = {"validation": (34256, 1406), "test": (44175, 1255)}
THRESHOLDS = {"기존 94열": 0.04830527678132057, "95열 Optuna": 0.0583612434566021}
SCOPES = {"all_h": (1, 60), "h60": (60, 60), "h42_55": (42, 55)}
COUNTS = ("TP", "FN", "FP", "TN", "N")
RATES = ("Accuracy", "Precision", "Recall", "F1", "F4", "ROC-AUC", "PR-AUC", "AP", "inspection_reduction")


def require(condition, context):
    if not condition:
        raise ValueError(context)


def parse(path, group_fields):
    required = {"split", "model", "threshold", *COUNTS, *RATES, *group_fields}
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        fields = reader.fieldnames or []
        require(len(fields) == len(set(fields)), f"{path.name}: duplicate column")
        require(required.issubset(fields), f"{path.name}: missing columns")
        for line, row in enumerate(reader, start=2):
            context = f"{path.name} line {line}"
            require(None not in row and all(row[key] is not None for key in required), f"{context}: malformed row")
            split, model = row["split"], row["model"]
            require(split in COHORTS and model in THRESHOLDS, f"{context}: unknown split or model")
            require(float(row["threshold"]) == THRESHOLDS[model], f"{context}: threshold differs")
            counts = {key: int(row[key]) for key in COUNTS}
            rates = {key: float(row[key]) for key in RATES}
            require(all(value >= 0 for value in counts.values()), f"{context}: negative count")
            require(all(isfinite(value) and 0 <= value <= 1 for value in rates.values()), f"{context}: invalid metric")
            yield context, row, counts, rates


def check_population(counts, split, number_of_horizons, context):
    n, positives = COHORTS[split]
    require(counts["N"] == n * number_of_horizons, f"{context}: N differs")
    require(counts["TP"] + counts["FN"] == positives * number_of_horizons, f"{context}: positive count differs")
    require(counts["FP"] + counts["TN"] == (n - positives) * number_of_horizons, f"{context}: negative count differs")


def confusion_rates(counts):
    tp, fn, fp, tn, n = (counts[key] for key in COUNTS)
    return {
        "Accuracy": (tp + tn) / n,
        "Precision": tp / (tp + fp) if tp + fp else 0.0,
        "Recall": tp / (tp + fn),
        "F1": 2 * tp / (2 * tp + fn + fp),
        "F4": 17 * tp / (17 * tp + 16 * fn + fp),
        "inspection_reduction": tn / n,
    }


def check_rates(rates, expected, context):
    for name, value in expected.items():
        require(isclose(rates[name], value, rel_tol=0, abs_tol=1e-12), f"{context}: {name} differs")


def verify_results(directory):
    groups = defaultdict(dict)
    for context, row, counts, rates in parse(directory / "per_horizon.csv", ("H",)):
        split, model, horizon = row["split"], row["model"], int(row["H"])
        group = groups[split, model]
        require(1 <= horizon <= 60 and horizon not in group, f"{context}: invalid or duplicate H")
        check_population(counts, split, 1, context)
        check_rates(rates, confusion_rates(counts), context)
        group[horizon] = (counts, rates)

    expected_groups = {(split, model) for split in COHORTS for model in THRESHOLDS}
    require(set(groups) == expected_groups, "per_horizon.csv: missing split/model group")
    for (split, model), group in groups.items():
        require(len(group) == 60, f"per_horizon.csv: incomplete H coverage for {split}/{model}")

    seen = set()
    for context, row, counts, rates in parse(directory / "summary.csv", ("scope", "specimens", "positive_specimens")):
        split, model, scope = row["split"], row["model"], row["scope"]
        key = (split, model, scope)
        require(scope in SCOPES and key not in seen, f"{context}: invalid or duplicate scope")
        seen.add(key)
        start, stop = SCOPES[scope]
        require((int(row["specimens"]), int(row["positive_specimens"])) == COHORTS[split], f"{context}: specimen count differs")
        check_population(counts, split, stop - start + 1, context)
        parts = [groups[split, model][h] for h in range(start, stop + 1)]
        require(counts == {key: sum(c[key] for c, _ in parts) for key in COUNTS}, f"{context}: totals differ from H rows")
        if scope == "all_h":
            check_rates(rates, confusion_rates(counts), context)
        else:
            check_rates(rates, {key: fmean(r[key] for _, r in parts) for key in RATES}, context)

    require(seen == {(split, model, scope) for split, model in groups for scope in SCOPES}, "summary.csv: missing scope")
    return len(seen), sum(len(group) for group in groups.values())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path, help="directory with summary.csv and per_horizon.csv")
    args = parser.parse_args(argv)
    try:
        summary_rows, horizon_rows = verify_results(args.results_dir)
    except (OSError, ValueError, csv.Error) as exc:
        parser.exit(1, f"Result verification failed: {exc}\n")
    print(f"Verified {summary_rows} summary rows and {horizon_rows} horizon rows.")


if __name__ == "__main__":
    main()
