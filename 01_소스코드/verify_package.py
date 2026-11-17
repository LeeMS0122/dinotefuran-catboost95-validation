"""Verify the public package and optionally reproduce the synthetic sample with a local model."""

import argparse
import csv
from datetime import date
import hashlib
import json
from math import isfinite
from pathlib import Path, PurePosixPath
import re
import struct
import subprocess
import sys
from tempfile import TemporaryDirectory

from final_model95.verify_results import THRESHOLDS, verify_results


SOURCE_DIR = Path(__file__).resolve().parent
ROOT = SOURCE_DIR.parent
MANIFEST = "09_검증대상버전_파일목록/파일목록_SHA256.csv"
SETTINGS = "02_학습모델파일/model_settings.json"
RESULTS = "05_성능검증_상세자료"
DEMO_INPUT = "06_입출력정의_샘플/prepared_features_example.csv"
DEMO_OUTPUT = "06_입출력정의_샘플/example_output.csv"
REQUIRED_FILES = {
    "01_소스코드/verify_package.py",
    "01_소스코드/final_model95/__init__.py",
    "01_소스코드/final_model95/__main__.py",
    "01_소스코드/final_model95/verify_results.py",
    SETTINGS,
    f"{RESULTS}/summary.csv",
    f"{RESULTS}/per_horizon.csv",
    DEMO_INPUT,
    DEMO_OUTPUT,
}
METADATA = ["sample_id", "case_id", "scope", "D", "M", "H"]
PREDICTION_FIELDS = ["probability", "threshold", "prediction"]
PROBABILITY_TOLERANCE = 1e-6


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def valid_sha256(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def digest(path: Path) -> str:
    sha256 = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            sha256.update(block)
    return sha256.hexdigest()


def verify_manifest(root: Path) -> int:
    names: set[str] = set()
    keys: set[str] = set()
    with (root / MANIFEST).open(encoding="utf-8-sig", newline="") as source:
        reader = csv.reader(source, strict=True)
        require(next(reader, []) == ["file", "bytes", "sha256"], "Unexpected SHA-256 manifest header.")
        for row in reader:
            require(len(row) == 3, f"Malformed manifest row {reader.line_num}.")
            name, size, sha256 = row
            relative = PurePosixPath(name)
            require(
                bool(relative.parts) and not relative.is_absolute() and relative.as_posix() == name
                and ".." not in relative.parts and "\\" not in name and ":" not in name,
                f"Invalid relative path in manifest row {reader.line_num}.",
            )
            key = name.casefold()
            require(key != MANIFEST.casefold() and key not in keys, f"Repeated or self-referential manifest entry: {name}")
            names.add(name)
            keys.add(key)
            path = root.joinpath(*relative.parts).resolve()
            require(path.is_relative_to(root) and path.is_file(), f"Missing or out-of-package file: {name}")
            require(size.isascii() and size.isdecimal() and path.stat().st_size == int(size), f"Size mismatch: {name}")
            require(valid_sha256(sha256) and digest(path) == sha256, f"SHA-256 mismatch: {name}")
    require(REQUIRED_FILES <= names, f"Manifest is missing required files: {sorted(REQUIRED_FILES - names)}")
    return len(names)


def verify_settings(path: Path) -> dict:
    settings = json.loads(path.read_text(encoding="utf-8-sig"))
    require(isinstance(settings, dict), "Model settings must be a JSON object.")
    features = settings.get("feature_order")
    require(
        isinstance(features, list) and len(features) == 95
        and all(isinstance(feature, str) and feature.strip() for feature in features)
        and len(set(features)) == 95 and not set(features).intersection(METADATA + PREDICTION_FIELDS),
        "The locked schema must contain 95 unique model fields.",
    )
    require(settings.get("categorical_indices_zero_based") == [56, 57, 58, 59], "Categorical schema mismatch.")
    require(settings.get("window_days") == 90 and settings.get("horizons") == list(range(1, 61)), "Window or H mismatch.")
    require(settings.get("threshold") == THRESHOLDS["95열 Optuna"], "Settings and aggregated threshold differ.")
    require(settings.get("decision_rule") == "probability >= threshold", "Unexpected decision rule.")
    require(
        settings.get("threshold_selected_on") == "validation"
        and settings.get("test_used_for_model_or_threshold_selection") is False,
        "The model selection settings must record the locked validation selection.",
    )
    require(settings.get("best_trial") == 3 and settings.get("trees") == 138, "Locked trial or tree count mismatch.")
    require(valid_sha256(settings.get("model_sha256")), "Invalid model SHA-256 in settings.")
    return settings


def read_csv(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.reader(source, strict=True)
        header = next(reader, [])
        rows = list(reader)
    require(bool(header) and all(len(row) == len(header) for row in rows), f"Malformed CSV: {path.name}")
    return header, rows


def float32(value: float) -> float:
    return struct.unpack("f", struct.pack("f", value))[0]


def verify_sample_input(path: Path, settings: dict) -> tuple[list[str], list[list[str]]]:
    header, rows = read_csv(path)
    require(header == METADATA + settings["feature_order"], "Synthetic input headers differ from the locked schema.")
    require([row[5] for row in rows] == ["1", "30", "60"], "Synthetic input must contain H=1, 30, 60 in order.")
    require(len({row[0] for row in rows}) == len(rows), "Synthetic sample IDs must be unique.")
    for row in rows:
        sample_id, case_id, scope, origin, target, horizon = row[:len(METADATA)]
        require(
            sample_id.startswith("DEMO-") and case_id.startswith("SYNTHETIC-") and scope == "synthetic_smoke",
            "Unexpected metadata in the synthetic sample.",
        )
        require((date.fromisoformat(target) - date.fromisoformat(origin)).days == int(horizon), "Synthetic D/M/H mismatch.")
        features = [float(value) for value in row[len(METADATA):]]
        rounded_features = [float32(value) for value in features]
        require(all(isfinite(value) for value in rounded_features), "Invalid numeric feature in synthetic sample.")
        require(
            all(0 <= features[i] < 2**31 and features[i].is_integer() and features[i] == rounded_features[i]
                for i in settings["categorical_indices_zero_based"]),
            "Invalid categorical code in synthetic sample.",
        )
    return header, rows


def verify_sample_output(path: Path, sample: tuple[list[str], list[list[str]]], threshold: float) -> list[float]:
    input_header, input_rows = sample
    output_header, output_rows = read_csv(path)
    require(output_header == input_header + PREDICTION_FIELDS, f"Unexpected output header: {path.name}")
    require([row[:-3] for row in output_rows] == input_rows, f"Input cells or row order changed: {path.name}")
    scores = []
    for row in output_rows:
        raw_score, raw_threshold, decision = row[-3:]
        score = float(raw_score)
        require(0 <= score <= 1, f"Invalid score for {row[0]}: {path.name}")
        require(float(raw_threshold) == threshold, f"Threshold changed for {row[0]}: {path.name}")
        require(decision == str(int(score >= float32(threshold))), f"Decision differs for {row[0]}: {path.name}")
        scores.append(score)
    return scores


def verify_model(model: Path, sample: tuple[list[str], list[list[str]]], settings: dict, expected: list[float]) -> float:
    with TemporaryDirectory(prefix="dinotefuran_verify_") as temporary:
        output = Path(temporary) / "example_output.csv"
        command = [sys.executable, "-B", "-m", "final_model95", "predict", "--model", str(model.resolve()),
                   "--input", str(ROOT / DEMO_INPUT), "--output", str(output)]
        subprocess.run(command, cwd=SOURCE_DIR, check=True)
        scores = verify_sample_output(output, sample, settings["threshold"])
        errors = [abs(actual - reference) for actual, reference in zip(scores, expected, strict=True)]
        maximum = max(errors)
        require(maximum <= PROBABILITY_TOLERANCE, f"Native probability differs by {maximum:.8g}.")
        cutoff = float32(settings["threshold"])
        require([score >= cutoff for score in scores] == [score >= cutoff for score in expected], "Native decisions differ.")
        return maximum


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, help="local copy of the locked native model.cbm")
    parser.add_argument("--report-dir", type=Path, help="optional destination for 검증_결과.json")
    args = parser.parse_args(argv)
    report = {"passed": False, "native_model": {"status": "pending" if args.model else "skipped"}}
    try:
        report["manifest_files"] = verify_manifest(ROOT)
        settings = verify_settings(ROOT / SETTINGS)
        report["feature_count"] = len(settings["feature_order"])
        report["threshold"] = settings["threshold"]
        report["summary_rows"], report["horizon_rows"] = verify_results(ROOT / RESULTS)
        sample = verify_sample_input(ROOT / DEMO_INPUT, settings)
        expected = verify_sample_output(ROOT / DEMO_OUTPUT, sample, settings["threshold"])
        report["synthetic_rows"] = len(expected)
        print(f"Manifest: {report['manifest_files']} files verified.", flush=True)
        print(f"Aggregates: {report['summary_rows']} summary rows, {report['horizon_rows']} H rows.", flush=True)
        print(f"Synthetic contract: {report['feature_count']} model fields, {report['synthetic_rows']} examples.", flush=True)
        if args.model:
            report["native_model"]["status"] = "running"
            maximum = verify_model(args.model, sample, settings, expected)
            report["native_model"] = {"status": "passed", "sha256": settings["model_sha256"],
                                      "probability_tolerance": PROBABILITY_TOLERANCE, "max_abs_error": maximum}
            print(f"Native synthetic prediction: rows and decisions match, maximum probability error {maximum:.8g}.")
        else:
            print("Native synthetic prediction: skipped (no --model supplied).")
        report["passed"] = True
    except (OSError, ValueError, OverflowError, csv.Error, subprocess.CalledProcessError) as exc:
        message = f"Native inference exited with status {exc.returncode}." if isinstance(exc, subprocess.CalledProcessError) else str(exc)
        if report["native_model"]["status"] == "running":
            report["native_model"]["status"] = "failed"
        report["error"] = message
        print(f"Package verification failed: {message}", file=sys.stderr)

    if args.report_dir:
        try:
            args.report_dir.mkdir(parents=True, exist_ok=True)
            report_path = args.report_dir / "검증_결과.json"
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
            print(f"Audit report written to {report_path}.")
        except OSError as exc:
            print(f"Audit report could not be written: {exc}", file=sys.stderr)
            return 1
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
