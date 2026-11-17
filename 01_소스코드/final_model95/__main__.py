"""Score prepared model inputs and retain their original CSV columns."""

import argparse
import csv
from dataclasses import dataclass
import hashlib
from itertools import islice
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostError, Pool


SETTINGS_PATH = Path(__file__).resolve().parents[2] / "02_학습모델파일" / "model_settings.json"
BATCH_SIZE = 32768
THREAD_COUNT = 2
RESULT_COLUMNS = ("probability", "threshold", "prediction")


@dataclass(frozen=True)
class ModelSettings:
    features: list[str]
    categorical_indices: list[int]
    threshold: float
    trees: int
    model_sha256: str


def read_settings(path: Path) -> ModelSettings:
    config = json.loads(path.read_text(encoding="utf-8-sig"))
    required_keys = {
        "feature_order",
        "categorical_indices_zero_based",
        "threshold",
        "trees",
        "model_sha256",
    }
    if not isinstance(config, dict) or not required_keys.issubset(config):
        raise ValueError("The bundled settings are missing required model fields.")

    features = config["feature_order"]
    if (
        not isinstance(features, list)
        or len(features) != 95
        or not all(isinstance(name, str) and name for name in features)
        or len(set(features)) != len(features)
    ):
        raise ValueError("The bundled settings must define 95 unique feature names.")

    categorical_indices = config["categorical_indices_zero_based"]
    if categorical_indices != [56, 57, 58, 59] or any(type(index) is not int for index in categorical_indices):
        raise ValueError("The bundled settings must identify the four locked categorical columns.")

    threshold = config["threshold"]
    if type(threshold) not in (float, int) or not 0 <= threshold <= 1:
        raise ValueError("The bundled threshold must be between zero and one.")

    trees = config["trees"]
    if type(trees) is not int or trees < 1:
        raise ValueError("The bundled settings must specify a positive tree count.")

    model_sha256 = config["model_sha256"]
    if (
        not isinstance(model_sha256, str)
        or len(model_sha256) != 64
        or any(character not in "0123456789abcdef" for character in model_sha256)
    ):
        raise ValueError("The bundled settings must specify a lowercase SHA-256 model digest.")

    return ModelSettings(features, categorical_indices, float(threshold), trees, model_sha256)


def load_model(path: Path, settings: ModelSettings) -> CatBoostClassifier:
    model_bytes = path.read_bytes()
    digest = hashlib.sha256(model_bytes).hexdigest()
    if digest != settings.model_sha256:
        raise ValueError("The model file SHA-256 differs from the locked 95-column release.")

    model = CatBoostClassifier()
    model.load_model(blob=model_bytes)
    if model.feature_names_ != settings.features:
        raise ValueError("The native model feature names or order differ from the bundled settings.")
    if model.get_cat_feature_indices() != settings.categorical_indices:
        raise ValueError("The native model categorical indices differ from the bundled settings.")
    if model.tree_count_ != settings.trees:
        raise ValueError("The native model tree count differs from the bundled settings.")
    if not np.array_equal(model.classes_, [0, 1]):
        raise ValueError("The native model must provide probabilities in class order 0, 1.")
    return model


def feature_positions(header: list[str], settings: ModelSettings) -> list[int]:
    if not header or any(not name.strip() for name in header) or len(set(header)) != len(header):
        raise ValueError("The input CSV header must contain unique, nonempty column names.")

    collisions = [column for column in RESULT_COLUMNS if column in header]
    if collisions:
        raise ValueError(f"Input columns conflict with result columns: {', '.join(collisions)}")

    positions = {column: index for index, column in enumerate(header)}
    missing = [name for name in settings.features if name not in positions]
    if missing:
        raise ValueError(f"Missing required model features: {', '.join(missing)}")
    return [positions[name] for name in settings.features]


def feature_frame(
    rows: list[list[str]],
    positions: list[int],
    settings: ModelSettings,
    location: str,
) -> pd.DataFrame:
    try:
        values = np.asarray([[row[index] for index in positions] for row in rows], dtype=np.float64)
    except ValueError as exc:
        raise ValueError(f"{location} contain a missing or non-numeric model feature.") from exc

    with np.errstate(over="ignore", invalid="ignore"):
        numeric = values.astype(np.float32)
    if not np.isfinite(numeric).all():
        raise ValueError(f"{location} contain a feature outside the finite float32 range.")

    frame = pd.DataFrame(numeric, columns=settings.features)
    for index in settings.categorical_indices:
        codes = values[:, index]
        column = settings.features[index]
        if ((codes < 0) | (codes >= 2**31) | (codes != np.floor(codes))).any() or not np.array_equal(
            codes, numeric[:, index]
        ):
            raise ValueError(f"{location} require nonnegative integer codes with exact float32 representation in {column}.")
        frame[column] = codes.astype(np.int32).astype(str)
    return frame


def score(
    model: CatBoostClassifier,
    frame: pd.DataFrame,
    settings: ModelSettings,
) -> tuple[np.ndarray, np.ndarray]:
    pool = Pool(frame, cat_features=settings.categorical_indices, thread_count=THREAD_COUNT)
    probabilities = model.predict_proba(pool, thread_count=THREAD_COUNT)
    if probabilities.shape != (len(frame), 2):
        raise ValueError("Unexpected shape for the model's class probabilities.")
    scores = probabilities[:, 1].astype(np.float32)
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("The model returned an invalid probability.")
    decisions = (scores >= settings.threshold).astype(np.int8)
    return scores, decisions


def require_distinct_output(output: Path, inputs: tuple[Path, ...]) -> None:
    for source in inputs:
        if output.resolve() == source.resolve() or (
            output.exists() and source.exists() and output.samefile(source)
        ):
            raise ValueError("Choose an output path distinct from the input CSV, model, and settings.")


def predict_csv(
    model: CatBoostClassifier,
    settings: ModelSettings,
    input_path: Path,
    output_path: Path,
) -> int:
    temporary_path = None
    count = 0
    try:
        with input_path.open("r", encoding="utf-8-sig", newline="") as source:
            reader = csv.reader(source, strict=True)
            header = next(reader, [])
            positions = feature_positions(header, settings)
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8-sig",
                newline="",
                dir=output_path.parent,
                prefix=f".{output_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                writer = csv.writer(temporary)
                writer.writerow([*header, *RESULT_COLUMNS])
                while rows := list(islice(reader, BATCH_SIZE)):
                    location = f"CSV records {count + 2}-{count + len(rows) + 1}"
                    if any(len(row) != len(header) for row in rows):
                        raise ValueError(f"{location} do not all match the header width.")
                    frame = feature_frame(rows, positions, settings, location)
                    probabilities, decisions = score(model, frame, settings)
                    writer.writerows(
                        [*row, probability, settings.threshold, decision]
                        for row, probability, decision in zip(rows, probabilities.tolist(), decisions.tolist())
                    )
                    count += len(rows)
        if count == 0:
            raise ValueError("The input CSV contains no data records.")
        temporary_path.replace(output_path)
        return count
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score prepared 95-column inputs with the locked CatBoost model.")
    commands = parser.add_subparsers(dest="command", required=True)
    predict = commands.add_parser("predict", help="Append probabilities and decisions to a prepared CSV.")
    predict.add_argument("--model", type=Path, required=True, help="Path to the locked 95-column model.cbm")
    predict.add_argument("--input", type=Path, required=True, help="Prepared CSV containing the 95 named model fields")
    predict.add_argument("--output", type=Path, required=True, help="Destination CSV with original columns and predictions")
    args = parser.parse_args(argv)
    try:
        require_distinct_output(args.output, (args.input, args.model, SETTINGS_PATH))
        settings = read_settings(SETTINGS_PATH)
        model = load_model(args.model, settings)
        count = predict_csv(model, settings, args.input, args.output)
    except (ValueError, OSError, csv.Error, CatBoostError) as exc:
        parser.error(str(exc))
    print(f"Saved {count} predictions to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
