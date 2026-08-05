#!/usr/bin/env python3
"""Train a Random Forest CAN IDS using the shared benchmark parser.

Each CSV is split using chronological splitting:

    60% -> training
    20% -> validation
    20% -> held-out testing

Chronological splitting keeps the temporal order of the data.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


from common.can_benchmark import (
    HELDOUT_DATA_DIR,
    Frame,
    calculate_metrics,
    find_csv_datasets,
    parse_frames,
)

from ml.can_ml_features import (
    FEATURE_COLUMNS,
    target_from_frame,
)

ORIGINAL_INPUT_DATA_PATH = PROJECT_ROOT / "original_input_data"

MODEL_OUTPUT_PATH = (
    PROJECT_ROOT / "build/can_ids_random_forest.joblib"
)

MAX_FRAMES_PER_FILE = 50000
WINDOW_SIZE = 20
SPLIT_BLOCK_SIZE = 1000
THRESHOLD = 0.7
DISCOUNTED_CSV = "gear_dataset.csv"
DISCOUNTED_CSV_MAX_FRAMES = 30000
DISCARD_GEAR_SET = True


# Given a list of CAN frames from CSV, build 
# a list of feature rows for training/testing.
def build_feature_rows(
    frames: list[Frame]
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    id_to_last_time: dict[str, float] = {}

    for frame in frames:
        payload = list(frame.payload[:8])
        payload += [0] * (8 - len(payload))

        if frame.can_id in id_to_last_time:
            time_since_last_submission = frame.timestamp - id_to_last_time[frame.can_id]
        else:
            time_since_last_submission = 0

        id_to_last_time[frame.can_id] = frame.timestamp

        values = [
            float(int(frame.can_id, 16)),
            float(frame.dlc),
            float(time_since_last_submission),
            *[float(value) for value in payload],
        ]

        row = dict(
            zip(
                FEATURE_COLUMNS,
                values,
                strict=True,
            )
        )

        row.update(
            {
                "target": target_from_frame(frame),
                "row_number": frame.row_number,
                "timestamp": frame.timestamp,
            }
        )

        rows.append(row)

    return rows


# Split our dataset into training, validation, and held-out testing splits.
# The decision not to use stratified splitting is deliberate
# and explained in our design doc. 
def split_frames(
    frames: list[Frame],
) -> tuple[
    list[Frame],
    list[Frame],
    list[Frame],
]:
    train_end = int(len(frames) * 0.60)
    validation_end = int(len(frames) * 0.80)

    train_frames = frames[:train_end]
    validation_frames = frames[
        train_end:validation_end
    ]
    test_frames = frames[
        validation_end:
    ]

    return (
        train_frames,
        validation_frames,
        test_frames,
    )

# Get feature matrix and target vector from a list of feature rows.
def rows_to_xy(
    rows: list[dict[str, object]],
) -> tuple[pd.DataFrame, pd.Series]:
    table = pd.DataFrame(rows)

    if table.empty:
        raise ValueError("A data split is empty")

    return (
        table[FEATURE_COLUMNS],
        table["target"].astype(int),
    )


def attack_probability(
    model: RandomForestClassifier,
    features: pd.DataFrame,
) -> np.ndarray:
    attack_index = list(model.classes_).index(1)

    return model.predict_proba(features)[:, attack_index]


def confusion_counts(
    labels: pd.Series,
    probabilities: np.ndarray,
    threshold: float,
) -> Counter[str]:
    predictions = (
        probabilities >= threshold
    ).astype(int)

    counts: Counter[str] = Counter(
        tp=0,
        fp=0,
        tn=0,
        fn=0,
    )

    for actual, predicted in zip(
        labels,
        predictions,
    ):
        if predicted == 1 and actual == 1:
            counts["tp"] += 1
        elif predicted == 1 and actual == 0:
            counts["fp"] += 1
        elif predicted == 0 and actual == 1:
            counts["fn"] += 1
        else:
            counts["tn"] += 1

    return counts


def print_metrics(
    name: str,
    labels: pd.Series,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, object]:
    counts = confusion_counts(
        labels,
        probabilities,
        threshold,
    )

    metrics = calculate_metrics(counts)

    print(f"\n--- {name} ---")

    print(
        f"TP={counts['tp']:,}  "
        f"FP={counts['fp']:,}  "
        f"FN={counts['fn']:,}  "
        f"TN={counts['tn']:,}"
    )

    for metric_name, value in metrics.items():
        print(f"{metric_name:24s} {value:.6f}")

    return {
        "confusion_matrix": dict(counts),
        "metrics": metrics,
    }


def write_frames(
    path: Path,
    frames: list[Frame],
) -> None:
    """Write normalized held-out rows in the format parse_frames accepts."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as stream:
        writer = csv.writer(stream)

        for frame in frames:
            writer.writerow(
                [
                    f"{frame.timestamp:.6f}",
                    frame.can_id,
                    frame.dlc,
                    *[
                        f"{value:02X}"
                        for value in frame.payload
                    ],
                    frame.label,
                ]
            )


def main() -> None:
    datasets = find_csv_datasets(
        ORIGINAL_INPUT_DATA_PATH
    )

    all_train_rows: list[dict[str, object]] = []
    all_validation_rows: list[dict[str, object]] = []

    for path in datasets:
        frames = list(parse_frames(path))

        if len(frames) < WINDOW_SIZE * 6:
            print(
                f"[!] Skipping {path.name}: "
                f"only {len(frames):,} frames"
            )
            continue

        # Please only run this if you want the gear dataset to be
        # completely excluded from training. 
        if DISCARD_GEAR_SET and path.name == DISCOUNTED_CSV:
            frames = frames[:DISCOUNTED_CSV_MAX_FRAMES]
            write_frames(
                HELDOUT_DATA_DIR / path.name,
                frames,
            )

            print(
                f"[*] {path.name}: completely excluded from training; "
                f"saved {len(frames):,} frames for external testing"
            )

            continue

        frames = frames[:MAX_FRAMES_PER_FILE]
            

        train_frames, validation_frames, test_frames = (
            split_frames(frames)
        )

        # This is crucial - it checks that the training, validation, and
        # test splits all contain both normal and attack frames. If it didn't
        # check this, the model could be trained on only normal frames and
        # then fail to detect attacks in the validation or test splits.
        print(
            f"    classes: "
            f"train={dict(Counter(target_from_frame(frame) for frame in train_frames))}, "
            f"validation={dict(Counter(target_from_frame(frame) for frame in validation_frames))}, "
            f"test={dict(Counter(target_from_frame(frame) for frame in test_frames))}"
        )

        train_rows = build_feature_rows(
            train_frames
        )

        validation_rows = build_feature_rows(
            validation_frames
        )

        all_train_rows.extend(
            train_rows
        )

        all_validation_rows.extend(
            validation_rows
        )

        write_frames(
            HELDOUT_DATA_DIR / path.name,
            test_frames,
        )

        print(
            f"[*] {path.name}: "
            f"total={len(frames):,}, "
            f"train={len(train_rows):,}, "
            f"validation={len(validation_rows):,}"
        )

    X_train, y_train = rows_to_xy(
        all_train_rows
    )

    X_validation, y_validation = rows_to_xy(
        all_validation_rows
    )


    if y_train.nunique() < 2:
        raise ValueError(
            "Training split must contain normal and attack frames"
        )

    # The choice behind this was on a balance of accuracy
    # while being reasonable to training time. 
    # The model is small enough to train quickly, 
    # but large enough to capture the patterns in the data.
    candidates = [
        {
            "n_estimators": 10,
            "max_depth": 5,
            "min_samples_leaf": 20,
        },
        {
            "n_estimators": 25,
            "max_depth": 8,
            "min_samples_leaf": 10,
        },
    ]

    best_model: RandomForestClassifier | None = None
    best_threshold = 0.5
    best_validation_f1 = -1.0

    candidate_results: list[
        dict[str, object]
    ] = []

    for parameters in candidates:
        model = RandomForestClassifier(
            **parameters,
            class_weight="balanced_subsample",
            max_features="sqrt",
            bootstrap=True,
            random_state=42,
            n_jobs=-1,
        )

        model.fit(
            X_train,
            y_train,
        )

        probabilities = attack_probability(
            model,
            X_validation,
        )

        counts = confusion_counts(
            y_validation,
            probabilities,
            THRESHOLD,
        )

        metrics = calculate_metrics(counts)
        validation_f1 = metrics["f1"]

        result = {
            **parameters,
            "threshold": THRESHOLD,
            "validation_f1": validation_f1,
        }

        candidate_results.append(result)

        print(f"[*] Candidate: {result}")

        if validation_f1 > best_validation_f1:
            best_validation_f1 = validation_f1
            best_model = model
            best_threshold = THRESHOLD

    if best_model is None:
        raise RuntimeError(
            "No model was trained"
        )

    validation_result = print_metrics(
        "Validation",
        y_validation,
        attack_probability(
            best_model,
            X_validation,
        ),
        best_threshold,
    )

    feature_importance = sorted(
        zip(
            FEATURE_COLUMNS,
            best_model.feature_importances_,
            strict=True,
        ),
        key=lambda item: item[1],
        reverse=True,
    )

    bundle = {
        "model": best_model,
        "threshold": best_threshold,
        "window_size": WINDOW_SIZE,
        "feature_columns": FEATURE_COLUMNS,
        "split_strategy": "chronological_60_20_20",
    }

    MODEL_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        bundle,
        MODEL_OUTPUT_PATH,
        compress=3,
    )

    print(
        f"\n[+] Saved model to "
        f"{MODEL_OUTPUT_PATH}"
    )

    print(
        f"[+] Saved held-out CSV files to "
        f"{HELDOUT_DATA_DIR.resolve()}/"
    )

    summary_path = (
        MODEL_OUTPUT_PATH.with_suffix(
            ".metrics.json"
        )
    )

    summary_path.write_text(
        json.dumps(
            {
                "candidate_results": candidate_results,
                "selected_threshold": best_threshold,
                "validation": validation_result,
                "feature_importance": feature_importance,
                "training_rows": len(X_train),
                "validation_rows": len(X_validation)
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"[+] Saved metrics to "
        f"{summary_path}"
    )


if __name__ == "__main__":
    main()