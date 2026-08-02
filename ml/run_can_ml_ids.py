#!/usr/bin/env python3
"""Evaluate the trained Random Forest through common.can_benchmark."""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from common.can_benchmark import (
    CanDetector,
    DetectionResult,
    Frame,
    add_common_arguments,
    run_folder_benchmark,
)
from ml.can_ml_features import (
    FEATURE_COLUMNS,
    SlidingWindowFeatureExtractor,
)


warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
)


class RandomForestCanDetector(CanDetector):
    """ML detector adapter for the shared benchmark interface."""

    def __init__(self, model_path: Path) -> None:
        super().__init__()

        self.model_path = model_path
        self.model = None
        self.threshold = 0.5
        self.attack_index = 1
        self.extractor: SlidingWindowFeatureExtractor | None = None

        # Reuse one NumPy buffer for every frame instead of allocating
        # a new array on each prediction.
        self.feature_vector = np.empty(
            (1, len(FEATURE_COLUMNS)),
            dtype=np.float64,
        )

    def fit(self, calibration_frames: list[Frame]) -> None:
        """Load the trained model and reset streaming state."""

        bundle = joblib.load(self.model_path)

        if list(bundle["feature_columns"]) != FEATURE_COLUMNS:
            raise ValueError(
                "Saved model feature schema does not match this detector"
            )

        self.model = bundle["model"]

        # For one-row predictions, parallel job setup is slower than
        # evaluating the trees on one thread.
        self.model.n_jobs = 1

        try:
            self.attack_index = list(self.model.classes_).index(1)
        except ValueError as exc:
            raise RuntimeError(
                f"Model has no attack class 1; "
                f"classes={self.model.classes_}"
            ) from exc

        self.threshold = float(bundle["threshold"])
        self.extractor = SlidingWindowFeatureExtractor(
            int(bundle["window_size"])
        )

        self.ready = True

    def predict(self, frame: Frame) -> DetectionResult:
        if not self.ready or self.model is None or self.extractor is None:
            raise RuntimeError("Detector must be fitted before prediction")

        features = self.extractor.extract(frame)

        # Copy into the already allocated buffer.
        self.feature_vector[0, :] = features

        probability = float(
            self.model.predict_proba(
                self.feature_vector
            )[0, self.attack_index]
        )

        malicious = probability >= self.threshold

        return DetectionResult(
            malicious=malicious,
            reasons=(
                ("random_forest_threshold",)
                if malicious
                else ()
            ),
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)

    parser.set_defaults(
        input_dir=Path("../build/heldout_test_data"),
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=Path("../build/can_ids_random_forest.joblib"),
    )

    return parser


def detector_factory(
    args: argparse.Namespace,
) -> RandomForestCanDetector:
    return RandomForestCanDetector(args.model)


def main() -> None:
    args = build_parser().parse_args()

    run_folder_benchmark(
        args=args,
        detector_factory=detector_factory,
        training_data=[],
        label="ml",
    )


if __name__ == "__main__":
    main()