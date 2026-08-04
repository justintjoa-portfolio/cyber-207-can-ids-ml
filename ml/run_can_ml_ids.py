#!/usr/bin/env python3
"""Evaluate the trained Random Forest through common.can_benchmark."""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np

MODEL_PATH = Path(__file__).resolve().parent.parent / "build/can_ids_random_forest.joblib"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from common.can_benchmark import (
    CanDetector,
    Frame,
    run_folder_benchmark,
)
from ml.can_ml_features import (
    FEATURE_COLUMNS,
    padded_payload
)


warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
)

THRESHOLD = 0.7

class RandomForestCanDetector(CanDetector):
    """ML detector adapter for the shared benchmark interface."""

    def __init__(self, model_path: Path) -> None:
        super().__init__()

        self.model_path = model_path
        self.model = None
        self.attack_index = 1
        self._last_time_of_id = {}

        # Reuse one NumPy buffer for every frame instead of allocating
        # a new array on each prediction.
        self.feature_vector = np.empty(
            (1, len(FEATURE_COLUMNS)),
            dtype=np.float64,
        )

    def init(self) -> None:
        self._last_time_of_id = {}
        if not self.ready: 
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

            self.ready = True

    def extract(self, frame: Frame) -> list[float]:
        if self._last_time_of_id.get(frame.can_id) is not None:
            time_since_last_submission = frame.timestamp - self._last_time_of_id[frame.can_id]
        else:
            time_since_last_submission = 0

        self._last_time_of_id[frame.can_id] = frame.timestamp

        current_payload = np.asarray(padded_payload(frame), dtype=np.uint8)
        payload_float = current_payload.astype(float)

        features = [
            float(int(frame.can_id, 16)),
            float(frame.dlc),
            float(time_since_last_submission),
            *payload_float.tolist(),
        ]

        return features

    def predict_is_malicious(self, frame: Frame) -> bool:
        if not self.ready or self.model is None:
            raise RuntimeError("Detector must be fitted before prediction")

        features = self.extract(frame)

        # Copy into the already allocated buffer.
        self.feature_vector[0, :] = features

        probability = float(
            self.model.predict_proba(
                self.feature_vector
            )[0, self.attack_index]
        )

        return (probability >= THRESHOLD)


def main() -> None:
    detector = RandomForestCanDetector(
        model_path=MODEL_PATH
    )
    run_folder_benchmark(
        detector=detector,
        label="ml",
    )


if __name__ == "__main__":
    main()