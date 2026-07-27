import unittest

import numpy as np

from benchmarks.protocol import (
    DataSplit,
    _feature_row_groups,
    evaluate_detector,
    preprocess_split,
    select_validation_threshold,
    split_dataset,
)


class BenchmarkProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(2018)
        normal = rng.normal(size=(180, 6))
        anomalies = rng.normal(loc=4.0, scale=0.7, size=(20, 6))
        cls.X = np.vstack([normal, anomalies])
        cls.y = np.r_[np.zeros(len(normal)), np.ones(len(anomalies))].astype(np.int8)

    def test_split_is_reproducible_stratified_and_disjoint(self):
        first = split_dataset(self.X, self.y, seed=2018)
        second = split_dataset(self.X, self.y, seed=2018)

        for field in (
            "X_train",
            "X_validation",
            "X_test",
            "y_train",
            "y_validation",
            "y_test",
        ):
            np.testing.assert_array_equal(
                getattr(first, field),
                getattr(second, field),
            )

        self.assertEqual(len(first.X_train), 120)
        self.assertEqual(len(first.X_validation), 40)
        self.assertEqual(len(first.X_test), 40)
        for labels in (first.y_train, first.y_validation, first.y_test):
            self.assertEqual(set(labels.tolist()), {0, 1})

        row_sets = self._partition_group_sets(first)
        self.assertTrue(row_sets[0].isdisjoint(row_sets[1]))
        self.assertTrue(row_sets[0].isdisjoint(row_sets[2]))
        self.assertTrue(row_sets[1].isdisjoint(row_sets[2]))
        self.assertEqual(len(set.union(*row_sets)), len(self.X))

    def test_preprocessing_is_independent_of_test_values(self):
        split = split_dataset(self.X, self.y, seed=2018)
        changed_test = DataSplit(
            X_train=split.X_train,
            X_validation=split.X_validation,
            X_test=split.X_test + 10000.0,
            y_train=split.y_train,
            y_validation=split.y_validation,
            y_test=split.y_test,
        )

        transformed, _, _ = preprocess_split(split)
        transformed_changed, _, _ = preprocess_split(changed_test)
        np.testing.assert_allclose(
            transformed.X_train,
            transformed_changed.X_train,
        )
        np.testing.assert_allclose(
            transformed.X_validation,
            transformed_changed.X_validation,
        )

    def test_exact_duplicate_rows_stay_in_one_partition(self):
        base = np.arange(70, dtype=float).reshape(35, 2)
        labels = np.r_[np.zeros(25), np.ones(10)].astype(np.int8)
        X = np.repeat(base, 2, axis=0)
        y = np.repeat(labels, 2)

        split = split_dataset(X, y, seed=2018)
        row_sets = self._partition_group_sets(split)

        self.assertTrue(row_sets[0].isdisjoint(row_sets[1]))
        self.assertTrue(row_sets[0].isdisjoint(row_sets[2]))
        self.assertTrue(row_sets[1].isdisjoint(row_sets[2]))
        self.assertEqual(len(set.union(*row_sets)), len(base))

    def test_exact_duplicate_rows_with_conflicting_labels_fail(self):
        X = np.array(
            [[0.0], [0.0], [1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0]]
        )
        y = np.array([0, 1, 0, 0, 0, 0, 1, 1, 1], dtype=np.int8)

        with self.assertRaisesRegex(ValueError, "conflicting labels"):
            split_dataset(X, y, seed=2018)

    def test_non_binary_labels_fail_before_integer_cast(self):
        invalid = self.y.astype(float)
        invalid[0] = 0.5

        with self.assertRaisesRegex(ValueError, "only 0 and 1"):
            split_dataset(self.X, invalid, seed=2018)

    def test_preprocessing_is_independent_of_validation_values(self):
        split = split_dataset(self.X, self.y, seed=2018)
        changed_validation = DataSplit(
            X_train=split.X_train,
            X_validation=split.X_validation + 10000.0,
            X_test=split.X_test,
            y_train=split.y_train,
            y_validation=split.y_validation,
            y_test=split.y_test,
        )

        transformed, _, _ = preprocess_split(split)
        transformed_changed, _, _ = preprocess_split(changed_validation)
        np.testing.assert_allclose(
            transformed.X_train,
            transformed_changed.X_train,
        )
        np.testing.assert_allclose(
            transformed.X_test,
            transformed_changed.X_test,
        )

    def test_validation_threshold_tie_chooses_higher_value(self):
        labels = np.array([0, 1, 0, 1], dtype=np.int8)
        scores = np.array([0.1, 0.4, 0.2, 0.3])
        threshold = select_validation_threshold(labels, scores)
        self.assertEqual(threshold, 0.3)

    def test_test_labels_cannot_change_threshold_or_predictions(self):
        raw_split = split_dataset(self.X, self.y, seed=2018)
        split, preprocessing_fit, preprocessing_transform = preprocess_split(raw_split)
        flipped = DataSplit(
            X_train=split.X_train,
            X_validation=split.X_validation,
            X_test=split.X_test,
            y_train=split.y_train,
            y_validation=split.y_validation,
            y_test=1 - split.y_test,
        )

        common = {
            "dataset_name": "test",
            "dataset_sha256": "0" * 64,
            "source": "unit-test",
            "detector_name": "mahalanobis",
            "trial_seed": 2018,
            "preprocessing_fit_seconds": preprocessing_fit,
            "preprocessing_transform_seconds": preprocessing_transform,
        }
        original = evaluate_detector(split=split, **common)
        changed = evaluate_detector(split=flipped, **common)

        self.assertEqual(original["status"], "ok")
        self.assertEqual(changed["status"], "ok")
        self.assertEqual(original["threshold"], changed["threshold"])
        self.assertNotEqual(original["average_precision"], changed["average_precision"])

    def test_all_registered_detectors_return_finite_or_explicit_error_rows(self):
        raw_split = split_dataset(self.X, self.y, seed=2020)
        split, preprocessing_fit, preprocessing_transform = preprocess_split(raw_split)

        for detector_name in (
            "mahalanobis",
            "pca_reconstruction",
            "isolation_forest",
            "local_outlier_factor",
            "one_class_svm",
        ):
            with self.subTest(detector=detector_name):
                row = evaluate_detector(
                    dataset_name="test",
                    dataset_sha256="0" * 64,
                    source="unit-test",
                    detector_name=detector_name,
                    split=split,
                    trial_seed=2020,
                    preprocessing_fit_seconds=preprocessing_fit,
                    preprocessing_transform_seconds=preprocessing_transform,
                )
                self.assertEqual(row["status"], "ok", row["error"])
                self.assertGreaterEqual(row["average_precision"], 0.0)
                self.assertLessEqual(row["average_precision"], 1.0)
                self.assertGreaterEqual(row["fit_seconds"], 0.0)
                self.assertGreaterEqual(row["test_score_seconds"], 0.0)

    @staticmethod
    def _partition_group_sets(split):
        partitions = (
            split.X_train,
            split.X_validation,
            split.X_test,
        )
        lengths = [len(partition) for partition in partitions]
        groups = _feature_row_groups(np.vstack(partitions))
        stops = np.cumsum([0, *lengths])
        return [
            set(groups[stops[index] : stops[index + 1]].tolist())
            for index in range(3)
        ]


if __name__ == "__main__":
    unittest.main()
