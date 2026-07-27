import os
import tempfile
import unittest
from unittest import mock
from http.client import IncompleteRead

import numpy as np

from benchmarks.datasets import _fetch_uci_dataset, load_dataset, load_manifest
from benchmarks.protocol import _feature_row_groups, split_dataset


class DatasetManifestTests(unittest.TestCase):
    def test_uci_fetch_retries_transient_transport_failure(self):
        expected = object()
        fetcher = mock.Mock(side_effect=[IncompleteRead(b"partial"), expected])

        with mock.patch("benchmarks.datasets.time.sleep") as sleep:
            actual = _fetch_uci_dataset(fetcher, 52)

        self.assertIs(actual, expected)
        self.assertEqual(fetcher.call_count, 2)
        fetcher.assert_called_with(id=52)
        sleep.assert_called_once_with(0.5)

    def test_manifest_has_explicit_proxy_license_and_target_mapping(self):
        manifest = load_manifest()
        datasets = manifest["datasets"]
        expected_names = {
            "uci_cardiotocography",
            "uci_secom",
            "uci_ionosphere",
        }
        self.assertTrue(expected_names.issubset(datasets))

        for name in expected_names:
            entry = datasets[name]
            with self.subTest(dataset=name):
                self.assertIsInstance(entry["uci_id"], int)
                self.assertTrue(entry["doi"].startswith("10."))
                self.assertEqual(entry["license"], "CC BY 4.0")
                self.assertEqual(entry["kind"], "uci_proxy")
                self.assertTrue(entry["proxy_boundary"])
                self.assertTrue(entry["source_url"].startswith("https://"))
                self.assertTrue(entry["target_mapping"]["column"])
                self.assertTrue(entry["target_mapping"]["anomaly_values"])
                expected_hash = entry.get("expected_sha256")
                if expected_hash:
                    self.assertEqual(len(expected_hash), 64)
                    int(expected_hash, 16)

    def test_synthetic_dataset_is_deterministic_shuffled_and_binary(self):
        config = {
            "n_samples": 300,
            "n_features": 5,
            "contamination": 0.1,
        }
        first = load_dataset(
            "synthetic_historical",
            seed=2018,
            synthetic_config=config,
        )
        second = load_dataset(
            "synthetic_historical",
            seed=2018,
            synthetic_config=config,
        )

        np.testing.assert_array_equal(first.X, second.X)
        np.testing.assert_array_equal(first.y, second.y)
        self.assertEqual(first.sha256, second.sha256)
        self.assertEqual(first.X.shape, (300, 5))
        self.assertEqual(int(first.y.sum()), 30)
        self.assertEqual(set(first.y.tolist()), {0, 1})
        self.assertTrue(np.any(first.y[:50] == 1))
        self.assertTrue(np.any(first.y[-50:] == 0))

    @unittest.skipUnless(
        os.environ.get("RUN_UCI_INTEGRATION") == "1",
        "Set RUN_UCI_INTEGRATION=1 to download and verify UCI datasets.",
    )
    def test_public_dataset_hashes_match_manifest(self):
        with tempfile.TemporaryDirectory() as cache_dir:
            for name, entry in load_manifest()["datasets"].items():
                if entry["kind"] != "uci_proxy":
                    continue
                with self.subTest(dataset=name):
                    bundle = load_dataset(name, cache_dir=cache_dir)
                    self.assertEqual(bundle.X.ndim, 2)
                    self.assertEqual(bundle.y.ndim, 1)
                    self.assertEqual(len(bundle.X), len(bundle.y))
                    self.assertEqual(set(bundle.y.tolist()), {0, 1})
                    for seed in range(2018, 2023):
                        split = split_dataset(bundle.X, bundle.y, seed=seed)
                        partitions = (
                            split.X_train,
                            split.X_validation,
                            split.X_test,
                        )
                        label_partitions = (
                            split.y_train,
                            split.y_validation,
                            split.y_test,
                        )
                        lengths = [len(partition) for partition in partitions]
                        combined_groups = _feature_row_groups(np.vstack(partitions))
                        stops = np.cumsum([0, *lengths])
                        row_sets = [
                            set(
                                combined_groups[
                                    stops[index] : stops[index + 1]
                                ].tolist()
                            )
                            for index in range(3)
                        ]
                        self.assertTrue(row_sets[0].isdisjoint(row_sets[1]))
                        self.assertTrue(row_sets[0].isdisjoint(row_sets[2]))
                        self.assertTrue(row_sets[1].isdisjoint(row_sets[2]))

                        full_groups = _feature_row_groups(bundle.X)
                        max_group_size = int(np.bincount(full_groups).max())
                        fractions = (0.6, 0.2, 0.2)
                        for fraction, partition, labels in zip(
                            fractions,
                            partitions,
                            label_partitions,
                        ):
                            self.assertLessEqual(
                                abs(
                                    len(partition)
                                    - round(len(bundle.X) * fraction)
                                ),
                                max_group_size,
                            )
                            for label in (0, 1):
                                class_groups = full_groups[bundle.y == label]
                                max_class_group_size = int(
                                    np.bincount(class_groups).max()
                                )
                                self.assertLessEqual(
                                    abs(
                                        int(np.sum(labels == label))
                                        - round(
                                            int(np.sum(bundle.y == label))
                                            * fraction
                                        )
                                    ),
                                    max_class_group_size,
                                )


if __name__ == "__main__":
    unittest.main()
