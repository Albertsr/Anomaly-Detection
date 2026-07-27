import importlib.util
import pathlib
import unittest

import numpy as np
from sklearn.exceptions import NotFittedError

from anomaly_detection import MahalanobisDetector, PCAReconstructionDetector


REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_legacy_module(module_name, relative_path):
    """Load a historical script without changing its path or source."""
    module_path = REPOSITORY_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class StableAPIContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(2018)
        normal = rng.normal(loc=0.0, scale=1.0, size=(48, 4))
        outliers = rng.normal(loc=5.0, scale=0.8, size=(4, 4))
        cls.X = np.vstack([normal, outliers])

    def test_fit_score_predict_contract(self):
        for detector in (
            MahalanobisDetector(contamination=0.1),
            PCAReconstructionDetector(contamination=0.1),
        ):
            with self.subTest(detector=type(detector).__name__):
                self.assertIs(detector.fit(self.X), detector)

                scores = detector.score_samples(self.X)
                prediction = detector.predict(self.X)

                self.assertEqual(scores.shape, (len(self.X),))
                self.assertTrue(np.all(np.isfinite(scores)))
                self.assertEqual(prediction.shape, (len(self.X),))
                self.assertTrue(set(prediction.tolist()).issubset({0, 1}))
                self.assertGreaterEqual(
                    int(prediction.sum()),
                    int(np.ceil(len(self.X) * 0.1)),
                )

    def test_predict_marks_scores_at_or_above_training_threshold(self):
        for detector in (
            MahalanobisDetector(contamination=0.1),
            PCAReconstructionDetector(contamination=0.1),
        ):
            with self.subTest(detector=type(detector).__name__):
                scores = detector.fit(self.X).score_samples(self.X)
                np.testing.assert_array_equal(
                    detector.predict(self.X),
                    (scores >= detector.threshold_).astype(np.int8),
                )

    def test_mahalanobis_scores_clear_outliers_higher_on_average(self):
        scores = MahalanobisDetector(contamination=0.1).fit(self.X).train_scores_
        self.assertGreater(
            float(np.mean(scores[48:])),
            float(np.mean(scores[:48])),
        )

    def test_predict_accepts_one_sample_after_fit(self):
        detector = MahalanobisDetector().fit(self.X)
        prediction = detector.predict(self.X[:1])
        self.assertEqual(prediction.shape, (1,))

    def test_scores_do_not_depend_on_test_batch_composition(self):
        for detector in (
            MahalanobisDetector(contamination=0.1),
            PCAReconstructionDetector(contamination=0.1),
        ):
            with self.subTest(detector=type(detector).__name__):
                detector.fit(self.X)
                scores_together = detector.score_samples(self.X[:3])
                scores_separately = np.array(
                    [
                        detector.score_samples(self.X[index : index + 1])[0]
                        for index in range(3)
                    ]
                )
                np.testing.assert_allclose(
                    scores_together,
                    scores_separately,
                    rtol=1e-12,
                    atol=1e-12,
                )

    def test_not_fitted_and_invalid_inputs_fail_clearly(self):
        with self.assertRaises(NotFittedError):
            MahalanobisDetector().score_samples(self.X)

        zero_contamination = MahalanobisDetector(contamination=0).fit(self.X)
        self.assertEqual(int(zero_contamination.predict(self.X).sum()), 0)
        self.assertTrue(np.isinf(zero_contamination.threshold_))

        for contamination in (-0.01, 0.51, np.nan, np.inf, True):
            with self.subTest(contamination=contamination):
                with self.assertRaises(ValueError):
                    MahalanobisDetector(contamination=contamination).fit(self.X)

        with self.assertRaises(ValueError):
            MahalanobisDetector().fit(np.array([[0.0, np.nan], [1.0, 2.0]]))

        with self.assertRaises(ValueError):
            MahalanobisDetector().fit(np.array([[0.0, np.inf], [1.0, 2.0]]))

        with self.assertRaises(ValueError):
            MahalanobisDetector().fit(np.array([0.0, 1.0, 2.0]))

        detector = MahalanobisDetector().fit(self.X)
        with self.assertRaisesRegex(ValueError, "features"):
            detector.score_samples(np.ones((2, self.X.shape[1] + 1)))

    def test_mahalanobis_matches_historical_handwritten_scores(self):
        legacy = load_legacy_module(
            "legacy_mahal_dist",
            pathlib.Path("UnSupervised-Mahalanobis Distance") / "mahal_dist.py",
        )
        expected = legacy.cal_mahal_dist(self.X)
        actual = MahalanobisDetector().fit(self.X).train_scores_
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)

    def test_pca_matches_historical_handwritten_scores(self):
        legacy = load_legacy_module(
            "legacy_pca_reconstruction",
            pathlib.Path("UnSupervised-Based on PCA") / "recon_error_pca.py",
        )
        np.random.seed(2018)
        expected = legacy.PCA_Recon_Error(self.X, contamination=0.1).get_anomaly_score()
        actual = PCAReconstructionDetector(contamination=0.1).fit(self.X).train_scores_
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)

    def test_frozen_numeric_fixture_protects_both_score_definitions(self):
        X_fixture = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 2.0],
                [2.0, 1.0],
                [3.0, 4.0],
                [8.0, -1.0],
            ]
        )
        expected_mahalanobis = np.array(
            [
                1.1329616721397215,
                0.8479893994062522,
                0.8479893994062522,
                0.11593472394004212,
                1.838579980334285,
                1.970890306980715,
            ]
        )
        expected_pca = np.array(
            [
                0.6700905182929273,
                0.5032595763279086,
                0.10845387754382688,
                0.05561031398833957,
                0.9536755891003301,
                0.3837386970526719,
            ]
        )

        np.testing.assert_allclose(
            MahalanobisDetector().fit(X_fixture).train_scores_,
            expected_mahalanobis,
            rtol=1e-10,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            PCAReconstructionDetector().fit(X_fixture).train_scores_,
            expected_pca,
            rtol=1e-10,
            atol=1e-10,
        )

    def test_singular_covariance_has_finite_scores(self):
        X_singular = np.column_stack(
            [
                np.arange(8, dtype=float),
                np.arange(8, dtype=float),
            ]
        )
        scores = MahalanobisDetector(contamination=0.25).fit(X_singular).train_scores_
        self.assertTrue(np.all(np.isfinite(scores)))

    def test_threshold_ties_are_inclusive_and_documented(self):
        X_tied = np.ones((8, 2), dtype=float)
        detector = MahalanobisDetector(contamination=0.25).fit(X_tied)

        self.assertEqual(detector.threshold_, 0.0)
        self.assertEqual(int(detector.predict(X_tied).sum()), len(X_tied))

    def test_threshold_count_uses_ceiling_for_odd_sample_count(self):
        rng = np.random.default_rng(7)
        X_odd = rng.normal(size=(5, 3))
        detector = MahalanobisDetector(contamination=0.5).fit(X_odd)

        self.assertEqual(len(np.unique(detector.train_scores_)), len(X_odd))
        self.assertEqual(int(detector.predict(X_odd).sum()), 3)

    def test_pca_supports_single_feature_and_more_features_than_samples(self):
        single_feature = np.arange(8, dtype=float).reshape(-1, 1)
        high_dimensional = np.arange(15, dtype=float).reshape(3, 5)

        for X in (single_feature, high_dimensional):
            with self.subTest(shape=X.shape):
                scores = PCAReconstructionDetector(contamination=0.5).fit(X).train_scores_
                self.assertEqual(scores.shape, (len(X),))
                self.assertTrue(np.all(np.isfinite(scores)))

    def test_pca_rejects_all_constant_training_data(self):
        with self.assertRaisesRegex(ValueError, "non-zero variance"):
            PCAReconstructionDetector().fit(np.ones((8, 3)))

    def test_repeated_fit_is_deterministic(self):
        first_detector = PCAReconstructionDetector(random_state=2018).fit(self.X)
        second_detector = PCAReconstructionDetector(random_state=2018).fit(self.X)

        self.assertEqual(first_detector.pca_.svd_solver, "full")
        first = first_detector.train_scores_
        second = second_detector.train_scores_
        np.testing.assert_allclose(first, second, rtol=0.0, atol=0.0)


if __name__ == "__main__":
    unittest.main()
