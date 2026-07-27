import csv
import pathlib
import tempfile
import unittest

from benchmarks.run import load_config, run_benchmark, write_outputs


REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[1]


class BenchmarkSmokeTests(unittest.TestCase):
    def test_offline_smoke_runs_and_writes_complete_outputs(self):
        config = load_config(REPOSITORY_ROOT / "benchmarks" / "configs" / "smoke.json")
        rows = run_benchmark(config)

        self.assertEqual(len(rows), 10)
        self.assertTrue(all(row["status"] == "ok" for row in rows))
        self.assertEqual(
            [(row["dataset"], row["detector"], row["trial_seed"]) for row in rows],
            sorted(
                (row["dataset"], row["detector"], row["trial_seed"])
                for row in rows
            ),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            write_outputs(temp_dir, config, rows)
            output_path = pathlib.Path(temp_dir)
            expected_files = {
                "config.resolved.json",
                "run_metadata.json",
                "metrics.csv",
                "timings.csv",
                "summary.csv",
                "report.md",
            }
            self.assertEqual(
                {path.name for path in output_path.iterdir()},
                expected_files,
            )

            with (output_path / "metrics.csv").open(
                "r",
                encoding="utf-8",
                newline="",
            ) as handle:
                metrics = list(csv.DictReader(handle))
            self.assertEqual(len(metrics), len(rows))
            self.assertTrue(all(row["error"] == "" for row in metrics))

            report = (output_path / "report.md").read_text(encoding="utf-8")
            self.assertNotIn("Best Algorithm", report)
            self.assertIn("does not select an overall winner", report)
            self.assertIn("classification-to-anomaly proxies", report)
            self.assertIn("| Successful | Failed |", report)


if __name__ == "__main__":
    unittest.main()
