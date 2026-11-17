import csv
from pathlib import Path
import shutil
import tempfile
import unittest

from final_model95.verify_results import verify_results


class PublishedResultTests(unittest.TestCase):
    results = Path(__file__).resolve().parents[2] / "05_성능검증_상세자료"

    def test_published_tables_match_aggregation_rules(self):
        self.assertEqual(verify_results(self.results), (12, 240))

    def test_aggregate_with_changed_threshold_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            shutil.copyfile(self.results / "per_horizon.csv", directory / "per_horizon.csv")
            with (self.results / "summary.csv").open(encoding="utf-8", newline="") as input_file:
                reader = csv.DictReader(input_file)
                columns = reader.fieldnames
                rows = list(reader)
            rows[0]["threshold"] = "0.5"
            with (directory / "summary.csv").open("w", encoding="utf-8", newline="") as output_file:
                writer = csv.DictWriter(output_file, fieldnames=columns)
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "threshold differs"):
                verify_results(directory)


if __name__ == "__main__":
    unittest.main()
