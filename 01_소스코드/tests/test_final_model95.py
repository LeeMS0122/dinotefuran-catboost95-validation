import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from final_model95 import __main__ as cli


class FinalModel95Test(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.input_path = Path(self.directory.name) / "prepared.csv"
        self.output_path = Path(self.directory.name) / "predictions.csv"
        self.features = [f"feature_{index}" for index in range(95)]
        self.settings = cli.ModelSettings(
            features=self.features,
            categorical_indices=[56, 57, 58, 59],
            threshold=0.0583612434566021,
            trees=138,
            model_sha256="0" * 64,
        )

    def write_csv(self, header, rows):
        with self.input_path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(rows)

    def test_scores_float32_and_keeps_metadata_and_column_order(self):
        header = ["sample_id", *reversed(self.features), "note"]
        rows = [
            ["000123", *["2.0"] * 95, "값, 그대로"],
            ["000007", *["0"] * 95, "NA"],
        ]
        self.write_csv(header, rows)
        threshold = self.settings.threshold
        model = Mock()
        model.predict_proba.return_value = np.array([
            [1 - threshold + 1e-12, threshold - 1e-12],
            [0.99, 0.01],
        ])
        with patch.object(cli, "Pool") as pool:
            count = cli.predict_csv(model, self.settings, self.input_path, self.output_path)

        self.assertEqual(count, 2)
        frame = pool.call_args.args[0]
        self.assertEqual(list(frame.columns), self.features)
        self.assertEqual(list(frame["feature_56"]), ["2", "0"])
        self.assertEqual(frame["feature_0"].dtype, np.dtype("float32"))
        self.assertEqual(pool.call_args.kwargs["cat_features"], [56, 57, 58, 59])
        with self.output_path.open(newline="", encoding="utf-8-sig") as handle:
            output = list(csv.reader(handle))
        self.assertEqual(output[0], [*header, "probability", "threshold", "prediction"])
        self.assertEqual([row[:-3] for row in output[1:]], rows)
        self.assertEqual(output[1][-3:], [str(float(np.float32(threshold))), str(threshold), "1"])
        self.assertEqual(output[2][-1], "0")

    def test_rejects_missing_duplicate_and_reserved_headers(self):
        for header, message in (
            (self.features[:-1], "Missing required"),
            ([*self.features, self.features[0]], "unique"),
            ([*self.features, "prediction"], "conflict"),
        ):
            with self.subTest(message=message):
                self.write_csv(header, [])
                with self.assertRaisesRegex(ValueError, message):
                    cli.predict_csv(Mock(), self.settings, self.input_path, self.output_path)
                self.assertFalse(self.output_path.exists())

    def test_rejects_nonfinite_features_and_invalid_categorical_codes(self):
        for index, value, message in (
            (0, "inf", "finite float32"),
            (0, "1e40", "finite float32"),
            (56, "1.5", "nonnegative integer"),
            (56, "-1", "nonnegative integer"),
            (56, "16777217", "exact float32"),
        ):
            with self.subTest(value=value):
                row = ["0"] * 95
                row[index] = value
                self.write_csv(self.features, [row])
                with self.assertRaisesRegex(ValueError, message):
                    cli.predict_csv(Mock(), self.settings, self.input_path, self.output_path)
                self.assertFalse(self.output_path.exists())

    def test_failed_later_batch_does_not_replace_existing_output(self):
        self.write_csv(self.features, [["0"] * 95, ["0"] * 94])
        previous = b"previous result\n"
        self.output_path.write_bytes(previous)
        model = Mock()
        model.predict_proba.return_value = np.array([[0.9, 0.1]])
        with patch.object(cli, "BATCH_SIZE", 1), patch.object(cli, "Pool"):
            with self.assertRaisesRegex(ValueError, "records 3-3"):
                cli.predict_csv(model, self.settings, self.input_path, self.output_path)
        self.assertEqual(self.output_path.read_bytes(), previous)
        self.assertEqual(list(self.output_path.parent.glob("*.tmp")), [])

    def test_disallows_output_alias_of_an_input(self):
        with self.assertRaisesRegex(ValueError, "distinct"):
            cli.require_distinct_output(self.input_path, (self.input_path,))

    def test_rejects_a_different_model_before_native_loading(self):
        model_path = self.input_path.parent / "other.cbm"
        model_path.write_bytes(b"other model")
        with patch.object(cli, "CatBoostClassifier") as classifier:
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                cli.load_model(model_path, self.settings)
            classifier.assert_not_called()


if __name__ == "__main__":
    unittest.main()
