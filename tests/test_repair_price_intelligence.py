import csv
import tempfile
import unittest
from pathlib import Path

from repair_price_intelligence import (
    DATASET_PATH,
    RepairRateDataError,
    compare_repair_price,
    load_repair_rates,
)


class RepairPriceIntelligenceTests(unittest.TestCase):
    def setUp(self):
        self.dataset = load_repair_rates(DATASET_PATH)

    def _compare(self, amount, **overrides):
        values = {
            "city": "Bengaluru",
            "locality": "HSR Layout",
            "category": "Painting / Repainting",
            "claimed_amount": amount,
        }
        values.update(overrides)
        return compare_repair_price(self.dataset["records"], **values)

    def test_loads_real_sample_csv_and_labels_its_provenance(self):
        self.assertEqual(len(self.dataset["records"]), 250)
        self.assertEqual(self.dataset["dataset_name"], DATASET_PATH.name)
        self.assertIn("Sample repair-rate data", self._compare(10000)["sample_data_warning"])

    def test_exact_locality_category_match_and_claim_below_range(self):
        result = self._compare(7000)

        self.assertEqual(result["match_level"], "city_locality_category")
        self.assertEqual(result["lower"], 8800)
        self.assertEqual(result["upper"], 15400)
        self.assertEqual(result["position"], "below")
        self.assertEqual(result["difference"], 1800)
        self.assertEqual(result["difference_percent"], 1800 / 8800 * 100)

    def test_claim_inside_range_is_not_reported_as_a_zero_price_verdict(self):
        result = self._compare(10000)

        self.assertEqual(result["position"], "within")
        self.assertEqual(result["difference"], 0)
        self.assertEqual(result["match_level"], "city_locality_category")

    def test_claim_above_range_is_measured_from_upper_bound(self):
        result = self._compare(16000)

        self.assertEqual(result["position"], "above")
        self.assertEqual(result["difference"], 600)
        self.assertEqual(result["difference_percent"], 600 / 15400 * 100)

    def test_missing_locality_aggregates_city_category_range(self):
        result = self._compare(16000, locality=None)

        expected_rows = [
            row for row in self.dataset["records"]
            if row["city"] == "Bengaluru"
            and row["category"] == "Painting / Repainting"
        ]
        self.assertEqual(result["match_level"], "city_category")
        self.assertEqual(result["record_count"], 5)
        self.assertEqual(result["lower"], min(row["minimum"] for row in expected_rows))
        self.assertEqual(result["upper"], max(row["maximum"] for row in expected_rows))
        self.assertIn("aggregated across listed localities", result["match_description"])

    def test_unknown_locality_does_not_silently_fall_back_to_city_range(self):
        result = self._compare(1000, locality="Unknown locality")

        self.assertEqual(result["status"], "no_match")
        self.assertEqual(result["matched_records"], [])

    def test_no_matching_city_returns_no_estimate(self):
        result = self._compare(1000, city="Unknown City")

        self.assertEqual(result["status"], "no_match")
        self.assertNotIn("lower", result)

    def test_invalid_prices_are_skipped_and_reported(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "rates.csv"
            with path.open("w", encoding="utf-8-sig", newline="") as target:
                writer = csv.DictWriter(target, fieldnames=[
                    "City", "Locality / Area", "Category", "Min Price (₹)", "Max Price (₹)",
                ])
                writer.writeheader()
                writer.writerow({
                    "City": "Bengaluru", "Locality / Area": "HSR Layout",
                    "Category": "Painting / Repainting",
                    "Min Price (₹)": "not-a-price", "Max Price (₹)": "14000",
                })
            with self.assertRaisesRegex(RepairRateDataError, "no usable records"):
                load_repair_rates(path)

    def test_missing_required_column_has_clear_error(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "rates.csv"
            path.write_text(
                "City,Locality / Area,Category,Min Price (₹)\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RepairRateDataError, "missing required columns"):
                load_repair_rates(path)

    def test_missing_csv_has_clear_error(self):
        with self.assertRaisesRegex(RepairRateDataError, "not found"):
            load_repair_rates(Path(DATASET_PATH.parent) / "not-present.csv")


if __name__ == "__main__":
    unittest.main()
