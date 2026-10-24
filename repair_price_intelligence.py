"""Deterministic repair-rate matching and comparison for sample market data."""
from __future__ import annotations

import csv
import math
import re
from pathlib import Path
from typing import Any


DATASET_PATH = (
    Path(__file__).resolve().parent
    / "DepositProof_10Metros_50Localities_Repair_Rates-v3 (1).csv"
)
DATASET_NAME = DATASET_PATH.name
SAMPLE_DATA_WARNING = (
    "Sample repair-rate data only. These entries are not verified quotations, "
    "transaction records, or guaranteed local prices."
)
REQUIRED_COLUMNS = (
    "City",
    "Locality / Area",
    "Category",
    "Min Price (₹)",
    "Max Price (₹)",
)


class RepairRateDataError(ValueError):
    """The repair-rate dataset is missing or unusable."""


def _normalise(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").casefold()).strip()


def load_repair_rates(csv_path: str | Path = DATASET_PATH) -> dict[str, Any]:
    """Load valid sample rates, retaining source fields and reporting bad rows."""
    path = Path(csv_path)
    if not path.is_file():
        raise RepairRateDataError(f"Repair-rate dataset not found: {path.name}")
    records: list[dict[str, Any]] = []
    issues: list[str] = []
    seen: set[tuple[str, ...]] = set()
    duplicates = 0
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            headers = set(reader.fieldnames or ())
            missing = [column for column in REQUIRED_COLUMNS if column not in headers]
            if missing:
                raise RepairRateDataError(
                    "Repair-rate dataset is missing required columns: " + ", ".join(missing)
                )
            for row_number, row in enumerate(reader, start=2):
                try:
                    minimum = float(row["Min Price (₹)"])
                    maximum = float(row["Max Price (₹)"])
                    if (not math.isfinite(minimum) or not math.isfinite(maximum)
                            or minimum < 0 or maximum < minimum):
                        raise ValueError("price range is invalid")
                    required_values = [row[column].strip() for column in REQUIRED_COLUMNS]
                    if any(not value for value in required_values):
                        raise ValueError("required value is blank")
                except (AttributeError, TypeError, ValueError, KeyError):
                    issues.append(f"Skipped row {row_number}: required fields or price range are invalid.")
                    continue

                signature = tuple(
                    str(row.get(column) or "").strip().casefold()
                    for column in reader.fieldnames or ()
                )
                if signature in seen:
                    duplicates += 1
                    continue
                seen.add(signature)
                records.append({
                    "city": row["City"].strip(),
                    "locality": row["Locality / Area"].strip(),
                    "category": row["Category"].strip(),
                    "minimum": minimum,
                    "maximum": maximum,
                })
    except (OSError, UnicodeError, csv.Error) as exc:
        raise RepairRateDataError(f"Could not read repair-rate dataset: {type(exc).__name__}.") from exc

    if not records:
        detail = f" {issues[0]}" if issues else ""
        raise RepairRateDataError("Repair-rate dataset contains no usable records." + detail)
    return {
        "records": records,
        "issues": issues,
        "duplicate_count": duplicates,
        "dataset_name": path.name,
    }


def compare_repair_price(
    records: list[dict[str, Any]],
    *,
    city: str,
    locality: str | None,
    category: str,
    claimed_amount: float,
) -> dict[str, Any]:
    """Compare a claim with the selected city/locality/category sample rates."""
    try:
        claim = float(claimed_amount)
    except (TypeError, ValueError) as exc:
        raise ValueError("Enter a valid claimed amount.") from exc
    if not math.isfinite(claim) or claim < 0:
        raise ValueError("Claimed amount must be a finite, non-negative number.")
    if not str(city or "").strip():
        raise ValueError("Enter a city from the 10-metro dataset.")
    if not str(category or "").strip():
        raise ValueError("Select a repair category.")

    city_category_rows = [
        row for row in records
        if _normalise(row["city"]) == _normalise(city)
        and _normalise(row["category"]) == _normalise(category)
    ]
    requested_locality = str(locality or "").strip()
    if requested_locality:
        selected = [
            row for row in city_category_rows
            if _normalise(row["locality"]) == _normalise(requested_locality)
        ]
        match_level = "city_locality_category"
    else:
        selected = city_category_rows
        match_level = "city_category"

    common = {
        "source_dataset": DATASET_NAME,
        "sample_data_warning": SAMPLE_DATA_WARNING,
        "city": str(city).strip(),
        "locality": requested_locality or None,
        "category": str(category).strip(),
        "claimed_amount": claim,
    }
    if not selected:
        return {
            **common,
            "status": "no_match",
            "match_level": None,
            "matched_records": [],
            "record_count": 0,
            "explanation": (
                "No sample-rate record was found for this city, locality, and category. "
                "No estimate was created."
            ),
        }

    lower = min(row["minimum"] for row in selected)
    upper = max(row["maximum"] for row in selected)
    if claim < lower:
        position = "below"
        difference = lower - claim
        percent = (difference / lower * 100) if lower else None
    elif claim > upper:
        position = "above"
        difference = claim - upper
        percent = (difference / upper * 100) if upper else None
    else:
        position, difference, percent = "within", 0.0, 0.0

    return {
        **common,
        "status": "matched",
        "match_level": match_level,
        "match_description": {
            "city_locality_category": "Exact city, locality, and category",
            "city_category": "City-wide category range aggregated across listed localities",
        }[match_level],
        "lower": lower,
        "upper": upper,
        "position": position,
        "difference": difference,
        "difference_percent": percent,
        "record_count": len(selected),
        "matched_records": selected,
        "calculation": (
            "The comparison range is the lowest listed minimum through the highest listed maximum "
            "among the matched sample records. Above/below differences are measured from the "
            "nearest range boundary; an in-range claim is not assigned a difference."
        ),
    }
