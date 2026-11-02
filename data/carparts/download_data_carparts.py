"""Download and preprocess the Car Parts monthly time-series dataset."""

import csv
import io
import zipfile
from pathlib import Path
from urllib.request import urlopen

import numpy as np


DATA_DIR = Path(__file__).resolve().parent / "data" / "carparts"
ARCHIVE_NAME = "car_parts_dataset_without_missing_values.zip"
TSF_NAME = "car_parts_dataset_without_missing_values.tsf"
DOWNLOAD_URL = (
    "https://zenodo.org/records/4656021/files/"
    "car_parts_dataset_without_missing_values.zip?download=1"
)


def _monthly_dates(start: str, length: int) -> list[str]:
    year, month = (int(part) for part in start[:7].split("-"))
    dates = []
    for _ in range(length):
        dates.append(f"{year:04d}-{month:02d}")
        month += 1
        if month == 13:
            year += 1
            month = 1
    return dates


def _parse_tsf(content: str) -> tuple[list[str], np.ndarray, str]:
    lines = content.lstrip("\ufeff").splitlines()
    try:
        data_start = next(
            index for index, line in enumerate(lines)
            if line.strip().lower() == "@data"
        )
    except StopIteration as error:
        raise ValueError("The downloaded TSF file has no @data section.") from error

    headers = {line.strip().lower() for line in lines[:data_start]}
    if "@frequency monthly" not in headers:
        raise ValueError("Expected monthly Car Parts observations.")
    if "@missing false" not in headers or "@equallength true" not in headers:
        raise ValueError("Expected complete, equally sized time series.")

    names = []
    rows = []
    start_timestamp = None
    for line_number, line in enumerate(lines[data_start + 1:], start=data_start + 2):
        if not line.strip():
            continue
        parts = line.split(":", 2)
        if len(parts) != 3:
            raise ValueError(f"Malformed TSF record at line {line_number}.")
        name, timestamp, raw_values = parts
        values = [int(value) for value in raw_values.split(",")]
        if not values or any(value < 0 for value in values):
            raise ValueError(f"Invalid sales values at line {line_number}.")
        if start_timestamp is None:
            start_timestamp = timestamp
        elif timestamp != start_timestamp:
            raise ValueError("Series have different start timestamps.")
        names.append(name)
        rows.append(values)

    if len(rows) != 2674:
        raise ValueError(f"Expected 2674 series, found {len(rows)}.")
    if not rows or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("Car Parts series must all have the same length.")
    if len(rows[0]) != 51:
        raise ValueError(f"Expected 51 monthly observations, found {len(rows[0])}.")

    data = np.asarray(rows, dtype=np.int32)
    return names, data, start_timestamp[:10]


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading Car Parts dataset from {DOWNLOAD_URL}")
    with urlopen(DOWNLOAD_URL, timeout=60) as response:
        archive_bytes = response.read()

    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        tsf_files = [
            name for name in archive.namelist()
            if name.lower().endswith(".tsf") and not name.endswith("/")
        ]
        if len(tsf_files) != 1:
            raise ValueError(f"Expected one TSF file in archive; found {tsf_files}.")
        tsf_bytes = archive.read(tsf_files[0])

    names, data, start_date = _parse_tsf(tsf_bytes.decode("utf-8-sig"))
    dates = _monthly_dates(start_date, data.shape[1])

    (DATA_DIR / ARCHIVE_NAME).write_bytes(archive_bytes)
    (DATA_DIR / TSF_NAME).write_bytes(tsf_bytes)
    np.save(DATA_DIR / "carparts_X_51.npy", data)
    np.savez_compressed(DATA_DIR / "carparts_X_51.npz", data=data[..., np.newaxis])

    with (DATA_DIR / "carparts_series.csv").open(
        "w", encoding="utf-8", newline=""
    ) as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["series_name", *dates])
        writer.writerows(
            [name, *row.tolist()] for name, row in zip(names, data)
        )

    print(f"Saved {len(names)} series with {data.shape[1]} monthly observations.")
    print(f"Array shape: {data.shape}; zero ratio: {np.mean(data == 0):.2%}")
    print(f"Preprocessed dataset: {DATA_DIR}")


if __name__ == "__main__":
    main()
