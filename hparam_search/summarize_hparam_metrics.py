"""Summarize metric values across hyperparameter configurations.

Run from any directory with:
    python hparam_search/summarize_hparam_metrics.py

To also update the summary columns in tables.txt:
    python hparam_search/summarize_hparam_metrics.py --update-tables
"""

import argparse
import csv
import math
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TABLE_SOURCES = {
    "iot_gan": ROOT / "iot" / "hparam_search_iot_gan.csv",
    "iot_vae": ROOT / "iot" / "hparam_search_iot_vae.csv",
    "m5_gan": ROOT / "m5" / "hparam_search_m5_gan.csv",
    "m5_vae": ROOT / "m5" / "hparam_search_m5_vae.csv",
}
TABLE_PATH = ROOT / "tables.txt"
MAX_CONFIGURATIONS = 5

METRICS = {
    "delta0ratio": "zero_ratio_diff",
    "deltamean": "mean_diff",
    "deltastd": "std_diff",
    "deltaskewness": "skewness_diff",
    "deltakurtosis": "kurtosis_diff",
    "deltaq10": "quantile_10_diff",
    "deltaq25": "quantile_25_diff",
    "deltaq50": "quantile_50_diff",
    "deltaq75": "quantile_75_diff",
    "deltaq90": "quantile_90_diff",
    "deltaq95": "quantile_95_diff",
    "deltaq99": "quantile_99_diff",
    "deltaqmean": "mean_quantile_diff",
    "autocorrmae": "autocorr_mae",
    "wassersteindist": "wasserstein_distance",
    "wassersteindistance": "wasserstein_distance",
    "kl divergence": "kl_divergence",
    "kldivergence": "kl_divergence",
    "mmd": "mmd",
    "ps": "ps",
    "ds": "ds",
    "lps": "lps",
    "lds": "lds",
    "train time s": "train_time",
}
METRIC_NAMES = tuple(dict.fromkeys(METRICS.values()))


def normalized_label(label):
    label = label.replace(r"\shortstack", "").replace(r"\mathit", "")
    return re.sub(r"[^a-z0-9]", "", label.lower())


def load_summaries(csv_path):
    with csv_path.open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        if not reader.fieldnames:
            raise ValueError(f"No header found in {csv_path}")

        metric_columns = {
            metric: "elapsed_s" if metric == "train_time" else f"metric__{metric}"
            for metric in METRIC_NAMES
        }
        missing = sorted(
            column for column in metric_columns.values()
            if column not in reader.fieldnames
        )
        if missing:
            raise ValueError(f"Missing metric columns in {csv_path}: {', '.join(missing)}")

        values = {metric: [] for metric in metric_columns}
        for row in reader:
            status = row.get("status", "").strip().lower()
            if status and status != "ok":
                continue
            for metric, column in metric_columns.items():
                raw_value = row.get(column, "").strip()
                if not raw_value:
                    raise ValueError(
                        f"Empty value for {column} in {csv_path}, run_id={row.get('run_id')}"
                    )
                try:
                    value = float(raw_value)
                except ValueError as error:
                    raise ValueError(
                        f"Invalid value for {column} in {csv_path}, "
                        f"run_id={row.get('run_id')}: {raw_value!r}"
                    ) from error
                if not math.isfinite(value):
                    raise ValueError(
                        f"Non-finite value for {column} in {csv_path}, "
                        f"run_id={row.get('run_id')}: {raw_value!r}"
                    )
                values[metric].append(value)

    summaries = {}
    for metric, metric_values in values.items():
        if not metric_values:
            raise ValueError(f"No successful values for {metric} in {csv_path}")
        summaries[metric] = (
            math.fsum(metric_values) / len(metric_values),
            min(metric_values),
            max(metric_values),
        )
    return summaries


def format_number(value):
    return f"{value:.3f}"


def format_range(minimum, maximum):
    return f"{format_number(minimum)}--{format_number(maximum)}"


def metric_for_row(label):
    normalized = normalized_label(label)
    for alias, metric in METRICS.items():
        if normalized_label(alias) == normalized:
            return metric
    return None


def update_table_block(block, summaries, table_name):
    lines = block.splitlines()
    header_start = next(
        (index for index, line in enumerate(lines) if r"\textbf{RunID}" in line),
        None,
    )
    if header_start is None:
        raise ValueError(f"Could not find RunID header for {table_name}")
    header_end = next(
        (index for index in range(header_start + 1, len(lines))
         if r"\midrule" in lines[index]),
        None,
    )
    if header_end is None:
        raise ValueError(f"Could not find header separator for {table_name}")

    old_header = "\n".join(lines[header_start:header_end])
    run_ids = re.findall(r"\\textbf\{(\d+)\}", old_header)
    if not run_ids:
        raise ValueError(f"Could not read configuration IDs for {table_name}")
    kept_ids = run_ids[:MAX_CONFIGURATIONS]

    header = (
        r"\textbf{RunID} & "
        + " & ".join(rf"\textbf{{{run_id}}}" for run_id in kept_ids)
        + r" & \textbf{Mean} & \textbf{Min--Max} \\"
    )
    lines[header_start:header_end] = [header]
    longtable_index = next(
        (index for index, line in enumerate(lines) if r"\begin{longtable}" in line),
        None,
    )
    if longtable_index is not None:
        lines[longtable_index] = re.sub(
            r"\{l\s+c+\}",
            "{l " + "c" * (len(kept_ids) + 2) + "}",
            lines[longtable_index],
        )
        for index, line in enumerate(lines):
            if r"\multicolumn{" in line and "Continued on next page" in line:
                lines[index] = re.sub(
                    r"\\multicolumn\{\d+\}",
                    rf"\\multicolumn{{{len(kept_ids) + 3}}}",
                    line,
                )
    else:
        for index, line in enumerate(lines):
            if r"\begin{tabular*}" in line:
                lines[index] = re.sub(
                    r"l\|c+",
                    "l|" + "c" * (len(kept_ids) + 2),
                    line,
                )
                break

    metric_rows = set()
    for index, line in enumerate(lines):
        if index == header_start or "&" not in line:
            continue
        row_ending = re.search(r"(\\\\(?:\s*\\hline)?)\s*$", line)
        if row_ending:
            content = line[:row_ending.start()].rstrip()
            suffix = " " + row_ending.group(1)
        else:
            content = line.rstrip()
            suffix = ""
        parts = content.split("&")
        metric = metric_for_row(parts[0])
        config_values = [part.strip() for part in parts[1:1 + len(kept_ids)]]
        if len(config_values) != len(kept_ids):
            raise ValueError(
                f"Not enough configuration values for row {parts[0].strip()} in {table_name}"
            )
        if metric is not None:
            if metric in metric_rows:
                raise ValueError(f"Duplicate metric row {metric} in {table_name}")
            metric_rows.add(metric)
            mean, minimum, maximum = summaries[metric]
            summary_cells = [format_number(mean), format_range(minimum, maximum)]
        else:
            summary_cells = ["", ""]
        lines[index] = (
            parts[0].rstrip()
            + " & "
            + " & ".join(config_values)
            + " & "
            + " & ".join(summary_cells)
            + suffix
        )

    expected_metrics = set(summaries)
    if metric_rows != expected_metrics:
        missing = sorted(expected_metrics - metric_rows)
        extra = sorted(metric_rows - expected_metrics)
        raise ValueError(
            f"Metric rows do not match in {table_name}; "
            f"missing={missing}, extra={extra}"
        )
    data, model = table_name.split("_")
    data_label = {"iot": "IoT", "m5": "M5"}[data]
    caption = (
        f"Results obtained by the ZITS-{model.upper()} model on the {data_label} data "
        "(five configurations ordered by Borda ranking; mean and min--max are "
        "across all configurations)"
    )
    updated_block = "\n".join(lines)
    return re.sub(
        r"\\caption\{.*?\}",
        lambda _match: rf"\caption{{{caption}}}",
        updated_block,
        count=1,
        flags=re.DOTALL,
    )


def update_tables(all_summaries, table_path):
    text = table_path.read_text(encoding="utf-8")
    for table_name, summaries in all_summaries.items():
        label = rf"\label{{tab:hyperparams_{table_name}}}"
        label_position = text.find(label)
        if label_position < 0:
            raise ValueError(f"Could not find table label {label}")
        begin_candidates = [
            position for marker in (r"\begin{longtable}", r"\begin{table}")
            if (position := text.rfind(marker, 0, label_position)) >= 0
        ]
        if not begin_candidates:
            raise ValueError(f"Could not find beginning of table {table_name}")
        start = max(begin_candidates)
        end_candidates = [
            position for marker in (r"\end{longtable}", r"\end{table}")
            if (position := text.find(marker, start)) >= 0
        ]
        if not end_candidates:
            raise ValueError(f"Could not find end of table {table_name}")
        end = min(end_candidates) + len(
            r"\end{longtable}" if text.find(r"\end{longtable}", start) == min(end_candidates)
            else r"\end{table}"
        )
        block = text[start:end]
        updated = update_table_block(block, summaries, table_name)
        text = text[:start] + updated + text[end:]
    table_path.write_text(text, encoding="utf-8", newline="")


def print_summaries(all_summaries):
    for table_name, summaries in all_summaries.items():
        print(f"\n{table_name}:")
        print(f"{'metric':<25} {'mean':>12} {'min--max':>25}")
        for metric, (mean, minimum, maximum) in summaries.items():
            print(
                f"{metric:<25} {format_number(mean):>12} "
                f"{format_range(minimum, maximum):>25}"
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--update-tables",
        action="store_true",
        help="keep the first five configurations and write mean/min-max columns to tables.txt",
    )
    parser.add_argument(
        "--tables",
        type=Path,
        default=TABLE_PATH,
        help="LaTeX table file to update (default: hparam_search/tables.txt)",
    )
    args = parser.parse_args()

    all_summaries = {
        table_name: load_summaries(csv_path)
        for table_name, csv_path in TABLE_SOURCES.items()
    }
    print_summaries(all_summaries)
    if args.update_tables:
        update_tables(all_summaries, args.tables)
        print(f"\nUpdated tables in {args.tables}")


if __name__ == "__main__":
    main()
