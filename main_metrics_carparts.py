import csv
import os
import tempfile

import numpy as np

from metrics import calculate_evaluation_metrics


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
REAL_DATA_PATH = os.path.join(ROOT_DIR, "data", "carparts", "carparts_X_51.npz")
OUTPUT_CSV = os.path.join(ROOT_DIR, "results_cp", "carparts_metrics.csv")

RUN_FILES = {
    "zip": {
        f"run{run}": f"baseline/run{run}/zip_generated_data.npz"
        for run in range(1, 4)
    },
    "hurdle": {
        f"run{run}": f"baseline/run{run}/hurdle_generated_data.npz"
        for run in range(1, 4)
    },
    "chronogan": {
        f"run{run}": f"chronogan/run{run}/synthetic_data.npz"
        for run in range(1, 4)
    },
    "fide": {
        f"run{run}": f"fide/run{run}/fide_generated_data.npz"
        for run in range(1, 4)
    },
    "timegan": {
        f"run{run}": f"timegan/run{run}/carparts_X_51.npz"
        for run in range(1, 4)
    },
    "transfusion": {
        f"run{run}": f"transfusion/run{run}/synthetic_data.npz"
        for run in range(1, 4)
    },
    "tsgm_timegan": {
        f"run{run}": f"tsgm_timegan/run{run}/tsgm_timegan_gen.npz"
        for run in range(1, 4)
    },
    "tsgm_vae": {
        f"run{run}": f"tsgm_vae/run{run}/tsgm_vae_gen.npz"
        for run in range(1, 4)
    },
    "timeVAE": {
        f"run{run}": (
            f"timeVAE/carparts_X_51_run{run}/"
            "timeVAE_carparts_X_51_prior_samples.npz"
        )
        for run in range(1, 4)
    },
    "vae_dense": {
        f"run{run}": (
            f"timeVAE/carparts_X_51_run{run}/"
            "vae_dense_carparts_X_51_prior_samples.npz"
        )
        for run in range(1, 4)
    },
    "vae_conv": {
        f"run{run}": (
            f"timeVAE/carparts_X_51_run{run}/"
            "vae_conv_carparts_X_51_prior_samples.npz"
        )
        for run in range(1, 4)
    },
    "zits-gan": {
        f"run{run}": f"zits/run{run}/zits_gan_generated_data.npz"
        for run in range(1, 4)
    },
    "zits-vae": {
        f"run{run}": f"zits/run{run}/zits_vae_generated_data.npz"
        for run in range(1, 4)
    },
}


def _load_time_series(path, expected_timesteps):
    with np.load(path, allow_pickle=True) as archive:
        if "data" not in archive:
            raise ValueError(f"{path} does not contain a 'data' array")
        data = np.asarray(archive["data"], dtype=np.float64)

    if data.ndim == 3 and data.shape[-1] == 1:
        data = data[..., 0]
    if data.ndim != 2:
        raise ValueError(f"Expected data with 2 dimensions, got shape {data.shape}")
    if data.shape[1] != expected_timesteps:
        raise ValueError(
            f"Expected {expected_timesteps} carparts timesteps, got shape {data.shape}"
        )
    return np.nan_to_num(data)


def _save_checkpoint(records):
    output_dir = os.path.dirname(OUTPUT_CSV)
    os.makedirs(output_dir, exist_ok=True)
    columns = list(dict.fromkeys(
        column for record in records for column in record
    ))
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        suffix=".tmp",
        dir=output_dir,
        delete=False,
    ) as temporary_file:
        temporary_path = temporary_file.name
        writer = csv.DictWriter(temporary_file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)
    os.replace(temporary_path, OUTPUT_CSV)


def _read_checkpoint():
    if not os.path.exists(OUTPUT_CSV):
        return []
    with open(OUTPUT_CSV, "r", encoding="utf-8", newline="") as checkpoint:
        return list(csv.DictReader(checkpoint))


def _record_result(records, result):
    records[:] = [
        record
        for record in records
        if (record.get("model"), record.get("run"))
        != (result["model"], result["run"])
    ]
    records.append(result)
    _save_checkpoint(records)


def main():
    real_data = _load_time_series(REAL_DATA_PATH, expected_timesteps=51)
    records = _read_checkpoint()
    completed = {
        (record.get("model"), record.get("run"))
        for record in records
        if record.get("status") == "complete"
    }
    total_runs = sum(len(runs) for runs in RUN_FILES.values())
    run_number = 0

    for model, runs in RUN_FILES.items():
        for run, relative_path in runs.items():
            run_number += 1
            key = (model, run)
            if key in completed:
                print(f"[{run_number}/{total_runs}] Skipping completed {model} {run}")
                continue

            source_path = os.path.join(ROOT_DIR, "results_cp", relative_path)
            print(f"[{run_number}/{total_runs}] Evaluating {model} {run}: {relative_path}")
            result = {
                "model": model,
                "run": run,
                "status": "failed",
                "source_file": relative_path,
                "error": "",
            }
            try:
                synthetic_data = _load_time_series(
                    source_path, expected_timesteps=real_data.shape[1]
                )
                result.update(calculate_evaluation_metrics(real_data, synthetic_data, iterations=1000, batch_size=64))
                result["status"] = "complete"
            except Exception as error:
                result["error"] = f"{type(error).__name__}: {error}"
                print(f"  ERROR: {result['error']}")

            _record_result(records, result)
            if result["status"] == "complete":
                completed.add(key)
                print(f"  Saved results to {OUTPUT_CSV}")

    failures = [
        record for record in records if record.get("status") != "complete"
    ]
    configured_runs = {
        (model, run) for model, runs in RUN_FILES.items() for run in runs
    }
    completed_count = len(completed & configured_runs)
    print(f"Finished: {completed_count}/{total_runs} runs complete.")
    if failures:
        print(f"{len(failures)} run(s) failed and will be retried next time:")
        for record in failures:
            print(f"  {record['model']} {record['run']}: {record['error']}")
    print(f"CSV: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
