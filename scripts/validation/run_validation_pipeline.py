#!/usr/bin/env python
"""Short driver: runs the real numbered pipeline end-to-end for a given
--level / --subsample_size, as a smoke test (e.g. celltype.l1 + a 10k subsample).

Each stage below is the actual production script (or its dev-only bridge, for
stage 1, when no raw GEO data is present) -- nothing here reimplements their
logic; this just calls them in the right order with consistent arguments.

    seed_full_dataset_from_h5mu.py  (bridge; used unless --use_real_loader is
                                     passed, in which case stage 1 is the real
                                     1_load_full_dataset.py -- requires the raw
                                     GEO archive, see config.RAW_ARCHIVE_PATH)
    2_create_subsample_datasets.py
    3_exploratory_analysis.py
    4_split_dataset.py
    5_feature_selection.py
    deep_learning/2_tune.py         (hyperparameter search)
    deep_learning/1_train.py        (trains the MLP; needed before ig_mlp scoring)
    6_ground_truth.py
    7_scoring.py  (once per method: spearman, partial_corr, mi_ksg, ig_mlp)

Run:  python scripts/validation/run_validation_pipeline.py --level celltype.l1 \\
          --subsample_size 10000
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from src import config

SCRIPTS_DIR = Path(__file__).resolve().parent.parent
DEEP_LEARNING_DIR = SCRIPTS_DIR / "deep_learning"
REPO_ROOT = SCRIPTS_DIR.parent


def run(script_dir: Path, script_name: str, cli_args: list[str]):
    cmd = [sys.executable, str(script_dir / script_name), *cli_args]
    print(f"\n>>> {' '.join(cmd[1:])}", flush=True)
    subprocess.run(cmd, cwd=script_dir, check=True)


def main(args):
    level = args.level
    subsample_size = args.subsample_size
    seed = args.seed
    test_split_size = args.test_split_size
    n_epochs = args.n_epochs
    n_trials = args.n_trials
    h5mu_path = args.h5mu_path
    root_dir = args.root_dir
    force_recreate = args.force_recreate

    shared = ["--level", level, "--seed", str(seed)]
    subsampled = shared + ["--subsample_size", str(subsample_size)]
    split = subsampled + ["--test_split_size", str(test_split_size)]

    # 1. Full dataset. Real loader (parses the raw GEO archive) when
    #    --use_real_loader is passed and the archive is present; otherwise the
    #    seed-from-h5mu dev bridge, for checkouts with no raw GEO data.
    if args.use_real_loader:
        run(SCRIPTS_DIR, "1_load_full_dataset.py",
            ["--level", level, "--force_recreate", str(force_recreate)])
    else:
        seed_args = ["--level", level, "--force_recreate", str(force_recreate)]
        if h5mu_path:  # else let the bridge fall back to its own VALIDATION_H5MU default
            seed_args += ["--h5mu_path", h5mu_path]
        run(SCRIPTS_DIR / "validation", "seed_full_dataset_from_h5mu.py", seed_args)

    # 2. Subsample
    run(SCRIPTS_DIR, "2_create_subsample_datasets.py", subsampled)

    # 3. Exploratory analysis
    run(SCRIPTS_DIR, "3_exploratory_analysis.py",
        subsampled + ["--root_dir", str(root_dir)])

    # 4. Split (stratified)
    run(SCRIPTS_DIR, "4_split_dataset.py", split)

    # 5. Feature selection (HVG U markers, training split only)
    run(SCRIPTS_DIR, "5_feature_selection.py", split)

    # deep_learning: tune (draw the tuning subsample from THIS split, so a
    # separate full-dataset split is not required for a small/dev run) then train
    run(DEEP_LEARNING_DIR, "2_tune.py",
        shared + ["--test_split_size", str(test_split_size),
                  "--subsample_size", str(subsample_size),
                  "--n_trials", str(n_trials)])
    run(DEEP_LEARNING_DIR, "1_train.py",
        split + ["--n_epochs", str(n_epochs), "--root_dir", str(root_dir)])

    # 6. Ground truth (independent of training/scoring; needs stage 5's universe)
    run(SCRIPTS_DIR, "6_ground_truth.py",
        split + ["--root_dir", str(root_dir)])

    # 7. Scoring, once per method
    for method in ("spearman", "partial_corr", "mi_ksg", "ig_mlp"):
        run(SCRIPTS_DIR, "7_scoring.py",
            split + ["--method", method, "--root_dir", str(root_dir)])

    # Metadata the report notebook reads (level/subsample/split/seed/methods),
    # so it never has to hardcode this run's configuration.
    json.dump({"level": level, "subsample_size": subsample_size,
               "test_split_size": test_split_size, "seed": seed,
               "methods": ["spearman", "partial_corr", "mi_ksg", "ig_mlp"]},
              open(root_dir / "run_metadata.json", "w"), indent=1)

    print(f"\nDone. Repo-local artifacts (exploratory, MLP diagnostics, ground "
         f"truth, scores) are under {root_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=str, required=True)
    parser.add_argument("--subsample_size", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_split_size", type=int, default=40)
    parser.add_argument("--n_epochs", type=int, default=20)
    parser.add_argument("--n_trials", type=int, default=config.N_TRIALS,
                        help="Optuna trials for hyperparameter tuning (defaults "
                             "to the project's own config.N_TRIALS). Both the "
                             "sampler and each trial's model are now seeded "
                             "(see tuning.py), so a low trial count under-explores "
                             "the search space deterministically rather than "
                             "randomly -- pass a smaller value only for a "
                             "deliberately quick, non-representative smoke test.")
    parser.add_argument("--h5mu_path", type=str,
                        default="", required=False,
                        help="Passed to seed_full_dataset_from_h5mu.py; falls "
                             "back to its own VALIDATION_H5MU env var default "
                             "if omitted. Ignored when --use_real_loader is set.")
    parser.add_argument("--use_real_loader", action="store_true",
                        help="Stage 1 calls the real 1_load_full_dataset.py "
                             "(parses the raw GEO archive) instead of the "
                             "seed_full_dataset_from_h5mu.py dev bridge. "
                             "Requires the raw archive to be present "
                             "(see config.RAW_ARCHIVE_PATH).")
    parser.add_argument("--root_dir", type=str, default=None,
                        help="Repo-local output root for this run's exploratory/"
                             "ground-truth/scores/MLP-diagnostics artifacts. "
                             "Defaults to local_data/validation_<level>_<n>/.")
    parser.add_argument("--force_recreate", type=bool, default=False)
    parsed_args = parser.parse_args()

    if not parsed_args.root_dir:
        level_slug = parsed_args.level.replace(".", "")
        n_slug = (str(parsed_args.subsample_size)
                 if parsed_args.subsample_size > 0 else "full")
        parsed_args.root_dir = REPO_ROOT / "local_data" / f"validation_{level_slug}_{n_slug}"
    else:
        parsed_args.root_dir = Path(parsed_args.root_dir)
    parsed_args.root_dir.mkdir(parents=True, exist_ok=True)

    main(parsed_args)
