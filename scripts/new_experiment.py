"""Scaffold a new experiment directory under experiments/.

Usage:
    python scripts/new_experiment.py <experiment_name>
"""
import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"

NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")

# Subdirectories created empty (aside from a .gitkeep where needed to survive
# as a tracked-empty dir in git). configs/ (and Prototype/configs/) are
# intentionally left empty too -- parameter bounds/values are
# experiment-specific physics, authored by hand.
EMPTY_DIRS = [
    "configs", "data/csvs", "data/plots", "logs", "matlab_code", "Results",
    "Prototype/configs", "Prototype/data/csvs", "Prototype/Results",
    "Surrogate/data/csvs", "Surrogate/data/plots", "Surrogate/Results",
]
GITKEEP_DIRS = ["data/csvs", "data/plots", "logs", "matlab_code"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", help="Name of the new experiment (used as the directory name)")
    args = parser.parse_args()

    name = args.name
    if not NAME_RE.fullmatch(name):
        sys.exit(f"ERROR: invalid experiment name '{name}' -- use only letters, digits, '_' and '-'.")

    exp_dir = REPO_ROOT / "experiments" / name
    if exp_dir.exists():
        sys.exit(f"ERROR: {exp_dir} already exists; aborting to avoid overwriting an existing experiment.")

    for sub in EMPTY_DIRS:
        (exp_dir / sub).mkdir(parents=True)

    for sub in GITKEEP_DIRS:
        (exp_dir / sub / ".gitkeep").touch()

    main_py = (TEMPLATES_DIR / "main.py.template").read_text()
    (exp_dir / "main.py").write_text(main_py)

    sub_opt_job = (TEMPLATES_DIR / "sub_opt.job.template").read_text().replace("<name>", name)
    (exp_dir / "sub_opt.job").write_text(sub_opt_job)

    prototype_main_py = (TEMPLATES_DIR / "prototype_main.py.template").read_text()
    (exp_dir / "Prototype" / "main.py").write_text(prototype_main_py)

    sub_test_job = (TEMPLATES_DIR / "sub_test.job.template").read_text().replace("<name>", name)
    (exp_dir / "Prototype" / "sub_test.job").write_text(sub_test_job)

    surrogate_main_py = (TEMPLATES_DIR / "surrogate_main.py.template").read_text()
    (exp_dir / "Surrogate" / "test_surrogate.py").write_text(surrogate_main_py)

    sub_surrogate_job = (TEMPLATES_DIR / "sub_surrogate.job.template").read_text().replace("<name>", name)
    (exp_dir / "Surrogate" / "sub_surrogate.job").write_text(sub_surrogate_job)

    print(f"Created {exp_dir}")
    print("Next steps:")
    print(f"  - Author {exp_dir / 'configs' / 'default.yml'} (parameter bounds for optimization/surrogate)")
    print(f"  - Author {exp_dir / 'Prototype' / 'configs' / '<config>.yml'} (explicit prototype parameter values)")
    print(f"  - Populate {exp_dir / 'matlab_code'} with RCWA/FDTD simulation code")
    print(f"  - Submit optimization with: cd {exp_dir.relative_to(REPO_ROOT)} && sbatch sub_opt.job")
    print(f"  - Submit prototype with: cd {exp_dir.relative_to(REPO_ROOT)}/Prototype && sbatch sub_test.job <config>.yml")
    print(f"  - Submit surrogate test with: cd {exp_dir.relative_to(REPO_ROOT)}/Surrogate && sbatch sub_surrogate.job")


if __name__ == "__main__":
    main()
