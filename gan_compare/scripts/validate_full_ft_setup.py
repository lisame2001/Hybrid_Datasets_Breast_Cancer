import argparse
import csv
import hashlib
from pathlib import Path

import yaml


CONFIG_DIR = Path("gan_compare/configs/swin/hybrid_experiments/full_ft")
SEEDS = range(42, 47)
CONDITIONS = {
    "real_only": (True, False, 0),
    "hybrid_025": (True, True, 276),
    "hybrid_050": (True, True, 552),
    "hybrid_100": (True, True, 1104),
    "hybrid_200": (True, True, 2208),
    "hybrid_250": (True, True, 2760),
    "synthetic_only_matched": (False, True, 1104),
    "synthetic_only_maximum": (False, True, 3000),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the full fine-tuning configuration matrix and data layout."
    )
    parser.add_argument("--dataset-path", type=Path, default=Path("dataset16062024"))
    parser.add_argument(
        "--synthetic-data-dir",
        type=Path,
        default=Path("extension/synthetic_data/cbis-ddsm/c-dcgan"),
    )
    parser.add_argument(
        "--configs-only",
        action="store_true",
        help="Validate YAML files without requiring local image data.",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=Path("setup/paper_data_manifest.csv"),
        help="CSV manifest containing paths and SHA-256 checksums.",
    )
    parser.add_argument(
        "--verify-checksums",
        action="store_true",
        help="Verify every local image against the committed data manifest.",
    )
    return parser.parse_args()


def count_pngs(path: Path) -> int:
    return sum(1 for item in path.iterdir() if item.is_file() and item.suffix.lower() == ".png")


def validate_configs() -> None:
    checked = 0
    for condition, expected in CONDITIONS.items():
        expected_real, expected_synthetic, expected_count = expected
        for seed in SEEDS:
            path = CONFIG_DIR / f"{condition}_seed{seed}.yaml"
            if not path.is_file():
                raise FileNotFoundError(f"Missing configuration: {path}")

            config = yaml.safe_load(path.read_text())
            observed = (
                config.get("use_real"),
                config.get("use_synthetic"),
                config.get("synthetic_count"),
            )
            if observed != expected:
                raise ValueError(f"Unexpected condition values in {path}: {observed}, expected {expected}")
            if config.get("synthetic_sampling_seed") != seed:
                raise ValueError(f"Synthetic sampling seed mismatch in {path}")
            if config.get("num_epochs") != 300 or config.get("batch_size") != 64:
                raise ValueError(f"Training protocol mismatch in {path}")
            checked += 1

    print(f"Validated {checked} YAML configurations.")


def validate_real_data(dataset_path: Path) -> None:
    expected = {
        ("cbis-ddsm", "train"): {
            "is_benign_false": 525,
            "is_benign_true": 579,
        },
        ("cbis-ddsm", "val"): {
            "is_benign_false": 102,
            "is_benign_true": 90,
        },
        ("cbis-ddsm", "test"): {
            "is_benign_false": 157,
            "is_benign_true": 245,
        },
        ("bcdr", "test"): {
            "is_benign_false": 486,
            "is_benign_true": 620,
        },
    }

    for (dataset, subset), expected_classes in expected.items():
        subset_dir = dataset_path / dataset / subset
        observed_total = 0

        for class_name, expected_count in expected_classes.items():
            class_dir = subset_dir / class_name
            if not class_dir.is_dir():
                raise FileNotFoundError(f"Missing class directory: {class_dir}")

            observed_count = count_pngs(class_dir)
            if observed_count != expected_count:
                raise ValueError(
                    f"Unexpected image count for {dataset}/{subset}/{class_name}: "
                    f"{observed_count}, expected {expected_count}"
                )

            observed_total += observed_count

        print(
            f"Validated {dataset}/{subset}: {observed_total} PNG files "
            f"({expected_classes['is_benign_false']} malignant, "
            f"{expected_classes['is_benign_true']} benign)."
        )


def validate_synthetic_data(path: Path) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"Missing synthetic-data directory: {path}")
    names = sorted(item.name.lower() for item in path.iterdir() if item.suffix.lower() == ".png")
    malignant = sum("malignant" in name for name in names)
    benign = sum("benign" in name and "malignant" not in name for name in names)
    if (malignant, benign) != (1500, 1500):
        raise ValueError(
            f"Unexpected synthetic class counts: malignant={malignant}, benign={benign}; expected 1500 each"
        )
    print("Validated synthetic pool: 1,500 malignant and 1,500 benign PNG files.")



def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_manifest(manifest_path: Path) -> None:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing data manifest: {manifest_path}")

    checked = 0
    with manifest_path.open(newline="") as handle:
        reader = csv.DictReader(handle)

        for row in reader:
            path = Path(row["relative_path"])

            if not path.is_file():
                raise FileNotFoundError(f"Manifest file is missing locally: {path}")

            observed_size = path.stat().st_size
            expected_size = int(row["size_bytes"])
            if observed_size != expected_size:
                raise ValueError(
                    f"Size mismatch for {path}: "
                    f"{observed_size}, expected {expected_size}"
                )

            observed_hash = file_sha256(path)
            if observed_hash != row["sha256"]:
                raise ValueError(f"SHA-256 mismatch for {path}")

            checked += 1

    if checked != 5804:
        raise ValueError(
            f"Unexpected manifest length: {checked}, expected 5804"
        )

    print(f"Validated SHA-256 checksums for {checked} image files.")


def main() -> None:
    args = parse_args()
    validate_configs()
    if not args.configs_only:
        validate_real_data(args.dataset_path)
        validate_synthetic_data(args.synthetic_data_dir)
        if args.verify_checksums:
            validate_manifest(args.manifest_path)
    print("Full fine-tuning setup validation passed.")


if __name__ == "__main__":
    main()
