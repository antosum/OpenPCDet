from pathlib import Path
import argparse
import yaml
from easydict import EasyDict

from pcdet.datasets.custom.custom_dataset import create_custom_infos


def main():
    parser = argparse.ArgumentParser(
        description="Generate CustomDataset info files for Geminai dataset"
    )
    parser.add_argument(
        "--cfg",
        type=str,
        default="tools/cfgs/dataset_configs/geminai_dataset.yaml",
        help="Path to dataset YAML used only to supply encoding/classes mapping",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/home/antonin-sumner/data/openpcdet_data/data/geminai_V0",
        help="Absolute path to dataset root (contains ImageSets/points/labels)",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="Directory to write output pkl files (defaults to data_path)",
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of threads for info gen"
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=["belt_loader", "high_loader", "Other_vehicle"],
        help="Class names to include",
    )

    args = parser.parse_args()

    cfg_path = Path(args.cfg)
    with open(cfg_path, "r") as f:
        dataset_cfg = EasyDict(yaml.safe_load(f))

    data_path = Path(args.data_path)
    save_path = Path(args.save_path) if args.save_path is not None else data_path

    print("Generating infos with settings:")
    print(f"  cfg:        {cfg_path}")
    print(f"  data_path:  {data_path}")
    print(f"  save_path:  {save_path}")
    print(f"  classes:    {args.classes}")
    print(f"  workers:    {args.workers}")

    # Basic structure checks to help early failures
    expected = [data_path / "ImageSets" / "train.txt", data_path / "ImageSets" / "val.txt"]
    for p in expected:
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    create_custom_infos(
        dataset_cfg=dataset_cfg,
        class_names=list(args.classes),
        data_path=data_path,
        save_path=save_path,
        workers=args.workers,
    )

    print("Done. Expected outputs:")
    print(f"  {save_path / 'custom_infos_train.pkl'}")
    print(f"  {save_path / 'custom_infos_val.pkl'}")
    print(f"  {save_path / 'custom_dbinfos_train.pkl'} (for gt sampling)")


if __name__ == "__main__":
    main()

