import argparse
import json
import os

import pandas as pd


SPLITS = ["train", "dev", "test"]

EVENT_TO_DISASTER = {
    "hurricane_irma": "hurricane",
    "hurricane_harvey": "hurricane",
    "hurricane_maria": "hurricane",
    "california_wildfires": "wildfire",
    "mexico_earthquake": "earthquake",
    "iraq_iran_earthquake": "earthquake",
    "srilanka_floods": "flood",
}

SOURCES = {
    "task1a": "informative",
    "task1b": "informative",
    "task2": "humanitarian",
    "task3": "damage",
}


def load(split_dir, stem, split):
    path = os.path.join(
        split_dir,
        f"task_{stem}_text_img_{split}.tsv"
    )

    print(f"  Reading: {path}")

    df = pd.read_csv(path, sep="\t")

    return df


def build(task, df, root):
    print(f"  Raw rows: {len(df)}")

    # ---------------------------------------------------------
    # Task 1(a), Task 1(b), and Task 2:
    # keep only samples where text and image labels agree
    # ---------------------------------------------------------
    if task in ["task1a", "task1b", "task2"]:

        if {"label_text", "label_image"}.issubset(df.columns):

            before = len(df)

            df = df[
                df["label_text"] == df["label_image"]
            ].copy()

            print(
                f"  Agreement filter: "
                f"{before} -> {len(df)} rows"
            )

        else:
            raise KeyError(
                f"{task} requires label_text and label_image columns."
            )

    # ---------------------------------------------------------
    # Build common output
    # ---------------------------------------------------------
    out = pd.DataFrame()

    out["text"] = df["tweet_text"].fillna("")

    # TSV image paths are relative to CrisisMMD_v2.0
    out["image_path"] = df["image"].apply(
        lambda p: os.path.normpath(
            os.path.join(root, str(p))
        )
    )

    # ---------------------------------------------------------
    # Task 1(a)
    # Informative / not_informative
    # ---------------------------------------------------------
    if task == "task1a":

        out["label_name"] = df["label"]

    # ---------------------------------------------------------
    # Task 1(b)
    #
    # Informative samples become:
    # hurricane / wildfire / earthquake / flood
    #
    # Non-informative samples remain:
    # non_informative
    # ---------------------------------------------------------
    elif task == "task1b":

        def disaster_label(row):
            event = str(row["event_name"])

            original_label = str(row["label"])

            if original_label == "not_informative":
                return "non_informative"

            if event not in EVENT_TO_DISASTER:
                raise ValueError(
                    f"Unknown event name for Task 1(b): {event}"
                )

            return EVENT_TO_DISASTER[event]

        out["label_name"] = df.apply(
            disaster_label,
            axis=1
        )

    # ---------------------------------------------------------
    # Task 2
    # Humanitarian categories
    # ---------------------------------------------------------
    elif task == "task2":

        out["label_name"] = df["label"]

    # ---------------------------------------------------------
    # Task 3
    #
    # Use the image/damage label supplied by the official
    # damage split.
    # ---------------------------------------------------------
    elif task == "task3":

        if "label_image" in df.columns:
            out["label_name"] = df["label_image"]
        else:
            out["label_name"] = df["label"]

    # ---------------------------------------------------------
    # Check image files
    # ---------------------------------------------------------
    exists = out["image_path"].apply(os.path.isfile)

    missing = (~exists).sum()

    if missing > 0:
        print(
            f"  [{task}] dropped {missing} rows "
            f"because image files were missing"
        )

    out = out[exists].copy()

    # Remove rows with missing text/image/label
    out = out.dropna(
        subset=["text", "image_path", "label_name"]
    )

    out = out.reset_index(drop=True)

    print(f"  Final rows: {len(out)}")

    return out


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        default="data/CrisisMMD_v2.0"
    )

    parser.add_argument(
        "--out",
        default="data/processed"
    )

    args = parser.parse_args()

    root = os.path.normpath(args.root)

    split_dir = os.path.join(
        root,
        "crisismmd_datasplit_all"
    )

    os.makedirs(args.out, exist_ok=True)

    stats = []

    # ---------------------------------------------------------
    # Process all four tasks
    # ---------------------------------------------------------
    for task, stem in SOURCES.items():

        print()
        print("=" * 60)
        print(f"{task}")
        print("=" * 60)

        raw = {
            split: load(
                split_dir,
                stem,
                split
            )
            for split in SPLITS
        }

        # Print columns once for verification
        print(
            "  Columns:",
            list(raw["train"].columns)
        )

        built = {
            split: build(
                task,
                raw[split],
                root
            )
            for split in SPLITS
        }

        # -----------------------------------------------------
        # Create output directory
        # -----------------------------------------------------
        task_dir = os.path.join(
            args.out,
            task
        )

        os.makedirs(
            task_dir,
            exist_ok=True
        )

        # -----------------------------------------------------
        # Create label map using training labels
        # -----------------------------------------------------
        labels = sorted(
            built["train"]["label_name"]
            .unique()
        )

        label_map = {
            label: index
            for index, label in enumerate(labels)
        }

        label_map_path = os.path.join(
            task_dir,
            "label_map.json"
        )

        with open(
            label_map_path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                label_map,
                f,
                indent=2
            )

        print(
            "  Labels:",
            label_map
        )

        # -----------------------------------------------------
        # Save train/dev/test CSVs
        # -----------------------------------------------------
        for split, df in built.items():

            df = df.copy()

            df["label"] = (
                df["label_name"]
                .map(label_map)
            )

            output_path = os.path.join(
                task_dir,
                f"{split}.csv"
            )

            df.to_csv(
                output_path,
                index=False
            )

            # Stats
            counts = (
                df["label_name"]
                .value_counts()
            )

            for label_name, count in counts.items():

                stats.append(
                    {
                        "task": task,
                        "split": split,
                        "label": label_name,
                        "count": int(count),
                    }
                )

            print(
                f"  {split}: {len(df)} samples"
            )

    # ---------------------------------------------------------
    # Save statistics
    # ---------------------------------------------------------
    stats_df = pd.DataFrame(stats)

    stats_path = os.path.join(
        args.out,
        "stats.csv"
    )

    stats_df.to_csv(
        stats_path,
        index=False
    )

    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)
    print(
        f"Saved statistics to: {stats_path}"
    )


if __name__ == "__main__":
    main()