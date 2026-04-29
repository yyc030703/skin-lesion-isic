"""
Generate stratified train/val/test splits for ISIC 2018 Task 3 (classification).
Reads the official training ground truth CSV, converts one-hot encoding to
single-label, and writes three CSVs to data/splits/.
"""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# Paths — relative to project root
DATA_DIR = Path("data")
LABELS_CSV = DATA_DIR / "ISIC2018_Task3_Training_GroundTruth" / "ISIC2018_Task3_Training_GroundTruth.csv"
IMAGES_DIR = DATA_DIR / "ISIC2018_Task3_Training_Input"
SPLITS_DIR = DATA_DIR / "splits"
SPLITS_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
SEED = 42

def main():
    # Load and convert one-hot labels to a single integer label
    df = pd.read_csv(LABELS_CSV)
    df["label"] = df[CLASS_NAMES].values.argmax(axis=1)
    df["class_name"] = df["label"].apply(lambda i: CLASS_NAMES[i])
    df["image_path"] = df["image"].apply(lambda x: str(IMAGES_DIR / f"{x}.jpg"))

    df = df[["image", "image_path", "label", "class_name"]]

    # Stratified split: 70% train, 15% val, 15% test
    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df["label"], random_state=SEED
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["label"], random_state=SEED
    )

    train_df.to_csv(SPLITS_DIR / "train.csv", index=False)
    val_df.to_csv(SPLITS_DIR / "val.csv", index=False)
    test_df.to_csv(SPLITS_DIR / "test.csv", index=False)

    print(f"Total images: {len(df)}")
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    print("\nClass distribution per split:")
    for name, split in [("train", train_df), ("val", val_df), ("test", test_df)]:
        counts = split["class_name"].value_counts().sort_index()
        print(f"\n{name}:\n{counts}")

if __name__ == "__main__":
    main()