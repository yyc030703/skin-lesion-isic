"""Quick smoke test for the Dataset classes — make sure they load correctly."""

from data import ISICClassificationDataset, ISICSegmentationDataset

# Classification
clf_ds = ISICClassificationDataset("data/splits/train.csv")
print(f"Classification dataset: {len(clf_ds)} samples")
img, label = clf_ds[0]
print(f"  Image shape: {img.shape}, label: {label}")

# Segmentation
seg_ds = ISICSegmentationDataset(
    images_dir="data/ISIC2018_Task1-2_Training_Input",
    masks_dir="data/ISIC2018_Task1_Training_GroundTruth",
)
print(f"\nSegmentation dataset: {len(seg_ds)} samples")
img, mask = seg_ds[0]
print(f"  Image shape: {img.shape}")
print(f"  Mask shape: {mask.shape}, unique values: {np.unique(mask)}")