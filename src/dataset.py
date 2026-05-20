import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from sklearn.model_selection import train_test_split


IMAGE_SIZE  = 64
CROP_SIZE   = 224
NUM_CLASSES = 8

CLASS_NAMES = [
    "Round Elliptical",
    "In-Between Elliptical",
    "Cigar Elliptical",
    "Edge-On Disc",
    "Barred Spiral",
    "Unbarred Spiral",
    "Irregular",
    "Merger",
]


def assign_class(row):
    if row["Class8.7"] >= 0.05:
        return 7
    if row["Class6.1"] >= 0.50:
        return 6
    if row["Class4.1"] >= 0.50:
        return 4
    if row["Class4.2"] >= 0.50:
        return 5
    if row["Class2.1"] >= 0.50:
        return 3
    if row["Class7.1"] >= 0.50:
        return 2
    if row["Class1.2"] >= 0.469:
        return 1
    if row["Class1.1"] >= 0.469:
        return 0
    return -1


def load_labels(csv_path, img_dir):
    df = pd.read_csv(csv_path)
    print(f"Loaded CSV: {len(df)} total galaxies")

    df["class_id"] = df.apply(assign_class, axis=1)
    df = df[df["class_id"] >= 0].copy()
    print(f"After thresholding: {len(df)} labelled galaxies")

    df["filepath"] = df["GalaxyID"].astype(str).apply(
        lambda gid: os.path.join(img_dir, gid + ".jpg")
    )
    df = df[df["filepath"].apply(os.path.exists)].copy()
    df = df.reset_index(drop=True)
    print(f"After image matching: {len(df)} usable samples")

    print("\nClass distribution:")
    for cid, cname in enumerate(CLASS_NAMES):
        count = (df["class_id"] == cid).sum()
        pct   = count / len(df) * 100
        bar   = "#" * int(pct / 2)
        print(f"  {cid} {cname:<25} {count:>6}  ({pct:.1f}%)  {bar}")

    return df


def get_transforms(split="train", image_size=IMAGE_SIZE, crop_size=CROP_SIZE):
    normalise = transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std =[0.5, 0.5, 0.5]
    )
    if split == "train":
        return transforms.Compose([
            transforms.CenterCrop(crop_size),
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomApply([transforms.RandomRotation(degrees=(90,  90))],  p=0.33),
            transforms.RandomApply([transforms.RandomRotation(degrees=(180, 180))], p=0.33),
            transforms.RandomApply([transforms.RandomRotation(degrees=(270, 270))], p=0.33),
            transforms.ColorJitter(brightness=0.05),
            transforms.ToTensor(),
            normalise,
        ])
    else:
        return transforms.Compose([
            transforms.CenterCrop(crop_size),
            transforms.Resize(image_size),
            transforms.ToTensor(),
            normalise,
        ])


class GZ2Dataset(Dataset):
    def __init__(self, df, transform=None):
        self.df        = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        label = int(row["class_id"])
        img   = Image.open(row["filepath"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def make_weighted_sampler(df):
    class_counts   = df["class_id"].value_counts().sort_index()
    class_weights  = 1.0 / class_counts
    sample_weights = torch.DoubleTensor(
        df["class_id"].map(class_weights).values
    )
    return WeightedRandomSampler(
        weights     = sample_weights,
        num_samples = len(sample_weights),
        replacement = True
    )


def get_dataloaders(
    csv_path,
    img_dir,
    batch_size  = 64,
    image_size  = IMAGE_SIZE,
    crop_size   = CROP_SIZE,
    num_workers = 2,
    use_sampler = True,
):
    df = load_labels(csv_path, img_dir)

    valid_classes = df["class_id"].value_counts()
    valid_classes = valid_classes[valid_classes >= 10].index
    df = df[df["class_id"].isin(valid_classes)].copy().reset_index(drop=True)

    df_train, df_temp = train_test_split(
        df, test_size=0.36, stratify=df["class_id"], random_state=42
    )
    df_val, df_test = train_test_split(
        df_temp, test_size=0.556, stratify=df_temp["class_id"], random_state=42
    )

    print(f"\nSplit sizes:")
    print(f"  Train : {len(df_train)}")
    print(f"  Val   : {len(df_val)}")
    print(f"  Test  : {len(df_test)}")

    train_tf = get_transforms("train", image_size, crop_size)
    eval_tf  = get_transforms("val",   image_size, crop_size)

    train_ds = GZ2Dataset(df_train, transform=train_tf)
    val_ds   = GZ2Dataset(df_val,   transform=eval_tf)
    test_ds  = GZ2Dataset(df_test,  transform=eval_tf)

    if use_sampler:
        sampler      = make_weighted_sampler(df_train)
        train_loader = DataLoader(
            train_ds, batch_size=batch_size,
            sampler=sampler, num_workers=num_workers, pin_memory=True
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=batch_size,
            shuffle=True, num_workers=num_workers, pin_memory=True
        )

    val_loader = DataLoader(
        val_ds, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader
