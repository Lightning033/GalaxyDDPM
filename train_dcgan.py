import sys
import torch

sys.path.append("src")

from dataset import get_dataloaders
from dcgan import DCGAN

CSV_PATH   = "data/training_solutions_rev1.csv"
IMG_DIR    = "data/images_training_rev1/"
SAVE_DIR   = "checkpoints/"

BATCH_SIZE = 64
N_EPOCHS   = 100
SAVE_EVERY = 10
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    print(f"Device: {DEVICE}")

    train_loader, _, _ = get_dataloaders(
        csv_path    = CSV_PATH,
        img_dir     = IMG_DIR,
        batch_size  = BATCH_SIZE,
        num_workers = 2,
    )

    dcgan = DCGAN(
        z_dim       = 128,
        num_classes = 8,
        device      = DEVICE,
        lr          = 2e-4,
        beta1       = 0.5,
        beta2       = 0.999,
    )

    dcgan.train_loop(
        train_loader = train_loader,
        n_epochs     = N_EPOCHS,
        save_dir     = SAVE_DIR,
        save_every   = SAVE_EVERY,
    )


if __name__ == "__main__":
    main()
