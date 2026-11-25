"""The GAN stage's datasets: DataComp-1B for training, ImageNet-1K validation for eval."""

from datasets import load_dataset
from torchvision import transforms
from torchvision.transforms import v2

from .crop import center_crop_arr
from .data_paths import dataset_path
from .datacomp_dataset import build_datacomp_train_dataset

DATACOMP_PATH = dataset_path("datacomp-1b")
IMAGENET_PATH = dataset_path("imagenet-1k")


def build_datasets(resolution=256):
    """-> (train, eval).

    train: DataComp-1B, a random resized crop (scale 0.08-1, bicubic) to `resolution` and a horizontal flip.
    eval: ImageNet-1K validation, centre-cropped to `resolution`, with `img_labels`.
    """
    train_transform = v2.Compose(
        [
            v2.RandomResizedCrop(
                size=resolution,
                scale=(0.08, 1.0),
                interpolation=v2.InterpolationMode.BICUBIC,
                antialias=True,
            ),
            v2.RandomHorizontalFlip(),
            v2.ToTensor(),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Lambda(lambda img: center_crop_arr(img, resolution)),
            transforms.ToTensor(),
        ]
    )

    train_ds = build_datacomp_train_dataset(data_dir=DATACOMP_PATH, train_transform=train_transform)

    raw_eval_ds = load_dataset(IMAGENET_PATH, split="validation")

    def process_eval(examples):
        examples["pixel_values"] = [eval_transform(img.convert("RGB")) for img in examples["image"]]
        examples["img_labels"] = examples["label"]
        return examples

    eval_ds = raw_eval_ds.with_transform(process_eval)

    return train_ds, eval_ds
