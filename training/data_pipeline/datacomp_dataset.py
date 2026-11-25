"""DataComp-1B images as a webdataset stream (the GAN stage's training images)."""

import os

import webdataset as wds


def build_datacomp_train_dataset(data_dir, train_transform):
    """Shards `<data_dir>/shards/00000000.tar .. 00059687.tar`, resampled, split by node, shuffled in a 1,000-sample
    buffer -> `{"pixel_values": train_transform(image)}`."""

    def preprocess_train(sample):
        return {"pixel_values": train_transform(sample["jpg"].convert("RGB"))}

    trainset_url = os.path.join(data_dir, "shards", "{00000000..00059687}.tar")

    train_ds = (
        wds.WebDataset(
            trainset_url,
            resampled=True,
            nodesplitter=wds.split_by_node,
        )
        .shuffle(1000)
        .decode("pil")
        .map(preprocess_train)
    )

    return train_ds
