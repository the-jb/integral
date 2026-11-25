import numpy as np
from PIL import Image


def center_crop_arr(pil_image, image_size):
    """A PIL image -> its `image_size` square centre crop: halved with a box filter while the short side is at least
    twice `image_size`, resized bicubic to a short side of `image_size`, then cropped (guided-diffusion's
    `center_crop_arr`)."""
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])
