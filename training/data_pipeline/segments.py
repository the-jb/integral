"""The sample representation: segments, rendering to tokens, and one packed row.

A sample is a list of segments:

    Text(ids, supervise)        supervise -> cross-entropy on these positions
    Image(pixels, generate)     generate  -> a flow target here; either way context for what follows

    t2i          [Text(request, F), Image(target, T)]
    editing      [Image(source, F), Text(instruction, F), Image(target, T)]
    interleaved  [Text(question, F), Text(thought, T), Image(figure, T), Text(more, T), Image(figure2, T), ...]

`Text` segments arrive already tokenized, chat markup included. Images are uint8 pixels; the vision tokenizer encodes
them in the training loop. An image occupies the same span whether it is generated or given: `generate` decides only
whether a flow target and the generation marker go with it. Position ids are built by the model per sample
(`get_rope_index`), from the row's `cu_seqlens`.
"""

from dataclasses import dataclass, field

import torch

IGNORE = -100          # torch's cross_entropy ignore_index
# The linear-attention kernel's chunk length (`chunk_size=64` in fla's gated delta rule); see `collate`.
CHUNK = 64


@dataclass(frozen=True)
class Geometry:
    """How an image's size becomes token counts.

    `patch` is the vision tokenizer's patch size and `pool` its spatial merge size: an image of `h x w` pixels has
    `(h / patch) x (w / patch)` latents and `slots` = latents / pool**2 positions in the sequence.
    """

    patch: int = 16
    pool: int = 2

    def grid(self, h: int, w: int) -> tuple:
        assert h % self.patch == 0 and w % self.patch == 0, \
            f"{h}x{w} is not a multiple of patch {self.patch}"
        return h // self.patch, w // self.patch

    def slots(self, h: int, w: int) -> int:
        gh, gw = self.grid(h, w)
        assert gh % self.pool == 0 and gw % self.pool == 0, \
            f"grid {gh}x{gw} is not a multiple of the {self.pool}x{self.pool} pooling"
        return (gh // self.pool) * (gw // self.pool)


@dataclass(frozen=True)
class Marker:
    """The generation marker that opens a generated image.

    `prefix_ids(h, w)` is the marker's ids for an `h x w` image (`gen_marker`, ending on the plain-text
    `<vision_start>`). A target's condition ends after it. `supervise_open` makes the marker a cross-entropy target.
    A context image is opened by Qwen's `<|vision_start|>` instead (`render`).
    """

    prefix_ids: "Callable[[int, int], list]"  # noqa: F821
    supervise_open: bool = True


@dataclass
class Text:
    """Already-tokenized text. `supervise` puts these positions under cross-entropy."""

    ids: list
    supervise: bool = False


@dataclass
class Image:
    """One image as uint8 pixels `[3, H, W]`. `generate` attaches a flow target and the generation marker."""

    pixels: torch.Tensor
    generate: bool = False


@dataclass
class Rendered:
    """One sample flattened to tokens. Every index is local to the sample.

    `image_at` holds each image span's pixel index: a target whose span is cut off the end (`useful_len`) keeps its
    pixels (its flow target) but has no span.
    """

    ids: list
    labels: list                      # IGNORE everywhere but supervised text and the marker
    image_at: list                    # (slot_start, n_slots, pixel_index) per image that has slots
    pixels: list                      # uint8 [3,H,W] per image, sequence order
    gen_point: list                   # per generated image: index of its first slot
    gen_image: list                   # per generated image: its index into `pixels`
    source: str = "?"

    def __len__(self):
        return len(self.ids)

    @property
    def n_targets(self) -> int:
        return len(self.gen_point)


def render(sample, *, marker: Marker, image_token_id: int, vision_start_id: int,
           vision_end_id: int, geom: Geometry, source: str = "?",
           max_len: int = 0, text_loss: bool = True) -> Rendered:
    """A list of segments -> tokens, labels, and where the images and targets sit.

    A generated image is `[marker] [<|image_pad|> x slots] [<|vision_end|>]`, the marker supervised with
    `marker.supervise_open`. A context image is `[<|vision_start|>] [<|image_pad|> x slots] [<|vision_end|>]`,
    unsupervised. The slot count is per image, from its own size.

    A target's generation point is its first slot; its condition is `[sample start, gen_point)`. Labels are not
    shifted here (the loss shifts them).

    The sample is then cut to what is read (`useful_len`) and to `max_len` (the room left in the row, 0 = no cap).
    An image span that does not fit whole is dropped, and a target whose condition does not fit leaves with its
    pixels.

    Args:
        text_loss: False for a run without cross-entropy: the sample is cut after its last target's condition.
    """
    ids, labels, image_at, pixels, gen_point, gen_image = [], [], [], [], [], []
    for seg in sample:
        if isinstance(seg, Text):
            assert seg.ids, "an empty text segment"
            ids += list(seg.ids)
            labels += list(seg.ids) if seg.supervise else [IGNORE] * len(seg.ids)
            continue
        if not isinstance(seg, Image):
            raise TypeError(f"a sample holds {type(seg).__name__}, not Text or Image")

        assert seg.pixels.dim() == 3 and seg.pixels.shape[0] == 3, \
            f"an image is [3, H, W], got {tuple(seg.pixels.shape)}"
        assert seg.pixels.dtype == torch.uint8, \
            f"images cross the worker boundary as uint8, got {seg.pixels.dtype}"
        h, w = int(seg.pixels.shape[1]), int(seg.pixels.shape[2])
        n_slots = geom.slots(h, w)

        if seg.generate:
            pre = list(marker.prefix_ids(h, w))
            assert pre, "the generation marker is empty"
            ids += pre
            labels += pre if marker.supervise_open else [IGNORE] * len(pre)
            gen_point.append(len(ids))                      # after the marker; the slots follow
            gen_image.append(len(pixels))
        else:
            ids.append(vision_start_id)
            labels.append(IGNORE)

        image_at.append((len(ids), n_slots, len(pixels)))   # the slots, not the delimiters
        ids += [image_token_id] * n_slots
        labels += [IGNORE] * n_slots
        ids.append(vision_end_id)
        labels.append(IGNORE)
        pixels.append(seg.pixels)

    assert ids, "an empty sample"
    assert len(ids) == len(labels)

    useful = useful_len(labels, gen_point, text_loss)
    end = min(useful, max_len) if max_len else useful
    if end < len(ids):
        ids, labels = ids[:end], labels[:end]
        image_at = [(s, k, i) for s, k, i in image_at if s + k <= end]
        keep = [g for g, p in enumerate(gen_point) if p <= end]
        if len(keep) != len(gen_point):
            # `pixels` is renumbered; `image_at` and `gen_image` both index it
            live = sorted({gen_image[g] for g in keep} | {i for _, _, i in image_at})
            remap = {old: new for new, old in enumerate(live)}
            pixels = [pixels[i] for i in live]
            image_at = [(s, k, remap[i]) for s, k, i in image_at]
            gen_image = [remap[gen_image[g]] for g in keep]
            gen_point = [gen_point[g] for g in keep]
    return Rendered(ids=ids, labels=labels, image_at=image_at, pixels=pixels,
                    gen_point=gen_point, gen_image=gen_image, source=source)


def useful_len(labels: list, gen_point: list, text_loss: bool = True) -> int:
    """How many leading tokens of a rendered sample are read: up to the last supervised label (predicted from the
    position before it) and the last position of every target's condition (`gen_point - 1`).

    Without cross-entropy (`text_loss` False) only the conditions count, and a sample with no target is empty. A
    trailing target ends the sample with `gen_point == len(ids)`.
    """
    last_label = (max((i for i, x in enumerate(labels) if x != IGNORE), default=-1) if text_loss
                  else -1)
    last_cond = max((p - 1 for p in gen_point), default=-1)
    return max(last_label, last_cond) + 1


@dataclass
class PackedBatch:
    """One row: several samples laid end to end, `cu_seqlens` marking where each begins."""

    input_ids: torch.Tensor           # [1, L]
    labels: torch.Tensor              # [1, L]; IGNORE where not supervised
    image_slots: torch.Tensor         # [1, L] bool; True where a latent is scattered in
    attn: torch.Tensor                # [1, L] long; 0 on the padded tail of the row
    cu_seqlens: torch.Tensor          # [S+1] int32, sample boundaries in flat positions
    pixels: torch.Tensor              # [sum(3*h*w)] uint8, flat, sequence order -- see `images()`
    pixel_hw: torch.Tensor            # [N_img, 2] long, each image's (h, w) in pixels
    gen_point: torch.Tensor           # [N_tgt] flat index of each target's first slot
    gen_start: torch.Tensor           # [N_tgt] flat index where that target's sample begins
    gen_image: torch.Tensor           # [N_tgt] index into `pixels`
    slot_image: torch.Tensor          # [N_span] pixel index per image span, in span order
    slot_grid: torch.Tensor           # [N_span, 2] (gh, gw) per span
    sample_spans: list = field(default_factory=list)  # [S] image spans in each sample
    sources: list = field(default_factory=list)       # [S]
    sample_lens: list = field(default_factory=list)   # [S]

    @property
    def n_targets(self) -> int:
        return int(self.gen_point.numel())

    _LISTS = ("sample_spans", "sources", "sample_lens")

    def images(self) -> list:
        """-> `[3, h, w]` uint8 per image, in sequence order: views on the flat `pixels` buffer."""
        out, off = [], 0
        for h, w in self.pixel_hw.tolist():
            n = 3 * h * w
            out.append(self.pixels.narrow(0, off, n).view(3, h, w))
            off += n
        assert off == self.pixels.numel(), \
            f"pixel_hw covers {off} bytes of a {self.pixels.numel()}-byte buffer"
        return out

    def as_dict(self) -> dict:
        """The fields as a dict (what `Trainer._prepare_inputs` moves to the device)."""
        return {f: getattr(self, f) for f in
                ("input_ids", "labels", "image_slots", "attn", "cu_seqlens", "pixels", "pixel_hw",
                 "gen_point", "gen_start", "gen_image", "slot_image", "slot_grid") + self._LISTS}

    @classmethod
    def from_dict(cls, d: dict) -> "PackedBatch":
        return cls(**d)


def collate(rendered: list, pad_id: int, max_len: int,
            geom: Geometry = Geometry()) -> PackedBatch:
    """Rendered samples -> one row, as wide as they are (rounded up to `CHUNK` when that fits in `max_len`).

    Samples are not split across rows: the packer cuts a sample at the row boundary (`render`'s `max_len`).
    """
    assert rendered, "collate() got no samples"
    lens = [len(r) for r in rendered]
    assert min(lens) > 0, "collate() got a sample that rendered to zero tokens"
    assert max(lens) <= max_len, f"a sample of {max(lens)} tokens in a row of {max_len}"
    assert sum(lens) <= max_len, \
        f"{sum(lens)} tokens do not fit one row of {max_len}; the packer must close rows"
    # The width is rounded up to a multiple of the linear-attention chunk, so the kernel has no partial trailing
    # block; the padding is closed as its own segment by `packing_kwargs` and is unlabelled and masked.
    B = 1
    total = sum(lens)
    up = (total + CHUNK - 1) // CHUNK * CHUNK
    L = up if up <= max_len else total

    input_ids = torch.full((B, L), pad_id, dtype=torch.long)
    labels = torch.full((B, L), IGNORE, dtype=torch.long)
    image_slots = torch.zeros(B, L, dtype=torch.bool)
    attn = torch.zeros(B, L, dtype=torch.long)

    cu, pixels, gen_point, gen_start, gen_image = [0], [], [], [], []
    slot_image, slot_grid, sample_spans = [], [], []
    for s, r in enumerate(rendered):
        row, col = 0, cu[-1]
        n = len(r)
        input_ids[row, col:col + n] = torch.tensor(r.ids, dtype=torch.long)
        lab = torch.tensor(r.labels, dtype=torch.long)
        # a sample's first token is never predicted (it would be predicted from the previous sample)
        lab[0] = IGNORE
        labels[row, col:col + n] = lab
        attn[row, col:col + n] = 1
        sample_spans.append(len(r.image_at))
        # spans in position order: `slot_image` follows the row-major order the slots are filled in
        for start, k, img in sorted(r.image_at):
            image_slots[row, col + start: col + start + k] = True
            slot_image.append(len(pixels) + img)
            h, w = int(r.pixels[img].shape[1]), int(r.pixels[img].shape[2])
            slot_grid.append(geom.grid(h, w))
        # flat indices into the row, like `cu_seqlens`
        base = col
        for p, gi in zip(r.gen_point, r.gen_image):
            gen_point.append(base + p)
            gen_start.append(base)
            gen_image.append(len(pixels) + gi)
        pixels += r.pixels
        cu.append(cu[-1] + n)

    assert int(image_slots.sum()) == sum(k for r in rendered for _, k, _ in r.image_at), \
        "the image slot mask does not cover exactly the rendered spans"

    return PackedBatch(
        input_ids=input_ids, labels=labels, image_slots=image_slots, attn=attn,
        cu_seqlens=torch.tensor(cu, dtype=torch.int32),
        pixels=(torch.cat([p.reshape(-1) for p in pixels]) if pixels
                else torch.zeros(0, dtype=torch.uint8)),
        pixel_hw=torch.tensor([[int(p.shape[1]), int(p.shape[2])] for p in pixels],
                              dtype=torch.long).reshape(-1, 2),
        gen_point=torch.tensor(gen_point, dtype=torch.long),
        gen_start=torch.tensor(gen_start, dtype=torch.long),
        gen_image=torch.tensor(gen_image, dtype=torch.long),
        slot_image=torch.tensor(slot_image, dtype=torch.long),
        slot_grid=torch.tensor(slot_grid, dtype=torch.long).reshape(-1, 2),
        sample_spans=sample_spans,
        sources=[r.source for r in rendered], sample_lens=lens)
