"""The visual generation training stream: sources -> mixer -> segments -> packed rows.

Everything here runs in CPU dataloader workers. Images arrive as encoded bytes and leave as uint8 pixels; the vision
tokenizer encodes them later, on the GPU.

The mixer draws a source per sample and never waits for one: a source whose unit is exhausted ends that unit and the
next draw picks again. Samples are rejected (unreadable image, empty text, nothing to generate) above the mixer.

One sample is one sequence: a multi-turn chain arrives from `sources.py` as one ordered item list and stays one
sample in the packed row, with every turn's generated image a flow target.
"""

import io
import json
import math
import random

import numpy as np
import torch
from PIL import Image as PILImage
from torch.utils.data import IterableDataset
from webdataset.utils import pytorch_worker_info

from .packer import RowPacker
from .segments import Geometry, Image, Marker, Text
from .sources import ITEMS_KEY, IMG_PREFIX, build_edit_units
from .t2i_dataset import drop


class EditSource:
    """One dataset of `sources.EDIT_SOURCES`. `open_unit` reads one unit (a row group, an index range or a tar shard),
    shuffled."""

    def __init__(self, name: str, instruction: str = None):
        self.name = name
        self.units = build_edit_units(name, instruction=instruction)

    def describe(self) -> str:
        return self.units.describe()

    def open_unit(self, rng):
        unit = rng.choice(self.units.units)
        try:
            samples = self.units.read(unit, rng)
        except Exception as exc:                                  # noqa: BLE001
            drop(f"{self.name}/unit", f"{unit}: {type(exc).__name__}: {exc}")
            samples = []
        return iter(samples)


class SourceMix(IterableDataset):
    """Infinite raw-sample stream: draw a source per sample by its share, read from its open unit.

    Seeded per (rank, dataloader worker). A draw opens at most `_MAX_TRIES` units; `_MAX_DRY` consecutive draws that
    produce nothing raise.
    """

    _MAX_TRIES = 8

    def __init__(self, sources, weights, seed: int):
        super().__init__()
        self.sources = sources
        self.weights = weights
        self.seed = seed

    _MAX_DRY = 256

    def __iter__(self):
        rank, _, worker, _ = pytorch_worker_info()
        rng = random.Random(self.seed * 1000003 + rank * 1009 + worker)
        idx = list(range(len(self.sources)))
        its = [None] * len(self.sources)
        dry = 0
        while dry < self._MAX_DRY:
            i = rng.choices(idx, weights=self.weights)[0]
            src = self.sources[i]
            sample = None
            for _ in range(self._MAX_TRIES):
                if its[i] is None:
                    try:
                        its[i] = src.open_unit(rng)
                    except Exception as exc:                      # noqa: BLE001
                        drop(f"{src.name}/unit", f"open failed: {type(exc).__name__}: {exc}")
                        continue
                try:
                    sample = next(its[i])
                except StopIteration:
                    its[i] = None
                    continue
                except Exception as exc:                          # noqa: BLE001
                    drop(f"{src.name}/unit", f"read failed: {type(exc).__name__}: {exc}")
                    its[i] = None
                    continue
                break
            if sample is None:
                dry += 1
                continue
            dry = 0
            yield sample, src.name
        raise RuntimeError(
            f"{self._MAX_DRY} consecutive draws produced no sample. Every source in the mix "
            f"failed to open or read a unit; failing here rather than spinning silently.")


def to_pixels(blob: bytes, resolution) -> torch.Tensor:
    """Encoded bytes -> uint8 `[3, H, W]`: resized to cover `resolution` (a side or `(H, W)`), then centre-cropped.

    No random crop and no flip: the text refers to positions in the image.
    """
    th, tw = (resolution, resolution) if isinstance(resolution, int) else resolution
    with PILImage.open(io.BytesIO(blob)) as im:
        im = im.convert("RGB")
        w, h = im.size
        s = max(th / h, tw / w)
        nw, nh = max(tw, round(w * s)), max(th, round(h * s))
        im = im.resize((nw, nh), PILImage.BICUBIC)
        left, top = (nw - tw) // 2, (nh - th) // 2
        im = im.crop((left, top, left + tw, top + th))
        arr = np.asarray(im, dtype=np.uint8)
    return torch.from_numpy(arr.copy()).permute(2, 0, 1)


def fit_size(w: int, h: int, budget: int, stride: int = 32, clamp: float = 2.0) -> tuple:
    """A source image's `(w, h)` -> the `(H, W)` it is trained at.

    1. The aspect ratio is clamped to `1:clamp .. clamp:1` (the centre crop in `to_pixels` realizes it).
    2. The area is capped at `budget` pixels (latents are `H * W / patch**2`); a smaller image is not upscaled.
    3. Each side is floored to a multiple of `stride` (patch * merge), at least `stride`.
    4. The clamp is applied again, since step 3 can push the ratio past it.
    """
    ar = w / h
    if ar > clamp:
        w = h * clamp
    elif ar < 1 / clamp:
        h = w * clamp
    if w * h > budget:
        s = math.sqrt(budget / (w * h))
        w, h = w * s, h * s
    W = max(stride, int(w) // stride * stride)
    H = max(stride, int(h) // stride * stride)
    if W > clamp * H:
        W = max(stride, int(clamp * H) // stride * stride)
    elif H > clamp * W:
        H = max(stride, int(clamp * W) // stride * stride)
    return H, W


# Qwen3.5's chat markup, tokenized once per adapter. The assistant turn after the last user turn opens with the
# empty think block, as `apply_chat_template(..., add_generation_prompt=True, enable_thinking=False)` ends; earlier
# assistant turns open bare.
SYSTEM_TURN = "<|im_start|>system\n{system}<|im_end|>\n"
USER_OPEN = "<|im_start|>user\n"
ASSISTANT_OPEN = "<|im_start|>assistant\n"
ASSISTANT_OPEN_LAST = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
# The turn end: `<|im_end|>` (the eos the model emits) and the template's separating newline, separate segments
# because only the first can be supervised.
TURN_EOS = "<|im_end|>"
TURN_NL = "\n"


class SegmentAdapter:
    """A raw sample -> a list of `Text` / `Image` segments in Qwen3.5's chat format.

    For a chain of k turns:

        <|im_start|>system\\n ... <|im_end|>\\n                  unsupervised
        <|im_start|>user\\n  Image(img0)  instruction_1  <|im_end|>\\n   unsupervised
        <|im_start|>assistant\\n                               unsupervised (not the last assistant turn)
          Image(img1, generate=True)                           `render` writes the generation marker in front
        <|im_end|>                                             supervised with `supervise_open`
        \\n                                                     unsupervised
        <|im_start|>user\\n instruction_2 <|im_end|>\\n
        <|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n           the last assistant turn
          Image(img2, generate=True)  <|im_end|>  \\n

    A segment the model produces (a generated image, supervised text) is the assistant's; everything else is the
    user's. Instructions are not supervised. A segment that cannot be built (an undecodable image, a text that
    tokenizes to nothing) truncates the chain there unless the sample's `__trim__` is false, in which case the sample
    is dropped.

    Args:
        tokenizer: the model's tokenizer.
        resolution: the image side when `max_area` is 0.
        system: the system prompt ("" = no system turn).
        supervise_open: label the `<|im_end|>` of an assistant turn that only generates images.
        user_template: wraps every request for an image, with `{h}`, `{w}` (that image's size) and `{caption}`.
        caption_dropout: the probability of emptying the last request's caption.
        max_area: 0 = every image at `resolution`; otherwise each image is sized by `fit_size` within this many pixels.
        stride: patch * merge.
        ar_clamp: the aspect-ratio limit of `fit_size`.
    """

    def __init__(self, tokenizer, resolution: int, system: str = "",
                 supervise_open: bool = True, user_template: str | None = None,
                 caption_dropout: float = 0.0, max_area: int = 0, stride: int = 32,
                 ar_clamp: float = 2.0):
        self.tok = tokenizer
        self.resolution = resolution
        self.max_area = int(max_area)
        self.stride = int(stride)
        self.ar_clamp = float(ar_clamp)
        self.user_template = user_template
        self.caption_dropout = float(caption_dropout)
        self._supervise_open = bool(supervise_open)
        self._sys = self._ids(SYSTEM_TURN.format(system=system)) if system else []
        self._user_open = self._ids(USER_OPEN)
        self._asst_open = self._ids(ASSISTANT_OPEN)
        self._asst_open_last = self._ids(ASSISTANT_OPEN_LAST)
        self._eos = self._ids(TURN_EOS)
        self._nl = self._ids(TURN_NL)

    def _ids(self, text: str) -> list:
        return self.tok(text, add_special_tokens=False)["input_ids"]

    @staticmethod
    def _role(seg) -> str:
        produced = seg.generate if isinstance(seg, Image) else seg.supervise
        return "assistant" if produced else "user"

    def _wrap(self, body: list) -> list:
        """Segments in sample order -> the same segments inside `<|im_start|>role ... <|im_end|>\\n` turns.

        Consecutive segments of one role are one turn.
        """
        runs, i = [], 0
        while i < len(body):
            role = self._role(body[i])
            j = i
            while j < len(body) and self._role(body[j]) == role:
                j += 1
            runs.append((role, body[i:j]))
            i = j
        last_user = max((n for n, (r, _) in enumerate(runs) if r == "user"), default=-1)

        out = []
        for n, (role, segs) in enumerate(runs):
            if role == "user":
                out.append(Text(self._user_open, supervise=False))
                out += segs
                out.append(Text(self._eos, supervise=False))
                out.append(Text(self._nl, supervise=False))
                continue
            out.append(Text(self._asst_open_last if n > last_user else self._asst_open, supervise=False))
            out += segs
            # an assistant turn with supervised text always ends on a supervised eos; one that only generates
            # images does with `supervise_open`
            says_words = any(isinstance(s, Text) and s.supervise for s in segs)
            out.append(Text(self._eos, supervise=(self._supervise_open or says_words)))
            out.append(Text(self._nl, supervise=False))
        return out

    def _request_items(self, items: list) -> set:
        """Indexes of the text items that request an image: the last item of a user run, a text, followed by a
        generated image."""
        def user(it):
            return not (bool(it[2]) if len(it) > 2 else False)

        out, i = set(), 0
        while i < len(items):
            j = i
            while j < len(items) and user(items[j]) == user(items[i]):
                j += 1
            if user(items[i]) and j < len(items) and items[j][0] == "image" \
                    and items[j - 1][0] == "text":
                out.add(j - 1)
            i = j
        return out

    def body_of(self, raw: dict, name: str, rng=None):
        """The sample's segments before the chat turns are wrapped around them, or None if the sample is dropped."""
        items = json.loads(raw[ITEMS_KEY])
        wrap_at = self._request_items(items) if self.user_template else set()
        # the last request is the one the caption dropout empties
        drop_at = (max(wrap_at) if wrap_at and rng is not None
                   and rng.random() < self.caption_dropout else None)
        blobs = [raw[k] for k in sorted(
            (k for k in raw if k.startswith(IMG_PREFIX) and "." in k),
            key=lambda k: int(k[len(IMG_PREFIX):k.index(".")]))]
        trim = bool(raw.get("__trim__", True))

        # Each image's size from its own header (PIL reads the header only).
        sizes = [self.resolution] * len(blobs)
        if self.max_area and blobs:
            try:
                for i, b in enumerate(blobs):
                    with PILImage.open(io.BytesIO(b)) as probe:
                        sizes[i] = fit_size(*probe.size, budget=self.max_area,
                                            stride=self.stride, clamp=self.ar_clamp)
            except Exception as exc:                                   # noqa: BLE001
                drop("adapt/size", f"could not read an image's size: {type(exc).__name__}")
                return None
        # a request names the size of the image that answers it
        req_size = {n: sizes[items[n + 1][1]] for n in wrap_at
                    if n + 1 < len(items) and items[n + 1][0] == "image"}

        body, bad = [], None
        for n, it in enumerate(items):
            if it[0] == "text":
                if n in wrap_at:
                    # formatted, then tokenized in one call; `.strip()` is the template's `|trim`
                    rs = req_size.get(n, self.resolution)
                    req_h, req_w = (rs, rs) if isinstance(rs, int) else rs
                    ids = self._ids(self.user_template.format(
                        h=req_h, w=req_w,
                        caption="" if n == drop_at else it[1]).strip())
                else:
                    ids = self._ids(it[1])
                if not ids:
                    bad = f"a {it[0]} item tokenized to nothing"
                    break
                body.append(Text(ids, supervise=bool(it[2]) if len(it) > 2 else False))
            else:
                _, idx, generate = it
                try:
                    px = to_pixels(blobs[idx], sizes[idx])
                except Exception as exc:                          # noqa: BLE001
                    bad = f"undecodable image: {type(exc).__name__}"
                    break
                body.append(Image(px, generate=bool(generate)))

        if bad is not None:
            if not trim:
                drop(f"{name}/adapt", f"{bad}; this source may not be truncated")
                return None
            while body and isinstance(body[-1], Text):
                body.pop()                    # an instruction whose image never arrived
            if not any(isinstance(s, Image) and s.generate for s in body):
                drop(f"{name}/adapt", f"{bad}; nothing with a target left after truncation")
                return None
            drop(f"{name}/adapt", f"{bad}; chain truncated to {len(body)} segments")

        return body

    def __call__(self, raw: dict, name: str, rng=None):
        body = self.body_of(raw, name, rng)
        if body is None:
            return None
        segs = [Text(self._sys)] if self._sys else []
        return segs + self._wrap(body)


class SegmentStream:
    """The sample factory `RowPacker` calls: a fresh mixer and adapter output, built in the consuming process (so the
    mixer reads its worker identity there)."""

    def __init__(self, sources, weights, seed: int, adapt: "SegmentAdapter"):
        self.sources, self.weights, self.seed, self.adapt = sources, weights, seed, adapt

    def __call__(self):
        # the caption dropout's RNG, separate from the mixer's
        rank, _, worker, _ = pytorch_worker_info()
        rng = random.Random(self.seed * 7_368_787 + rank * 1009 + worker)
        for raw, name in SourceMix(self.sources, self.weights, self.seed):
            segs = self.adapt(raw, name, rng)
            if segs is not None:
                yield segs, name


def build_interleaved_stream(cfg: dict, tokenizer, seed: int, *, geom: Geometry = Geometry()):
    """`cfg` -> an iterable of packed rows (`PackedBatch`).

    cfg keys:
      resolution      the image side when `max_area` is 0, in pixels
      row_len         the maximum width of a packed row
      sources         [{name, share}, ...] of `sources.EDIT_SOURCES`
      system          the system prompt
      gen_marker      the text before a generated image; `{h}` / `{w}` are filled with its size
      user_template   the request for an image, with `{h}`, `{w}`, `{caption}`
      caption_dropout the probability of emptying the last request's caption
      instruction     the reconstruction sources' request
      max_area        0, or the pixel budget of `fit_size`
      ar_clamp        the aspect-ratio limit of `fit_size`
      supervise_open  whether the generation marker (and the eos of an image-only turn) is a cross-entropy target
      max_targets, max_target_latents
                      caps on a row's targets and their latents; 0 = off. See `RowPacker`.
      text_loss       whether the run scores cross-entropy; False cuts each sample after its last target's condition
                      (`segments.useful_len`)
      report_every    rows between packing statistics (0 = never)
    """
    resolution = int(cfg["resolution"])
    assert resolution % (geom.patch * geom.pool) == 0, \
        f"resolution {resolution} must be a multiple of patch {geom.patch} x pool {geom.pool}"

    sources, weights = [], []
    for spec in cfg["sources"]:
        s = EditSource(spec["name"], instruction=cfg.get("instruction"))
        sources.append(s)
        weights.append(float(spec["share"]))
        print(f"[interleaved] {spec['name']}: {s.describe()}, share={weights[-1]:.4g}")

    total = sum(weights)
    for s, w in zip(sources, weights):
        print(f"[interleaved] {s.name}: prob={w / total:.4f}")

    supervise_open = bool(cfg.get("supervise_open", True))
    max_area = int(cfg.get("max_area", 0))
    adapt = SegmentAdapter(tokenizer, resolution, system=cfg.get("system", ""),
                           supervise_open=supervise_open,
                           user_template=cfg.get("user_template"),
                           caption_dropout=float(cfg.get("caption_dropout", 0.0)),
                           max_area=max_area, stride=geom.patch * geom.pool,
                           ar_clamp=float(cfg.get("ar_clamp", 2.0)))
    if max_area:
        print(f"[interleaved] DYNAMIC RESOLUTION: up to {max_area:,} px per image "
              f"({max_area // (geom.patch ** 2)} latents), sides a multiple of "
              f"{geom.patch * geom.pool}, aspect clamped to "
              f"1:{cfg.get('ar_clamp', 2.0)}..{cfg.get('ar_clamp', 2.0)}:1")
    marker_text = cfg["gen_marker"]
    cache: dict = {}

    def prefix_ids(h, w):
        if (h, w) not in cache:
            cache[(h, w)] = tokenizer(marker_text.format(h=h, w=w),
                                      add_special_tokens=False)["input_ids"]
        return cache[(h, w)]

    marker = Marker(prefix_ids=prefix_ids, supervise_open=supervise_open)

    packer = RowPacker(
        SegmentStream(sources, weights, seed, adapt),
        max_len=int(cfg["row_len"]),
        pad_id=tokenizer.pad_token_id,
        marker=marker,
        image_token_id=tokenizer.convert_tokens_to_ids("<|image_pad|>"),
        vision_start_id=tokenizer.convert_tokens_to_ids("<|vision_start|>"),
        vision_end_id=tokenizer.convert_tokens_to_ids("<|vision_end|>"),
        geom=geom,
        report_every=int(cfg.get("report_every", 0)),
        max_targets=int(cfg.get("max_targets", 0)),
        max_target_latents=int(cfg.get("max_target_latents", 0)),
        text_loss=bool(cfg.get("text_loss", True)),
    )
    return packer
