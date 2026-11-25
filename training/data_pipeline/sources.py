"""The visual generation training datasets, one sequence per sample.

Whatever a dataset calls a record, a sample is an ordered list of items:

    ("text",  str, supervise: bool)   text; `supervise` marks text the model produces
    ("image", ref, generate: bool)    an image; `generate` marks images the model produces (flow targets)

    single-turn edit   [img(given), text(instruction), img(made)]
    k-turn chain       [img0(given), t1, img1(made), t2, img2(made), ..., tk, imgk(made)]
    reasoning trace    [text(question), img(given), text(thought, T), img(made), ..., text(answer, T)]

What the model produces is the assistant's turn and the rest the user's (`stream.SegmentAdapter`). Every made image
of a chain is a target, all in one sequence: the language model runs once over the chain and each target's condition
ends at its own generation marker.

Each sample carries `__trim__`: whether a sample that loses a segment (an unreadable image) may be truncated there
(an edit chain) or must be dropped (multi-reference and reasoning samples, whose text refers to every image).

Images are encoded bytes here; they are decoded in `stream.py`.
"""

import glob
import json
import os
import re
import tarfile
from dataclasses import dataclass

import numpy as np
import pyarrow.parquet as pq
from webdataset.utils import pytorch_worker_info

from .data_paths import dataset_path
from .t2i_dataset import drop

ITEMS_KEY = "items.json"        # the sequence: order, roles, and which images are made
IMG_PREFIX = "img"              # img0.<ext>, img1.<ext>, ... in sequence order


def _ext_of(data: bytes, path: str = "") -> str:
    """The image's extension: from the magic bytes, the file name as fallback."""
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if data[:2] == b"\xff\xd8":
        return "jpg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    ext = os.path.splitext(path)[1].lstrip(".").lower()
    return {"": "jpg", "jpeg": "jpg"}.get(ext, ext)


def _raw(key: str, items: list, blobs: list, paths=None, trim: bool = True) -> dict:
    """One sample -> the raw dict `stream.SegmentAdapter` reads.

    `items` is the sequence with each image given as an index into `blobs`; the images are stored as
    `img<i>.<ext>`. `trim`: see the module docstring.
    """
    paths = paths or [""] * len(blobs)
    out = {"__key__": key, "__trim__": bool(trim),
           ITEMS_KEY: json.dumps(items).encode("utf-8")}
    for i, (data, p) in enumerate(zip(blobs, paths)):
        out[f"{IMG_PREFIX}{i}.{_ext_of(data, p)}"] = data
    return out


def n_generated(items: list) -> int:
    """The number of images the sample generates."""
    return sum(1 for it in items if it[0] == "image" and it[2])


def _walk(seq: list, blobs: list):
    """(item, its blob or None) in sequence order, so the two lists stay aligned."""
    k = 0
    for it in seq:
        if it[0] == "image":
            yield it, blobs[k]
            k += 1
        else:
            yield it, None


def one_turn(src, instruction: str, tgt) -> list:
    """given image -> instruction (unsupervised) -> made image."""
    return [("image", src, False), ("text", instruction, False), ("image", tgt, True)]


@dataclass(frozen=True)
class EditIndexSpec:
    """One annotation file (or directory of jsonl files) and image root.

    `sequence_of(record)` returns the record's item list with image refs as paths (relative to `root` or absolute),
    or [] to skip the record. `trim`: see the module docstring.
    """

    name: str
    annotation: str
    root: str
    sequence_of: "Callable[[dict], list]"  # noqa: F821
    trim: bool = True


# GPT-Image-Edit-1.5M: {"image": [input, output], "conversations": [{"from": "human", "value": "<image>\n..."}, ...]}
_GIE = dataset_path("gpt-image-edit")
_GIE_JSON = os.path.join(_GIE, "training_json")


def _gie(rec: dict) -> list:
    imgs = rec.get("image") or []
    if len(imgs) != 2:
        return []
    conv = rec.get("conversations") or []
    text = next((c.get("value", "") for c in conv if c.get("from") == "human"), "")
    text = text.replace("<image>", "").strip()      # the marker is positional, not part of it
    return one_turn(imgs[0], text, imgs[1]) if text else []


GPT_IMAGE_EDIT_PARTS = (
    EditIndexSpec("gpt-edit-hqedit-edit", os.path.join(_GIE_JSON, "hqedit_gpt_edit.json"),
                  os.path.join(_GIE, "hqedit/edit"), _gie),
    EditIndexSpec("gpt-edit-hqedit-generate", os.path.join(_GIE_JSON, "hqedit_gpt_generate.json"),
                  os.path.join(_GIE, "hqedit/generate"), _gie),
    EditIndexSpec("gpt-edit-omniedit", os.path.join(_GIE_JSON, "omniedit_gpt.json"),
                  os.path.join(_GIE, "omniedit"), _gie),
    EditIndexSpec("gpt-edit-omniedit-rewrite", os.path.join(_GIE_JSON, "omniedit_gpt_rewrite.json"),
                  os.path.join(_GIE, "omniedit"), _gie),
    EditIndexSpec("gpt-edit-complex", os.path.join(_GIE_JSON, "complexedit_gpt.json"),
                  os.path.join(_GIE, "omniedit/complex-edit"), _gie),
    EditIndexSpec("gpt-edit-ultraedit", os.path.join(_GIE_JSON, "ultraedit_gpt.json"),
                  os.path.join(_GIE, "ultraedit"), _gie),
)


# Echo-4o-Image: instruction-following and surreal-fantasy text-to-image sets, and multi-reference generation.
_ECHO = dataset_path("echo-4o-image")


def _echo_path(p: str) -> str:
    """An annotation path (`/Echo-4o-image/...`, whose first component's case differs from the directory's) -> the
    file under the dataset directory."""
    parts = [x for x in (p or "").split("/") if x]
    if parts and parts[0].lower() == "echo-4o-image":
        parts = parts[1:]
    return os.path.join(_ECHO, *parts)


def _echo_multiref(rec: dict) -> list:
    """Multi-reference: the given images in their order (the instruction refers to them by position), the
    instruction, the made image."""
    ins = (rec.get("instruction") or "").strip()
    refs = rec.get("input_images") or []
    out = rec.get("output_image")
    if not ins or not out or not refs:
        return []
    items = [("image", _echo_path(p), False) for p in refs]
    items.append(("text", ins, False))
    items.append(("image", _echo_path(out), True))
    return items


def _echo_t2i(rec: dict) -> list:
    """Instruction-following and surreal-fantasy: a caption, and the image it asks for."""
    ins = (rec.get("instruction") or "").strip()
    out = rec.get("output_image")
    return [("text", ins, False), ("image", _echo_path(out), True)] if ins and out else []


ECHO_MULTIREF_GROUPS = ("human", "object", "hum_obj", "hum_sce", "obj_sce", "hum_obj_sce", "other")

ECHO_PARTS = (
    EditIndexSpec("echo4o-instruction-following",
                  os.path.join(_ECHO, "Instruction-Following-Image",
                               "Instruction-Following-Image.jsonl"),
                  _ECHO, _echo_t2i),
    EditIndexSpec("echo4o-surreal-fantasy",
                  os.path.join(_ECHO, "Surrel-Fantasy-Image", "conflict.jsonl"),
                  _ECHO, _echo_t2i),
) + tuple(
    # one source per reference kind (a person, an object, a scene, and their combinations)
    EditIndexSpec(f"echo4o-multiref-{g.replace('_', '-')}",
                  os.path.join(_ECHO, "Multi-Reference-Image", "multi-ic", f"{g}.jsonl"),
                  _ECHO, _echo_multiref, trim=False)
    for g in ECHO_MULTIREF_GROUPS
)


class JsonlEditUnits:
    """The records of an `EditIndexSpec`, in units of `SAMPLES_PER_UNIT` consecutive samples.

    The index (item lists, no pixels) is built once; a unit's images are read from disk when the unit is read.
    """

    SAMPLES_PER_UNIT = 100

    def __init__(self, spec: EditIndexSpec):
        self.spec = spec
        files = ([spec.annotation] if os.path.isfile(spec.annotation)
                 else sorted(glob.glob(os.path.join(spec.annotation, "*.jsonl"))))
        assert files, f"{spec.name}: no annotation file at {spec.annotation}."

        # The index is one byte blob of json-encoded item lists and an offset array (two objects, so the pages
        # dataloader workers share through the fork are not copied by refcount writes). Records `sequence_of`
        # rejects are counted. The parse is cached next to the annotation as `.<name>.editindex.npz`, keyed on
        # the annotation files' names, sizes and mtimes; a cache that cannot be written is skipped.
        if not self._load_index(files):
            self.n_records = self.n_samples = self.n_targets = self.max_targets = 0
            parts, off, total = [], [0], 0

            def take(rec):
                nonlocal total
                self.n_records += 1
                seq = spec.sequence_of(rec)
                if not seq:
                    return
                enc = json.dumps(seq).encode("utf-8")
                parts.append(enc)
                total += len(enc)
                off.append(total)
                self.n_samples += 1
                t = n_generated(seq)
                self.n_targets += t
                self.max_targets = max(self.max_targets, t)

            for f in files:
                if f.endswith(".jsonl"):
                    with open(f) as fh:
                        for line in fh:
                            if line.strip():
                                take(json.loads(line))
                else:                                        # GPT-Image-Edit ships one json list
                    with open(f) as fh:
                        for rec in json.load(fh):
                            take(rec)
            assert self.n_samples, f"{spec.name}: annotation parsed to zero samples"
            self._blob = b"".join(parts)
            parts.clear()                                    # the join doubled it for a moment
            self._off = np.asarray(off, dtype=np.int64)
            self._save_index(files)
        assert self.n_samples, f"{spec.name}: annotation parsed to zero samples"

        n = self.SAMPLES_PER_UNIT
        self.units = [(i, min(i + n, self.n_samples))
                      for i in range(0, self.n_samples, n)]

    # The parse cache: `_blob`, `_off` and four counters.

    def _index_path(self) -> str:
        base = self.spec.annotation
        d = base if os.path.isdir(base) else os.path.dirname(base)
        return os.path.join(d, f".{self.spec.name}.editindex.npz")

    @staticmethod
    def _stamp(files):
        return np.asarray([[os.path.getsize(f), int(os.path.getmtime(f))] for f in files],
                          dtype=np.int64)

    def _load_index(self, files) -> bool:
        p = self._index_path()
        if not os.path.isfile(p):
            return False
        try:
            z = np.load(p, allow_pickle=False)
            if list(z["files"]) != [os.path.basename(f) for f in files]:
                return False
            if not np.array_equal(z["stamp"], self._stamp(files)):
                return False
            self._blob = z["blob"].tobytes()
            self._off = z["off"]
            (self.n_records, self.n_samples, self.n_targets,
             self.max_targets) = (int(x) for x in z["counts"])
        except Exception:                                     # noqa: BLE001
            return False
        return True

    def _save_index(self, files) -> None:
        p = self._index_path()
        try:
            tmp = f"{p}.{os.getpid()}.npz"
            np.savez(tmp,
                     files=np.asarray([os.path.basename(f) for f in files]),
                     stamp=self._stamp(files),
                     blob=np.frombuffer(self._blob, dtype=np.uint8),
                     off=self._off,
                     counts=np.asarray([self.n_records, self.n_samples,
                                        self.n_targets, self.max_targets], dtype=np.int64))
            os.replace(tmp, p)                                # every rank reaches this together
        except OSError:
            pass          # a read-only mount is not a reason to fail; it only costs the parse

    def __len__(self):
        return len(self.units)

    def _sequence(self, i: int) -> list:
        """Sample `i`'s item list (items as lists after the json round trip)."""
        return json.loads(self._blob[self._off[i]:self._off[i + 1]])

    def describe(self) -> str:
        lost = self.n_records - self.n_samples
        return (f"{self.n_samples:,} samples / {len(self.units):,} units, "
                f"{self.n_targets:,} targets, up to {self.max_targets} per sample"
                + (f"  [{lost:,} of {self.n_records:,} records rejected: "
                   f"{100 * lost / self.n_records:.1f}%]" if lost else ""))

    def _abs(self, p: str) -> str:
        return p if os.path.isabs(p) else os.path.join(self.spec.root, p)

    def read(self, unit, rng=None):
        lo, hi = unit
        order = list(range(lo, hi))
        if rng is not None:
            rng.shuffle(order)

        name = self.spec.name
        out = []
        for i in order:
            seq = self._sequence(i)
            paths = [self._abs(it[1]) for it in seq if it[0] == "image"]
            try:
                blobs = []
                for p in paths:
                    with open(p, "rb") as fh:
                        blobs.append(fh.read())
            except OSError as exc:
                drop(f"{name}/file", f"{type(exc).__name__}: {exc.filename}")
                continue
            if any(not b for b in blobs):
                drop(f"{name}/file", "zero-byte image")
                continue
            # A turn whose image is byte-identical to the previous one ends the chain there.
            keep, prev = [], None
            for it, blob in _walk(seq, blobs):
                if it[0] == "image" and blob is not None and blob == prev:
                    drop(f"{name}/dup", "a turn produced a byte-identical image; chain truncated")
                    break
                keep.append(it)
                if it[0] == "image":
                    prev = blob
            while keep and keep[-1][0] == "text":
                keep.pop()                       # an instruction with nothing to answer it
            if not any(it[0] == "image" and it[2] for it in keep):
                drop(f"{name}/dup", "nothing left with a target after truncation")
                continue

            items, kept_blobs, kept_paths, k = [], [], [], 0
            for it in keep:
                if it[0] == "image":
                    items.append(("image", k, it[2]))
                    kept_blobs.append(blobs[k])
                    kept_paths.append(paths[k])
                    k += 1
                else:
                    items.append(it)
            out.append(_raw(f"{name}{i:09d}", items, kept_blobs, kept_paths,
                            trim=self.spec.trim))
        return out


# Zebra-CoT: interleaved reasoning; parquet with the images inline.

_ZEBRA = dataset_path("zebra-cot")

# `<image_start>[reasoning_image_2]<image_end>` in the text: the name in the brackets is the row's image column.
# A marker that names no image drops the row.
ZEBRA_MARK = re.compile(r"<image_start>\[([^\]]+)\]<image_end>")

ZEBRA_SUBSETS = (
    "2D Visual Reasoning - Visual Jigsaw",
    "2D Visual Reasoning - Visual Search",
    "3D Visual Reasoning - Embodied CoT",
    "3D Visual Reasoning - Multi-Hop Objects Counting",
    "3D Visual Reasoning - Robot Planning",
    "Scientific Reasoning - Chemistry",
    "Scientific Reasoning - Competitive Programming",
    "Scientific Reasoning - Geometry",
    "Scientific Reasoning - Graph Algorithms",
    "Scientific Reasoning - Physics",
    "Visual Logic & Strategic Games - ARC-AGI",
    "Visual Logic & Strategic Games - Checkers",
    "Visual Logic & Strategic Games - Chess",
    "Visual Logic & Strategic Games - Ciphers",
    "Visual Logic & Strategic Games - Connect Four",
    "Visual Logic & Strategic Games - Maze",
    "Visual Logic & Strategic Games - RPM",
    "Visual Logic & Strategic Games - Tetris",
)


def _zebra_name(sub: str) -> str:
    """The subset directory -> the name a config writes. Lowercase, no spaces or punctuation."""
    return "zebra-" + re.sub(r"[^a-z0-9]+", "-", sub.lower()).strip("-")


class ZebraUnits:
    """(file, row group) units for one Zebra-CoT subset. One row becomes:

        user       the question, with its `problem_image_*` where the text names them
        assistant  the reasoning trace (supervised), with its `reasoning_image_*` generated where the text names
                   them, then the final answer (supervised)

    Not trimmed (`trim=False`).
    """

    def __init__(self, subset: str, path: str = _ZEBRA):
        self.subset = subset
        self.dir = os.path.join(path, subset)
        self.files = sorted(glob.glob(os.path.join(self.dir, "*.parquet")))
        assert self.files, f"no parquet under {self.dir}"
        self.units, self._handles = [], {}
        for i, f in enumerate(self.files):
            self.units += [(i, g) for g in range(pq.ParquetFile(f).metadata.num_row_groups)]
        cols = list(pq.ParquetFile(self.files[0]).schema_arrow.names)
        self.img_cols = [c for c in cols if c not in ("Question", "Text Reasoning Trace",
                                                      "Final Answer")]
        self.n_rows = sum(pq.ParquetFile(f).metadata.num_rows for f in self.files)

    def __len__(self):
        return len(self.units)

    def describe(self) -> str:
        return (f"{len(self.files)} parquet files / {len(self.units):,} row groups, "
                f"{self.n_rows:,} rows, {len(self.img_cols)} image columns")

    def _sequence(self, row: dict):
        """One row -> (items, blobs) or (None, the reason it is dropped). Images are indexes into `blobs`, in the
        order the markers appear."""
        blobs, items = [], []

        def add_text(s: str, supervise: bool):
            s = s.strip()
            if s:
                items.append(("text", s, supervise))

        def walk(text: str, supervise: bool, generate: bool):
            pos = 0
            for m in ZEBRA_MARK.finditer(text):
                add_text(text[pos:m.start()], supervise)
                name = m.group(1)
                cell = row.get(name)
                data = cell.get("bytes") if isinstance(cell, dict) else None
                if not data:
                    return f"marker [{name}] resolves to no image"
                items.append(("image", len(blobs), generate))
                blobs.append(data)
                pos = m.end()
            add_text(text[pos:], supervise)
            return None

        # the question and its figures are given; the trace and its figures are produced
        bad = walk(row.get("Question") or "", supervise=False, generate=False)
        if bad:
            return None, bad
        n_given = len(blobs)
        bad = walk(row.get("Text Reasoning Trace") or "", supervise=True, generate=True)
        if bad:
            return None, bad
        add_text(str(row.get("Final Answer") or ""), supervise=True)

        if len(blobs) == n_given:
            return None, "no reasoning image: nothing for the visual generation model to learn"
        if not any(it[0] == "text" and it[2] for it in items):
            return None, "no supervised text: the trace tokenized to nothing"
        # a row whose every generated figure is a copy of a given one is dropped
        given = set(blobs[:n_given])
        drawn = blobs[n_given:]
        if given and all(b in given for b in drawn):
            return None, "every drawn figure is a copy of a given one"
        return items, blobs

    def read(self, unit, rng=None):
        f_idx, group = unit
        handle = self._handles.get(f_idx)
        if handle is None:
            handle = self._handles[f_idx] = pq.ParquetFile(self.files[f_idx])
        table = handle.read_row_group(group)
        rows = table.to_pylist()

        order = list(range(len(rows)))
        if rng is not None:
            rng.shuffle(order)

        name = _zebra_name(self.subset)
        out = []
        for i in order:
            items, blobs = self._sequence(rows[i])
            if items is None:
                drop(f"{name}/row", blobs)
                continue
            out.append(_raw(f"{f_idx:04d}{group:05d}{i:04d}", items, blobs, trim=False))
        return out


# Reconstruction: an image is given and the same image is asked for back with the fixed `instruction` (the eval's
# request too). The ImageNet source reads the train split; the eval set is separate (`t2i_dataset`).
_IMAGENET = dataset_path("imagenet-1k")


class ImageNetReconUnits:
    """`[image(given), instruction, the same image(generate)]` from ImageNet's train parquets, one row group per
    unit. The image is one blob referenced twice. Not trimmed (`trim=False`)."""

    def __init__(self, instruction: str, path: str = _IMAGENET, split: str = "train"):
        assert instruction and instruction.strip(), \
            "the reconstruction source needs the instruction the eval conditions on"
        self.instruction = instruction.strip()
        self.path, self.split = path, split
        self.files = sorted(glob.glob(os.path.join(path, "data", f"{split}-*.parquet")))
        assert self.files, f"no ImageNet {split} parquet under {path}/data"
        self.units, self.n_rows = [], 0
        for i, f in enumerate(self.files):
            md = pq.ParquetFile(f).metadata
            self.units += [(i, g) for g in range(md.num_row_groups)]
            self.n_rows += md.num_rows
        self._handles: dict = {}

    def __len__(self):
        return len(self.units)

    def describe(self) -> str:
        return (f"{len(self.files)} parquet / {len(self.units):,} row groups, "
                f"{self.n_rows:,} samples, {self.n_rows:,} targets, 1 per sample, "
                f"RECONSTRUCTION ({self.split} split; the eval reads validation)")

    def read(self, unit, rng=None):
        f_idx, group = unit
        h = self._handles.get(f_idx)
        if h is None:
            h = self._handles[f_idx] = pq.ParquetFile(self.files[f_idx])
        col = h.read_row_group(group, columns=["image"]).column("image")
        order = list(range(len(col)))
        if rng is not None:
            rng.shuffle(order)
        out = []
        for i in order:
            v = col[i].as_py()
            blob = v["bytes"] if isinstance(v, dict) else v
            if not blob:
                drop("imagenet-recon/row", "empty image bytes")
                continue
            items = [("image", 0, False), ("text", self.instruction, False), ("image", 0, True)]
            out.append(_raw(f"{f_idx:04d}{group:05d}{i:04d}", items, [blob], trim=False))
        return out


_B3O_RECON = {"blip3o-long-recon": dataset_path("blip3o-long-caption"),
              "blip3o-short-recon": dataset_path("blip3o-short-caption")}


class TarStreamUnits:
    """Webdataset tar shards, one shard per unit, streamed from the first member (no index).

    `read` returns a generator of samples, shuffled in chunks of `CHUNK`. The tar is read in stream mode (`r|*`), so
    a member is read as the walk reaches it; members whose extension is not in `IMG_EXTS` / `SIDE_EXTS` are skipped.
    Subclasses define `_sample` (one tar entry -> a raw sample) and `_shape` (its description).
    """

    CHUNK = 100
    IMG_EXTS = ()
    SIDE_EXTS = ()

    def __init__(self, root: str, name: str):
        self.root, self.name = root, name
        found = sorted(glob.glob(os.path.join(root, "*.tar")))
        # some datasets keep part of their shards in `remaining/`
        found += sorted(glob.glob(os.path.join(root, "remaining", "*.tar")))
        assert found, f"{name}: no .tar shards in {root}"
        self.tars = found
        self._all = list(range(len(self.tars)))
        self._mine = None
        self.n_rejected = 0

    @property
    def units(self):
        """The shards this (rank, dataloader worker) may open: every `world * workers`-th shard, so no two workers
        stream the same shard (all shards if there are fewer shards than workers). Computed when asked, inside the
        worker."""
        rank, world, worker, nworkers = pytorch_worker_info()
        gw = int(rank) * max(int(nworkers), 1) + int(worker)
        gn = max(int(world), 1) * max(int(nworkers), 1)
        key = (gw, gn)
        if self._mine is None or self._mine[0] != key:
            mine = self._all[gw::gn] if gn > 1 else self._all
            self._mine = (key, mine or self._all)
        return self._mine[1]

    def describe(self) -> str:
        return (f"{len(self.tars):,} tars, STREAMED (no index, no prefix cap), "
                f"{self._shape()}")

    def _shape(self) -> str:
        raise NotImplementedError

    def _sample(self, shard: str, key: str, parts: dict, rng):
        """One grouped tar entry -> a raw sample, or None to drop it."""
        raise NotImplementedError

    def read(self, unit, rng=None):
        return self._stream(self.tars[unit], rng)

    def _stream(self, path, rng):
        want = set(self.IMG_EXTS) | set(self.SIDE_EXTS)
        shard = os.path.basename(path)
        chunk, cur, parts = [], None, {}
        try:
            tf = tarfile.open(path, "r|*")
        except Exception as exc:                                  # noqa: BLE001
            drop(f"{self.name}/unit", f"{shard}: open failed: {type(exc).__name__}: {exc}")
            return
        # Closed in `finally` (an abandoned generator closes during teardown, where stream-mode tarfile's own close
        # raises). A shard that ends early keeps the samples read before the damage.
        try:
            try:
                for m in tf:
                    if not m.isfile():
                        continue
                    key, _, ext = m.name.rpartition(".")
                    ext = ext.lower()
                    if ext not in want:
                        continue
                    if key != cur:
                        if cur is not None:
                            s = self._sample(shard, cur, parts, rng)
                            if s is not None:
                                chunk.append(s)
                        if len(chunk) >= self.CHUNK:
                            if rng is not None:
                                rng.shuffle(chunk)
                            yield from chunk
                            chunk = []
                        cur, parts = key, {}
                    try:
                        parts[ext] = tf.extractfile(m).read()
                    except Exception as exc:                      # noqa: BLE001
                        drop(f"{self.name}/file",
                             f"{shard}: member read failed: {type(exc).__name__}")
            except Exception as exc:                              # noqa: BLE001
                drop(f"{self.name}/unit",
                     f"{shard}: shard ended early: {type(exc).__name__}: {exc}")
            else:
                # the last entry, only when the walk finished (after an error it may be cut off)
                if cur is not None:
                    s = self._sample(shard, cur, parts, rng)
                    if s is not None:
                        chunk.append(s)
        finally:
            try:
                tf.close()
            except Exception:                                     # noqa: BLE001
                pass
        if chunk:
            if rng is not None:
                rng.shuffle(chunk)
            yield from chunk


class Blip3oReconUnits(TarStreamUnits):
    """BLIP-3o pretrain tars as reconstruction samples, `[image(given), instruction, image(same)]`, like
    `ImageNetReconUnits` (the `.txt` captions are not read). Not trimmed (`trim=False`)."""

    IMG_EXTS = ("jpg",)

    def __init__(self, instruction: str, root: str, name: str):
        assert instruction and instruction.strip(), \
            "the reconstruction source needs the instruction the eval conditions on"
        self.instruction = instruction.strip()
        super().__init__(root, name)

    def _shape(self) -> str:
        return "1 target per sample, RECONSTRUCTION (the .txt members are never read)"

    def _sample(self, shard, key, parts, rng):
        blob = parts.get("jpg")
        if not blob:
            self.n_rejected += 1
            return None
        # one blob: index 0 is both the given and the asked-for image
        items = [("image", 0, False), ("text", self.instruction, False), ("image", 0, True)]
        return _raw(f"{self.name}/{shard}/{key}", items, [blob],
                    [f"{shard}:{key}"], trim=False)


# ScaleEdit-12M, re-sharded into webdataset tars (each entry `<key>.src`, `<key>.tgt`, `<key>.txt`, consecutive; the
# images already resized and centre-cropped to `fit_size(w, h, 640 * 640)`). The shard names start with the
# release's category id, e.g. `4.1_movie_poster_text_editing.<parquet>.s007.tar`.
_SCALEEDIT = dataset_path("scaleedit")

# The category groups a config names: (source name, shard-name prefixes).
SCALEEDIT_GROUPS = (
    ("scaleedit-text", ("4.1_", "4.2_", "4.3_", "4.4_")),   # text editing
    ("scaleedit-extract", ("2.5_",)),                         # extract an object onto a plain background
    ("scaleedit-tone", ("1.2_",)),                            # tone
    ("scaleedit-action", ("2.4_",)),                          # action / pose
    ("scaleedit-beautify", ("3.3_",)),                        # portrait retouching
    ("scaleedit-bg", ("1.4_",)),                              # background
    ("scaleedit-comp", ("6.1_",)),                            # two edits in one instruction
)


class ScaleEditUnits(TarStreamUnits):
    """`[image(given), instruction, image(made)]` from tars of `.src` / `.tgt` / `.txt` entries, optionally only the
    shards whose name starts with one of `cats`."""

    IMG_EXTS = ("src", "tgt")
    SIDE_EXTS = ("txt",)

    def __init__(self, root: str = _SCALEEDIT, name: str = "scaleedit", cats: tuple = ()):
        super().__init__(root, name)
        if cats:
            self.tars = [t for t in self.tars
                         if os.path.basename(t).startswith(tuple(cats))]
            assert self.tars, f"{name}: no shard in {root} starts with any of {cats}"
            self._all = list(range(len(self.tars)))
        self.cats = tuple(cats)

    def _shape(self) -> str:
        if self.cats:
            return f"1 target per sample, editing ({', '.join(self.cats)})"
        return "1 target per sample, free-form editing (23 categories, flat shards)"

    def _sample(self, shard, key, parts, rng):
        src, tgt = parts.get("src"), parts.get("tgt")
        ins = (parts.get("txt") or b"").decode("utf-8", "replace").strip()
        if not src or not tgt or not ins:
            self.n_rejected += 1
            return None
        items = [("image", 0, False), ("text", ins, False), ("image", 1, True)]
        return _raw(f"{self.name}/{shard}/{key}", items, [src, tgt],
                    [f"{shard}:{key}:src", f"{shard}:{key}:tgt"], trim=True)


# Nano-consistent-150k (identity-preserving portrait editing), re-sharded in the ScaleEdit tar layout, the shard
# names starting with the task.
_NANO = dataset_path("nano150k")

NANO_GROUPS = (
    ("nano-background", ("background.",)),            # scene replacement, the person unchanged
    ("nano-action", ("action.",)),                    # pose change
    ("nano-time-change", ("time-change.",)),          # period restyling
    ("nano-black-headshot", ("black_headshot.",)),    # black-and-white studio portrait
    ("nano-hairstyle", ("hairstyle.",)),              # hair only
    ("nano-sweet-headshot", ("sweet_headshot.",)),    # warm portrait retouching
)


class NanoUnits(ScaleEditUnits):
    """Nano-consistent-150k's shards of one task (the ScaleEdit layout and reader)."""

    def __init__(self, name: str, tasks: tuple):
        super().__init__(root=_NANO, name=name, cats=tasks)

    def _shape(self) -> str:
        return f"1 target per sample, identity-preserving portrait editing ({self.cats[0][:-1]})"


class CaptionTarUnits(TarStreamUnits):
    """`[caption, image(made)]` from tars of `.jpg` / `.txt` entries."""

    IMG_EXTS = ("jpg",)
    SIDE_EXTS = ("txt",)

    def __init__(self, name: str, root: str):
        super().__init__(root, name)

    def _shape(self) -> str:
        return "1 target per sample, t2i (captions read as the stream passes them)"

    def _sample(self, shard, key, parts, rng):
        blob, raw = parts.get("jpg"), parts.get("txt")
        if not blob or raw is None:
            self.n_rejected += 1
            return None
        cap = raw.decode("utf-8", "replace").strip()
        if not cap:
            drop(f"{self.name}/row", "empty caption")
            return None
        items = [("text", cap, False), ("image", 0, True)]
        return _raw(f"{self.name}/{shard}/{key}", items, [blob],
                    [f"{shard}:{key}"], trim=True)


# OpenGPT-4o-Image, re-sharded into webdataset tars whose names start with the group: text-to-image (five groups,
# `.jpg` / `.txt`), single-input editing (`edit.`, the ScaleEdit layout) and multi-reference editing (`multiref.`,
# `.src` / `.ref` / `.tgt` / `.txt`).
_OPENGPT = dataset_path("opengpt4o")

OPENGPT4O_GEN_GROUPS = (
    ("opengpt4o-style", ("style.",)),        # a named style applied to a described scene
    ("opengpt4o-science", ("science.",)),    # scientific diagrams and illustrations
    ("opengpt4o-spatial", ("spatial.",)),    # spatial relations, counting, size comparison
    ("opengpt4o-complex", ("complex.",)),    # several requirements in one prompt
    ("opengpt4o-text", ("text.",)),          # typography and text layout
)


class OpenGPT4oT2IUnits(CaptionTarUnits):
    """OpenGPT-4o-Image text-to-image shards of one group (name prefix in `groups`)."""

    def __init__(self, name: str, groups: tuple):
        super().__init__(name, root=_OPENGPT)
        self.tars = [t for t in self.tars
                     if os.path.basename(t).startswith(tuple(groups))]
        assert self.tars, f"{name}: no shard in {_OPENGPT} starts with any of {groups}"
        self._all = list(range(len(self.tars)))
        self.groups = tuple(groups)

    def _shape(self) -> str:
        return f"1 target per sample, t2i ({self.groups[0][:-1]})"


class OpenGPT4oEditUnits(ScaleEditUnits):
    """OpenGPT-4o-Image single-input editing shards (the ScaleEdit layout and reader)."""

    def __init__(self, name: str = "opengpt4o-edit"):
        super().__init__(root=_OPENGPT, name=name, cats=("edit.",))

    def _shape(self) -> str:
        return "1 target per sample, editing (GPT-4o output, 51 subtasks, unlabelled)"


class OpenGPT4oMultiRefUnits(TarStreamUnits):
    """`[image(source), image(reference), instruction, image(made)]` from OpenGPT-4o-Image's multi-reference shards.
    Not trimmed (`trim=False`)."""

    IMG_EXTS = ("src", "ref", "tgt")
    SIDE_EXTS = ("txt",)

    def __init__(self, name: str = "opengpt4o-multiref"):
        super().__init__(_OPENGPT, name)
        self.tars = [t for t in self.tars
                     if os.path.basename(t).startswith("multiref.")]
        assert self.tars, f"{name}: no multiref shard in {_OPENGPT}"
        self._all = list(range(len(self.tars)))

    def _shape(self) -> str:
        return "1 target per sample, multi-reference editing (source + reference)"

    def _sample(self, shard, key, parts, rng):
        src, ref, tgt = parts.get("src"), parts.get("ref"), parts.get("tgt")
        ins = (parts.get("txt") or b"").decode("utf-8", "replace").strip()
        if not src or not ref or not tgt or not ins:
            self.n_rejected += 1
            return None
        items = [("image", 0, False), ("image", 1, False),
                 ("text", ins, False), ("image", 2, True)]
        return _raw(f"{self.name}/{shard}/{key}", items, [src, ref, tgt],
                    [f"{shard}:{key}:src", f"{shard}:{key}:ref", f"{shard}:{key}:tgt"],
                    trim=False)


RECON_SOURCE = "imagenet-recon"
# The sources whose text is the run's `instruction`; `build_edit_units` requires it for them.
RECON_SOURCES = frozenset({RECON_SOURCE} | set(_B3O_RECON))


# TextSculpt-Data: text-to-image with rendered lettering named in the caption, parquet read in place (a row group
# per unit).
_TEXTSCULPT = dataset_path("textsculpt")

# (source name, directory, caption column)
TEXTSCULPT_SUBSETS = (("textsculpt-long", "long_text_rendering", "prompt"),
                      ("textsculpt-qwen", "text_rendering_qwen_image", "re_caption"))


class TextSculptUnits:
    """`[caption, image(made)]` from one TextSculpt subset, a row group per unit. The image column is a Hugging Face
    `Image` feature (`{"bytes", "path"}`)."""

    def __init__(self, name: str, subset: str, caption_col: str, root: str = _TEXTSCULPT):
        self.name, self.subset, self.caption_col = name, subset, caption_col
        self.dir = os.path.join(root, subset)
        self.files = sorted(glob.glob(os.path.join(self.dir, "*.parquet")))
        assert self.files, f"{name}: no parquet under {self.dir}"
        self.units, self.n_rows = [], 0
        for i, f in enumerate(self.files):
            md = pq.ParquetFile(f).metadata          # the footer only, not a row group
            self.units += [(i, g) for g in range(md.num_row_groups)]
            self.n_rows += md.num_rows
        self._handles: dict = {}
        self.n_rejected = 0

    def __len__(self):
        return len(self.units)

    def describe(self) -> str:
        return (f"{len(self.files):,} parquet / {len(self.units):,} row groups, "
                f"{self.n_rows:,} samples, scene text rendering (t2i shape)")

    def read(self, unit, rng=None):
        f_idx, group = unit
        h = self._handles.get(f_idx)
        if h is None:
            h = self._handles[f_idx] = pq.ParquetFile(self.files[f_idx])
        t = h.read_row_group(group, columns=["image", self.caption_col])
        img_c, cap_c = t.column("image"), t.column(self.caption_col)
        order = list(range(t.num_rows))
        if rng is not None:
            rng.shuffle(order)
        out = []
        for i in order:
            img = img_c[i].as_py()
            blob = img.get("bytes") if isinstance(img, dict) else img
            path = (img.get("path") or "") if isinstance(img, dict) else ""
            cap = (cap_c[i].as_py() or "").strip()
            if not blob or not cap:
                drop(f"{self.name}/row", "no image bytes" if not blob else "empty caption")
                continue
            out.append(_raw(f"{self.name}/{f_idx:04d}{group:05d}{i:04d}",
                            [("text", cap, False), ("image", 0, True)], [blob],
                            [path], trim=True))
        return out


# The source names a config uses -> an `EditIndexSpec` or a factory of a units object.
EDIT_SOURCES: dict = {
    **{s.name: s for s in GPT_IMAGE_EDIT_PARTS},
    **{s.name: s for s in ECHO_PARTS},
    **{_zebra_name(s): (lambda s=s: ZebraUnits(s)) for s in ZEBRA_SUBSETS},
    **{n: (lambda n=n, c=c: ScaleEditUnits(name=n, cats=c)) for n, c in SCALEEDIT_GROUPS},
    **{n: (lambda n=n, t=t: NanoUnits(name=n, tasks=t)) for n, t in NANO_GROUPS},
    **{n: (lambda n=n, g=g: OpenGPT4oT2IUnits(name=n, groups=g)) for n, g in OPENGPT4O_GEN_GROUPS},
    "opengpt4o-edit": OpenGPT4oEditUnits,
    "opengpt4o-multiref": OpenGPT4oMultiRefUnits,
    # reconstruction: built by `build_edit_units` with the run's instruction
    RECON_SOURCE: ImageNetReconUnits,
    **{n: Blip3oReconUnits for n in _B3O_RECON},
    **{n: (lambda n=n, s=s, c=c: TextSculptUnits(n, s, c)) for n, s, c in TEXTSCULPT_SUBSETS},
}


def build_edit_units(name: str, instruction: str = None):
    """`name` -> a units object with `.units`, `.read(unit, rng)` and `.describe()`. The reconstruction sources
    take `instruction` as their request."""
    entry = EDIT_SOURCES.get(name)
    assert entry is not None, f"unknown edit source '{name}'; known: {sorted(EDIT_SOURCES)}"
    if name in RECON_SOURCES:
        assert instruction, (
            f"'{name}' is in the mix but no instruction was passed. It is what the sample "
            f"asks for, and it has to be the string the eval uses (`instruction` in the config).")
        if name == RECON_SOURCE:
            return ImageNetReconUnits(instruction)
        return Blip3oReconUnits(instruction, _B3O_RECON[name], name)
    if callable(entry):
        return entry()
    return JsonlEditUnits(entry)
