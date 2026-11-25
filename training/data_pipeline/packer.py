"""Row packing for the visual generation training stream, and per-source counters of the realized mix.

Runs inside each (rank, dataloader worker)'s stream: every worker draws its own samples and fills its own rows.

Every row carries a flow target: a row opens on a sample that generates an image, and the samples read while looking
for it are placed right behind it (nothing read is discarded). A row is filled to `max_len` tokens; a sample is cut at
the row boundary (`segments.render`), which keeps image spans whole and drops a target whose condition does not fit.
The draw never looks at the row, so the realized sample ratio follows the configured `share`s.
"""

from collections import Counter

from torch.utils.data import IterableDataset

from .segments import Image, collate, render
from .t2i_dataset import drop


class RowPacker(IterableDataset):
    """Samples in, packed rows out.

    `stream` is a factory returning an iterable of `(segments, source_name)` at the configured mix; it is called in
    `__iter__`, so the mixer is built (and seeds itself from its worker identity) in the consuming process.

    Two caps besides the row's tokens: `max_targets` (targets per row) and `max_target_latents` (their latents per
    row), 0 = off. A sample that would exceed a cap closes the row and is carried to the next one; the sample that
    opens a row is exempt, so a chain longer than a cap gets a row to itself. The caps are counted off the raw
    segments, an upper bound on what the rendered row keeps.
    """

    MAX_TRIES = 64          # unrenderable samples in a row before the stream is called broken

    def __init__(self, stream, *, max_len: int, pad_id: int, marker,
                 image_token_id: int, vision_start_id: int, vision_end_id: int, geom,
                 report_every: int = 0, max_targets: int = 0, max_target_latents: int = 0,
                 text_loss: bool = True):
        super().__init__()
        assert callable(stream), "RowPacker takes a factory of the sample stream, not an iterable"
        self.stream = stream
        self.max_len = max_len
        self.pad_id = pad_id
        self.marker = marker
        self.image_token_id = image_token_id
        self.vision_start_id = vision_start_id
        self.vision_end_id = vision_end_id
        self.geom = geom
        self.report_every = report_every
        assert max_targets >= 0, f"max_targets {max_targets} must be >= 0 (0 = off)"
        self.max_targets = max_targets
        assert max_target_latents >= 0, \
            f"max_target_latents {max_target_latents} must be >= 0 (0 = off)"
        self.max_target_latents = max_target_latents
        # False when the run has no cross-entropy: `render` cuts each sample after its last target's condition
        self.text_loss = bool(text_loss)

    def _render(self, segments, name, room=0):
        """-> the sample rendered and cut to `room` tokens (0 = `max_len`), or None if nothing of it is read."""
        room = room or self.max_len
        r = render(segments, marker=self.marker, image_token_id=self.image_token_id,
                   vision_start_id=self.vision_start_id, vision_end_id=self.vision_end_id,
                   geom=self.geom, source=name, max_len=room, text_loss=self.text_loss)
        want = sum(1 for s in segments if isinstance(s, Image))
        if len(r.pixels) != want:
            drop(f"{name}/pack",
                 f"cut to {room} tokens: {want} images -> {len(r.pixels)}")

        if len(r) == 0:
            drop(f"{name}/pack",
                 "nothing supervised and no generated image: every token would be forwarded and "
                 "read by nothing")
            return None
        return r

    @staticmethod
    def _has_target(segments) -> bool:
        """Whether the sample generates an image, off the segments."""
        return any(isinstance(s, Image) and s.generate for s in segments)

    @staticmethod
    def _n_targets(segments) -> int:
        """The images the sample generates, off the raw segments."""
        return sum(1 for s in segments if isinstance(s, Image) and s.generate)

    def _target_latents(self, segments) -> int:
        """The latents of the images the sample generates, off the raw segments."""
        p = self.geom.patch
        return sum((int(s.pixels.shape[1]) // p) * (int(s.pixels.shape[2]) // p)
                   for s in segments if isinstance(s, Image) and s.generate)

    def _rendered_latents(self, r) -> int:
        """The latents of the targets a rendered sample kept."""
        p = self.geom.patch
        return sum((int(r.pixels[g].shape[1]) // p) * (int(r.pixels[g].shape[2]) // p)
                   for g in r.gen_image)

    # Returned by `_next` when the next sample would take the row past a cap (None means the stream ended).
    TARGETS_FULL = object()

    def _next(self, it, room=0, pending=None, tgt_room=None, lat_room=None):
        """The next renderable sample, cut to `room` tokens, `TARGETS_FULL`, or None when the stream ends.

        `pending` (raw samples read while opening the row) is drained first. A sample with more targets than
        `tgt_room`, or more target latents than `lat_room` (None = no cap), goes back to the front of `pending`.
        """
        for tries in range(self.MAX_TRIES + 1):
            if pending:
                segments, name = pending.pop(0)
            else:
                nxt = next(it, None)
                if nxt is None:
                    return None
                segments, name = nxt
            if tgt_room is not None and self._n_targets(segments) > tgt_room:
                pending.insert(0, (segments, name))
                return self.TARGETS_FULL
            if lat_room is not None and self._target_latents(segments) > lat_room:
                pending.insert(0, (segments, name))
                return self.TARGETS_FULL
            r = self._render(segments, name, room)
            if r is not None:
                return r
        raise RuntimeError(
            f"{self.MAX_TRIES} samples in a row could not be rendered into a "
            f"{self.max_len}-token row; the stream or max_len is wrong. Failing here rather "
            f"than starving this rank silently.")

    def _open_row(self, it, pending):
        """The first sample that generates an image, from `pending` and then the stream; the ones passed over in the
        stream are appended to `pending`. None when the stream ends."""
        for i, held in enumerate(pending):
            if self._has_target(held[0]):
                return pending.pop(i)
        for _ in range(64 * self.MAX_TRIES):
            nxt = next(it, None)
            if nxt is None:
                return None
            if self._has_target(nxt[0]):
                return nxt
            pending.append(nxt)
        raise RuntimeError(
            f"{64 * self.MAX_TRIES} samples in a row generated no image. Every row needs one, or "
            f"this rank skips the visual generation model while the others all-gather it. Check the "
            f"mix: at least one source must produce generated images.")

    def __iter__(self):
        it = iter(self.stream())
        n_sample, n_token, n_target = Counter(), Counter(), Counter()
        rows = hunts = used = capacity = 0      # `hunts` = rows whose first draw did not generate
        buf, buf_len, buf_tgt = [], 0, 0
        buf_lat = 0                              # the row's target latents, for `max_target_latents`
        pending: list = []
        ended = False

        def take(r):
            nonlocal buf_len, buf_tgt, buf_lat
            buf.append(r)
            buf_len += len(r)
            buf_tgt += r.n_targets
            if self.max_target_latents:
                buf_lat += self._rendered_latents(r)
            n_sample[r.source] += 1
            n_token[r.source] += len(r)
            n_target[r.source] += r.n_targets

        def untake():
            """Pull the last sample back out (a trailing row with no target, at the end of the stream)."""
            nonlocal buf_len, buf_tgt, buf_lat
            r = buf.pop()
            buf_len -= len(r)
            buf_tgt -= r.n_targets
            if self.max_target_latents:
                buf_lat -= self._rendered_latents(r)
            n_sample[r.source] -= 1
            n_token[r.source] -= len(r)
            n_target[r.source] -= r.n_targets
            return r

        def room():
            return self.max_len - buf_len

        def full():
            return buf_len == self.max_len

        def close():
            nonlocal buf, buf_len, buf_tgt, buf_lat, rows, used, capacity
            batch = collate(buf, self.pad_id, self.max_len, self.geom)
            used += buf_len
            capacity += self.max_len
            rows += 1
            buf, buf_len, buf_tgt, buf_lat = [], 0, 0, 0
            return batch

        dry = 0
        while dry < self.MAX_TRIES:
            # Open the row with a sample that generates an image and keeps a target when rendered into an empty row.
            if not ended:
                for tries in range(self.MAX_TRIES + 1):
                    head = self._open_row(it, pending)
                    if head is None:
                        ended = True
                        break
                    hunts += bool(pending)
                    r = self._render(head[0], head[1], room())
                    if r is not None and r.n_targets >= 1:
                        take(r)
                        break
                    drop(f"{head[1]}/pack",
                         f"generates an image, but no target survives a cut to {self.max_len} "
                         f"tokens: it cannot open a row")
                    if tries >= self.MAX_TRIES:
                        raise RuntimeError(
                            f"{self.MAX_TRIES} samples in a row generated an image whose target "
                            f"did not survive a cut to {self.max_len} tokens (last from "
                            f"'{head[1]}'). Every row needs a target, or this rank skips the "
                            f"visual generation model while the others all-gather it.")

            # Fill to the token, each sample cut to the room left, until the row is full or a cap closes it.
            while not full():
                r = self._next(it, room(), pending,
                               tgt_room=(self.max_targets - buf_tgt) if self.max_targets else None,
                               lat_room=((self.max_target_latents - buf_lat)
                                         if self.max_target_latents else None))
                if r is self.TARGETS_FULL:
                    break
                if r is None:
                    ended = True
                    break
                take(r)

            if buf and buf_tgt == 0:
                # the stream ended before a target could be found; these cannot form a legal row
                drop("pack/tail", f"{len(buf)} sample(s) with no target at the end of the stream")
                while buf:
                    untake()
                dry += 1
            elif buf:
                dry = 0
                yield close()
                if self.report_every and rows % self.report_every == 0:
                    self._report(rows, hunts, n_sample, n_token, n_target, used, capacity)
            else:
                dry += 1

            if ended:
                return
        raise RuntimeError(
            f"{self.MAX_TRIES} consecutive passes closed no row: every one of them ended with "
            f"samples that carry no flow target. Failing here rather than spinning silently.")

    @staticmethod
    def _report(rows, reordered, n_sample, n_token, n_target, used, capacity):
        """Print the packing statistics: fill, rows that needed reordering, and each source's share of the samples,
        tokens and targets."""
        s, t, g = sum(n_sample.values()), sum(n_token.values()), sum(n_target.values())
        print(f"[pack] {rows} rows, fill {100 * used / max(capacity, 1):.1f}%, "
              f"reordered {reordered} ({100 * reordered / max(rows, 1):.2f}% of rows), "
              f"{s} samples / {t} tokens / {g} targets", flush=True)
        for name in sorted(n_sample):
            print(f"[pack]   {name:26s} sample {n_sample[name] / max(s, 1):6.3f}  "
                  f"token {n_token[name] / max(t, 1):6.3f}  "
                  f"target {n_target[name] / max(g, 1):6.3f}", flush=True)
