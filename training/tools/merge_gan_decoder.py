"""Write a vision tokenizer directory whose pixel decoder comes from a stage-`gan` checkpoint.

The GAN stage trains `pixel_decoder` and nothing else, so the output is the source tokenizer with exactly those tensors
replaced; the vision tower, abstraction fusion, bottleneck, merger and visual generation adapter stay bit-identical.

The output has the source's format (`config.json` + `model.safetensors`, same keys and dtypes). Before anything is
written it checks that the key set and every non-decoder tensor are unchanged and that every trained decoder tensor
differs from the source.

    python tools/merge_gan_decoder.py --src <tokenizer dir> --gan <gan run>/checkpoint-N.pth --dst <new dir>
"""

import argparse
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# What the GAN stage trains: GANTrainingModel.DECODER_PREFIXES without its `model.` prefix.
DECODER_PREFIXES = ("pixel_decoder.",)

FAILS = []


def ok(cond, msg):
    print(f"  [{'ok' if cond else 'FAIL'}] {msg}", flush=True)
    if not cond:
        FAILS.append(msg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="the tokenizer directory the GAN stage started from")
    ap.add_argument("--gan", required=True, help="checkpoint-N.pth from the GAN stage")
    ap.add_argument("--dst", required=True, help="directory to create (must not exist)")
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    if dst.exists():
        sys.exit(f"{dst} already exists; refusing to overwrite. Remove it yourself or pick another name.")

    print(f"src {src}\ngan {args.gan}\ndst {dst}\n")

    orig = load_file(str(src / "model.safetensors"))
    blob = torch.load(args.gan, map_location="cpu", weights_only=False)
    step = blob["step"]
    gan = {k[len("model."):]: v for k, v in blob["model"].items() if k.startswith("model.")}
    del blob
    print(f"source keys {len(orig)}   gan keys {len(gan)}   gan step {step}\n")

    dec_keys = sorted(k for k in orig if k.startswith(DECODER_PREFIXES))
    rest_keys = sorted(k for k in orig if not k.startswith(DECODER_PREFIXES))
    ok(bool(dec_keys), f"pixel decoder tensors in the source: {len(dec_keys)}")
    ok(all(k in gan for k in dec_keys),
       f"every pixel decoder tensor exists in the GAN checkpoint "
       f"({sum(k in gan for k in dec_keys)}/{len(dec_keys)})")

    # The decoder's buffers (computed from the config, not trained) must stay identical; read off a built decoder.
    from integral.configuration_integral import IntegralVisionTokenizerConfig
    from integral.modeling_integral import IntegralPixelDecoder

    cfg = IntegralVisionTokenizerConfig.from_pretrained(str(src))
    probe = IntegralPixelDecoder(cfg)
    buffer_keys = {f"pixel_decoder.{n}" for n, _ in probe.named_buffers()}
    n_probe_params = sum(1 for _ in probe.parameters())
    print(f"  decoder buffers (no gradient): {sorted(buffer_keys) or 'none'}")
    print(f"  expected trained tensors: {n_probe_params} (IntegralPixelDecoder)")
    del probe

    out = {}
    swapped, kept, unchanged = 0, 0, []
    for k, v in orig.items():
        if k in dec_keys:
            g = gan[k]
            if g.shape != v.shape or g.dtype != v.dtype:
                sys.exit(f"{k}: source {tuple(v.shape)}/{v.dtype} vs gan {tuple(g.shape)}/{g.dtype}")
            if torch.equal(g, v):
                unchanged.append(k)
            out[k] = g.clone()
            swapped += 1
        else:
            out[k] = v
            kept += 1

    ok(set(out) == set(orig), f"key set unchanged ({len(out)} keys, swapped {swapped}, kept {kept})")
    ok(all(out[k].shape == orig[k].shape and out[k].dtype == orig[k].dtype for k in orig),
       "every shape and dtype unchanged")
    ok(all(torch.equal(out[k], orig[k]) for k in rest_keys),
       f"the {len(rest_keys)} non-decoder tensors are bit-identical to the source")
    # every trained decoder tensor must differ from the source; the buffers must not
    trained_unchanged = [k for k in unchanged if k not in buffer_keys]
    buffers_changed = [k for k in dec_keys if k in buffer_keys and not torch.equal(orig[k], gan[k])]
    ok(not trained_unchanged,
       f"every trained decoder tensor changed "
       f"(unchanged: {len(trained_unchanged)}){f' {trained_unchanged[:4]}' if trained_unchanged else ''}")
    ok(not buffers_changed,
       f"the config-derived buffers are unchanged ({len(buffer_keys)} buffer(s))")
    n_trained = len(dec_keys) - len(buffer_keys & set(dec_keys))
    ok(n_trained == n_probe_params,
       f"trained decoder tensors = the decoder's parameter count ({n_trained} == {n_probe_params})")

    n_dec = sum(out[k].numel() for k in dec_keys) / 1e6
    n_rest = sum(out[k].numel() for k in rest_keys) / 1e6
    print(f"\n  replaced {n_dec:.1f}M parameters, kept {n_rest:.1f}M")

    if FAILS:
        print(f"\nFAILED {len(FAILS)}; nothing written")
        for f in FAILS:
            print("  -", f)
        sys.exit(1)

    dst.mkdir(parents=True)
    save_file(out, str(dst / "model.safetensors"), metadata={"format": "pt"})
    shutil.copy2(src / "config.json", dst / "config.json")
    (dst / "PROVENANCE.md").write_text(
        f"# {dst.name}\n\n"
        f"The vision tokenizer in `{src}` with its pixel decoder replaced by the GAN stage's step-{step}\n"
        f"checkpoint.\n\n"
        f"- source tokenizer : `{src}`\n"
        f"- GAN checkpoint   : `{args.gan}` (step {step})\n"
        f"- replaced         : {len(dec_keys)} tensors, {n_dec:.1f}M parameters (`pixel_decoder.*`)\n"
        f"- untouched        : {len(rest_keys)} tensors, {n_rest:.1f}M parameters\n\n"
        f"`config.json` is copied verbatim, so the geometry is the source's.\n"
        f"Built by `tools/merge_gan_decoder.py`.\n"
    )

    print(f"\n  wrote {dst}/model.safetensors ({(dst / 'model.safetensors').stat().st_size / 2**30:.2f} GiB)")
    print(f"  wrote {dst}/config.json")
    print(f"  wrote {dst}/PROVENANCE.md")

    back = load_file(str(dst / "model.safetensors"))
    ok(set(back) == set(orig), "re-read: key set matches the source")
    ok(all(torch.equal(back[k], out[k]) for k in orig), "re-read: every tensor round-trips")
    print("\nALL PASS" if not FAILS else "FAILED after write")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
