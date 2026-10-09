"""Prepare the BabiStories dataset (paper Section 4.1 / Appendix B.1).

Extracts traindataset.txt / valdataset.txt from the multi-volume 7z archive
shipped with facebookresearch/MemoryMosaics, tokenizes with the GPT-2 BPE
(tiktoken) and writes train.bin / val.bin (uint16 memmaps) — the exact format
of the official Library/data/BabiStories/prepare.py.

Run:
  python -m src.prepare_babistories \
      --archive-glob external/MemoryMosaics/BabiStories/data/babistories-dataset.7z.* \
      --out-dir data/babistories
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import tempfile

import numpy as np
import tiktoken
from tqdm import tqdm

CHUNK_TOKENS = 5_000_000  # flush to disk every ~10 MiB of uint16 tokens


def concat_volumes(pattern: str, dest: str) -> str:
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no archive volumes match {pattern}")
    if len(paths) == 1:
        return paths[0]
    print(f"concatenating {len(paths)} volumes -> {dest}")
    with open(dest, "wb") as out:
        for p in tqdm(paths, desc="volumes"):
            with open(p, "rb") as f:
                while chunk := f.read(1 << 22):
                    out.write(chunk)
    return dest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive-glob", default="external/MemoryMosaics/BabiStories/data/babistories-dataset.7z.*")
    ap.add_argument("--out-dir", default="data/babistories")
    ap.add_argument("--max-stories", type=int, default=0, help="0 = all (train: 2.2M stories)")
    ap.add_argument("--splits", default="train,val", help="comma-separated subset of splits to (re)build")
    args = ap.parse_args()

    splits = args.splits.split(",")
    os.makedirs(args.out_dir, exist_ok=True)
    import py7zr

    with tempfile.TemporaryDirectory() as tmp:
        archive = concat_volumes(args.archive_glob, os.path.join(tmp, "babistories.7z"))
        extract_dir = os.path.join(tmp, "extracted")
        with py7zr.SevenZipFile(archive, mode="r") as z:
            z.extractall(path=extract_dir)

        enc = tiktoken.get_encoding("gpt2")
        for split in splits:
            src = os.path.join(extract_dir, f"{split}dataset.txt")
            out_path = os.path.join(args.out_dir, f"{split}.bin")
            print(f"tokenizing {split} ({src}) ...")
            # stream tokens to disk in ~5M-token chunks: the full train split is
            # 475M tokens, holding it in a Python list needs >25 GB of RAM
            total = 0
            buf: list[np.ndarray] = []
            buf_n = 0
            with open(src) as f, open(out_path, "wb") as fout:
                for i, story in enumerate(tqdm(f, desc=f"{split} stories")):
                    if args.max_stories and i >= args.max_stories:
                        break
                    ids = enc.encode_ordinary(json.loads(story))
                    ids.append(enc.eot_token)
                    arr = np.asarray(ids, dtype=np.uint16)
                    buf.append(arr)
                    buf_n += arr.size
                    total += arr.size
                    if buf_n >= CHUNK_TOKENS:
                        fout.write(np.concatenate(buf).tobytes())
                        buf, buf_n = [], 0
                if buf:
                    fout.write(np.concatenate(buf).tobytes())
            print(f"  {out_path}: {total:,} tokens ({total * 2 / 2**20:.0f} MiB)")

    print("done.")


if __name__ == "__main__":
    main()
