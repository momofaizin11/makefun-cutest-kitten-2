#!/usr/bin/env python3
"""make.fun entry point.

Copies the finished illustration (entry.png, next to this file) byte for byte
into the output directory: $MAKEFUN_OUTPUT_DIR, or /output when unset.
Standard library only, deterministic, no side effects beyond that copy.
"""
import hashlib
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "entry.png")


def main():
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    os.makedirs(out_dir, exist_ok=True)
    dst = os.path.join(out_dir, "entry.png")
    if os.path.abspath(dst) != os.path.abspath(SRC):
        shutil.copyfile(SRC, dst)
    with open(SRC, "rb") as f:
        a = f.read()
    with open(dst, "rb") as f:
        b = f.read()
    if a != b:
        sys.stderr.write("copy verification failed\n")
        return 1
    print("entry.png -> %s (%d bytes, sha256 %s)"
          % (dst, len(b), hashlib.sha256(b).hexdigest()[:16]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
