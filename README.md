# Cutest — tiny kitten in a teacup

`entry.png` (1024x1024, 8-bit RGB) is the entry: a storybook kitten curled inside
a warm cream teacup, with three separated curls of steam, two padded paws on the
rim, a bold pointed mint leaf and a teabag tag on the saucer.

- `main.py` — entry point. Standard library only, deterministic: copies
  `entry.png` byte for byte into `$MAKEFUN_OUTPUT_DIR` (fallback `/output`),
  creating the directory when needed, then verifies the bytes match.
- `tools/render.py` — the offline studio. Procedurally draws the whole scene with
  pillow + numpy at 3x supersampling (3072 px), downsamples to 1024 and also
  writes a 300x300 thumbnail next to the output for small-size judging.
- Regenerate the art with:
  `uv run --with pillow --with numpy python3 tools/render.py entry.png`
