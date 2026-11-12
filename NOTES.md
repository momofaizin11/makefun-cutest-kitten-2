# Notes

Drawing approach: every shape is an antialiased alpha mask (polygon/ellipse drawn
at 3x then LANCZOS-downsampled), composited over the canvas. Outlines are the
silhouette blurred and thresholded outward, so a union of shapes (cup + handle,
head + ears + chest) gets one clean contour with no internal seams.

Choices that came out of looking at renders:
- Steam is drawn *behind* the kitten, so the head occludes it and the three curls
  stay separated instead of merging into a blob at thumbnail size.
- The handle hole is deliberately wide so background shows through it, keeping the
  handle readable as a separate loop.
- Paws have three lobes and two tapered toe creases; fur uses matte shading plus a
  few soft tufts, never a white specular highlight (that read as plastic).
- Strokes are resampled per segment, otherwise tapered lines break into dots.
- Background: deep warm peach + a soft light disc behind the cup for figure/ground
  separation, then a vignette. No randomness anywhere — renders are reproducible.
