# Skylark GCP Inference Pipeline

This repository contains a generic, manifest-driven inference pipeline for the
Skylark GCP localization assignment. It is designed around the supplied ONNX
model and does not contain assignment rasters, annotations, model weights, or
generated test predictions.

## Runtime contract

The evaluator invokes:

```bash
python -m solution.infer \
  --manifest /input/manifest.json \
  --model /model/gcp_pose.onnx \
  --output /output/predictions.json
```

The Docker image declares the required entrypoint:

```dockerfile
ENTRYPOINT ["python", "-m", "solution.infer"]
```

The implementation reads scene paths relative to the manifest, processes each
GeoTIFF/COG in bounded-memory windows, and writes one schema-shaped JSON output
containing every manifest scene exactly once.

## Pipeline design

1. Inspect raster dimensions, bands, dtype, mask/nodata, CRS, affine transform,
   resolution, and block layout through raster metadata.
2. Read 640x640 windows with configurable overlap. Right and bottom edge windows
   are padded while retaining their valid dimensions and mask.
3. Convert the inspected source bands to RGB float32 in `[0, 1]` and run the
   supplied ONNX model through the CPU execution provider.
4. Parse the published dense pose output: box `xywh`, three probability
   channels, and keypoint `xy`. The keypoint is the output location; marker
   classification is not emitted.
5. Apply confidence filtering and within-window NMS, map keypoints back to full
   raster pixel coordinates, and consolidate duplicates across overlapping
   windows.
6. Apply the raster affine transform and transform the source CRS to WGS84
   longitude/latitude.
7. Validate finite confidence values, pixel bounds, scene coverage, and empty
   scene handling before writing `predictions.json`.

Thresholds and overlap are command-line options so they can be tuned on private
development data without changing the manifest-driven code path.

## Local development

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run pure tests:

```bash
python -m pytest -q
```

Run the pipeline with local assignment inputs:

```bash
python -m solution.infer \
  --manifest /absolute/path/to/data/test/manifest.json \
  --model /absolute/path/to/model/gcp_pose.onnx \
  --output /absolute/path/to/output/predictions.json
```

The model specification is automatically loaded from the model directory when
present. Use `--model-spec` to provide an explicit path.

## Docker verification

Build from the submission root:

```bash
docker build --platform linux/amd64 -t gcp-submission .
```

Run with assignment data mounted read-only and with network disabled:

```bash
docker run --rm \
  --platform linux/amd64 \
  --network none \
  --read-only \
  --tmpfs /tmp:rw,size=2g \
  -v "$ASSIGNMENT_ROOT/data/test:/input:ro" \
  -v "$ASSIGNMENT_ROOT/model:/model:ro" \
  -v "$PWD/output:/output" \
  gcp-submission \
  --manifest /input/manifest.json \
  --model /model/gcp_pose.onnx \
  --output /output/predictions.json
```

The final submission ZIP must have `Dockerfile`, `README.md`,
`predictions.json`, and `solution/` immediately at its root. Do not include
rasters, annotations, the ONNX model, virtual environments, `.git`, caches,
tile artifacts, or another nested ZIP.

## Safe GitHub contents

This public repository is source-only. Synthetic tests and generic examples are
safe to publish. Assignment rasters, annotations, ONNX files, derived crops,
assignment-generated predictions, private manifests, checksums, and local logs
must remain private and must never be uploaded.

