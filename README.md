# Skylark GCP Inference Pipeline

This repository contains a generic, manifest-driven inference pipeline for the
Skylark GCP localization assignment. It is designed around the supplied ONNX
model and does not contain assignment rasters, annotations, model weights, or
generated test predictions.

## Assignment assets

The manifests are the source of truth for scene coverage and relative raster
paths. In the supplied package, the development manifest references
`dev_001.tif` through `dev_004.tif`, and the test manifest references
`test_001.tif` and `test_002.tif`; a complete manifest run therefore needs all
six rasters. `dev_004.tif` and `test_002.tif` are useful for a smaller smoke
run, but they are not a substitute for the full manifests. Keep all supplied
rasters, annotations, the ONNX model, manifests, schemas, and checksums in a
private assignment-assets directory outside the public repository.

## Current artifact scope and verification

The separate private `predictions.json` upload was generated for the two selected
scenes `dev_004` and `test_002`. It contains 51 and 41 detections respectively,
passes the supplied closed-world predictions schema, and uses no fields beyond
`schema_version`, `scenes`, `scene_id`, `detections`, `pixel_x`, `pixel_y`,
`longitude`, `latitude`, and `confidence`. The selected-scene output must not be
presented as a full six-scene result unless the assignment owner confirms that
this two-scene scope is authorized.

The verification record for this artifact is:

- `python -m pytest -q`: 10 tests passed;
- CPU inference and the offline Docker run both completed for the two selected
  scenes;
- the Docker result passed the supplied schema and coordinate bounds checks;
- the ZIP was extracted cleanly with the required files at its root and no
  rasters, annotations, model weights, caches, or nested archive.

To validate a generated output with the organizer's schema, use the supplied
schema file privately:

```bash
python -c "import json; from jsonschema import validate; validate(json.load(open('predictions.json')), json.load(open('predictions.schema.json'))); print('schema valid')"
```

The selected-scene Docker invocation is the same offline contract used for
verification, with a private manifest mounted at `/input/manifest.json`:

```bash
docker run --rm \
  --platform linux/amd64 \
  --network none \
  --read-only \
  --tmpfs /tmp:rw,size=2g \
  -v "$ASSET_ROOT:/input:ro" \
  -v "$MODEL_ROOT:/model:ro" \
  -v "$PWD/output:/output" \
  gcp-submission \
  --manifest /input/manifest.json \
  --model /model/gcp_pose.onnx \
  --output /output/predictions.json
```

Do not add extra metadata to `predictions.json`: the supplied schema sets
`additionalProperties` to false.

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
3. Convert the inspected source bands to RGB float32 in `[0, 1]`, apply the
   published centered 640x640 bilinear letterbox with RGB padding `114`, and
   run the supplied ONNX model through the CPU execution provider. The exact
   scale and padding are retained for inverse coordinate mapping.
4. Parse the published dense pose output: box `xywh`, three probability
   channels, and keypoint `xy`. The keypoint is the output location; marker
   classification is not emitted.
5. Apply confidence filtering and within-window NMS, undo the letterbox, map
   keypoints back to full raster pixel coordinates, and consolidate duplicates
   across overlapping windows.
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

The final submission ZIP must have `Dockerfile`, `README.md`, and `solution/`
immediately at its root. `requirements.txt` and other source files may also be
included at that root. Upload `predictions.json` separately in the public-test
predictions field; do not place it inside the submission ZIP. Do not include
rasters, annotations, the ONNX model, virtual environments, `.git`, caches,
tile artifacts, or another nested ZIP.

## Safe GitHub contents

This public repository is source-only. Synthetic tests and generic examples are
safe to publish. Assignment rasters, annotations, ONNX files, derived crops,
assignment-generated predictions, private manifests, checksums, and local logs
must remain private and must never be uploaded.
