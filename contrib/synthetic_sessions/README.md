# Synthetic sessions: SLEAP models run backward

Three stand-alone tools built on [sleap-nn](https://github.com/talmolab/sleap-nn):

| Script | What it does |
| --- | --- |
| `sleap_backward.py` | Give a SLEAP top-down model pair a skeleton, get an image the models read as that pose. |
| `sleap_synth_session.py` | Build multi-camera test sessions with **exact** ground truth (3D, 2D per camera, identities) that [LUCID](https://github.com/talmolab/luc3d) loads like a real recording. |
| `sleap_negatives_compare.py` | Compare centroid models for hallucination: how much an empty floor spot must change before each model reports an animal there. |

They are not part of the `sleap_io` package and need more than it does (torch,
sleap-nn, OpenCV, SciPy). Run them with sleap-nn's own Python, which has
everything:

```bash
PY=~/.local/share/uv/tools/sleap-nn/bin/python   # after: uv tool install sleap-nn
$PY contrib/synthetic_sessions/sleap_synth_session.py --help
```

or with `uv run contrib/synthetic_sessions/<script>.py` (each script declares its
dependencies).

## How "backward" works

A centroid model maps a frame to one peak per animal; a centered-instance model
maps a crop around each animal to one heatmap per node. Neither is invertible, so
`sleap_backward.py` searches for an image instead: gradient descent on the pixels
against the models' own training loss, through sleap-nn's own preprocessing
(sizematcher, scale, stride padding, centroid crops). The result is what the
models take to be that pose.

What keeps the result looking like an animal rather than noise that drives the
heatmaps just as well (each measured on DPBox `subModels_top`, 5 mice, frame held
out of training):

- **Start from a real animal.** Each pose starts as the best-fitting other
  labeled animal of the same video, cut out and pasted onto the video's median
  background. Starting from the empty arena reached 1.5 px but looks synthetic;
  the exemplar start reaches 1.2–1.5 px and looks real. The real frame itself
  reads back at 1.9 px against its hand labels.
- **Change only pixels near the animals**, and show each model random
  rotations/rescalings within its own training augmentation ranges.
- **Plain per-pixel Adam over a pyramid** whose finest octave is 2 px.
  Per-octave Adam stalled (instance heatmap error 0.59).

About 1 s/step on an M2 Pro for 650M + 780M-parameter models (0.25 s/step for
the DPBox `OptimizedModels` pair).

The round-trip check reproduces `sleap-nn track` **exactly** (0.000 px against
`run_inference` on CPU). Two details matter for that: sleap-nn 0.2.0 resizes
uint8 before dividing by 255, and `crop_bboxes` reads the crop size off the first
box in float32, so ~12% of frames get 499 px crops instead of 500. The
optimization itself must not call `crop_bboxes`: its backward allocates every
possible crop at once (1.2 TB). Crop targets are relative to the box's half-pixel
top-left corner, sleap-nn's own label frame, not the integer crop start; on 5,493
hand-labeled nodes the start frame is off by −0.66/−0.76 px, the corner frame by
−0.25/−0.34 px.

## Synthetic sessions with exact ground truth

The 3D poses are chosen, so they are known exactly; each camera's 2D is their
projection through the calibration, so it is known exactly too; and each camera's
image is drawn by running the models backward, so the models read that image as
that 2D.

```bash
S=contrib/synthetic_sessions/sleap_synth_session.py

# 1. Match a multi-camera .slp's hand labels across views and triangulate them.
$PY $S triangulate --labels labels.slp --session SESSION_DIR --out labels3d.json

# 1b. Optional: 2D for cameras without labels, from the models' predictions.
$PY $S attach --labels3d labels3d.json --centroid C --instance I \
    --predictions preds/ --out labels3d_all.json

# 1c. Optional: check the body model (what hides what) on real labeled frames.
$PY $S bodies --labels3d labels3d_all.json --frames 3355,926 --out bodies.png

# 2. A scenario: every animal's 3D pose on every frame.
$PY $S scenario --labels3d labels3d_all.json --frame 3355 --groups 3,4 \
    --frames 24 --cross 40 --out scenario.json

# 3. Draw every camera; write the session folder and the truth.
$PY $S generate --scenario scenario.json --labels3d labels3d_all.json \
    --centroid C --instance I --out OUT

# 4. Run the models forward on the generated videos, as `sleap-nn track` does.
$PY $S predict --out OUT --centroid C --instance I
```

`generate` writes, in `OUT/`:

- `session/` — a per-camera session folder: `calibration.toml` (the generated
  cameras' sections, verbatim) and one H.264 `.mp4` per camera, plus
  `<camera>.predictions.slp` after `predict`.
- `truth_session/` — the same videos with the exact 2D as user instances, one
  track per animal.
- `truth.json` — per frame and animal: 3D, 2D in every camera, which nodes are in
  frame, which are hidden behind another animal, and whether the animal was drawn
  in that camera at all.
- `summary.json` — the models' round trip per camera and frame, on the saved PNG
  and on the decoded `.mp4`.
- `work/<camera>/` — per frame: start image, overlay, heatmaps, progress, report.

Calibration conventions are OpenCV's, which are LUCID's (`Camera` in
`pose/pose-data.js`): `rotation` is a Rodrigues vector, `x_cam = R X + t`, then the
k1 k2 p1 p2 k3 lens model, then K. So `cv2.projectPoints` is LUCID's
`reprojectPointCamera`.

### Scenarios

`scenario` turns a labeled frame's animals into poses per frame:

- `--move NAME=DX,DY[,YAW]` — total motion over the sequence along the floor's
  two axes (mm) and about up (degrees), linear in time.
- `--cross MM` — the first two animals swap places, passing `MM` apart, each
  facing the way it walks.
- `--swap-at T [--swap A,B]` — a deliberate identity swap: from frame `T` each
  animal is where the other would have been.

Moving animals as rigid bodies lets them pass **through** each other, so every
frame is checked with the generator's body model: a tail node inside another
body is pushed out to its surface (sideways or over its back, never into the
floor), and bodies overlapping by more than `--max-overlap` (5 mm) refuse the
scenario and name the frames. `--allow-overlap` keeps the poses exactly as moved.

### What the generator does, and why

- **One real animal per synthetic animal, in every camera.** An exemplar is a
  triangulated label group (one animal at one moment, seen by every camera),
  chosen by its summed 2D shape fit over all cameras and distinct per animal, so
  an animal has the same coat in every view. Scale is a soft cost (2 px per 10%,
  up to ±50%): a hard ±15% per camera rejected every group for one mouse with 8
  cameras, since a side camera sees the same mouse 20–40% bigger or smaller as
  it walks. `--exemplars NAME=FRAME:GROUP` pins them, so re-running some cameras
  keeps the same animals. Later frames start from the previous generated frame,
  so a sequence does not flicker; a frame whose poses did not change reuses the
  image.
- **Start close, change little.** Measured on one camera, same exemplars: the
  start animal is bent onto the target by a thin-plate spline through every node
  (`--paste tps`; start loss 0.13 vs 0.20 for one rigid fit); only pixels within
  `--margin` of the drawn skeleton may change (`--mask body`; a disc around the
  centroid is ~3x the body's area and grew a ghost animal in the open arena); and
  `--keep 20` pulls the image toward the start, without which the optimizer
  darkens a halo around every animal to sharpen its peaks. The round trip is
  unchanged (1.14 → 1.16 px median); the image just still looks like the arena.
- **The empty arena must read as empty.** A median background is cleaner than any
  real frame, and on one camera that was enough for a centroid model to read two
  piles of white bedding as animals, on every generated frame (no real labeled
  frame showed an unexplained detection there). Each camera's background is first
  checked with sleap-nn's own inference and, where it sees anything, optimized in
  60 px discs until it does not (mean change 5.6 grey levels where changed).
  Before/after counts land in `truth.json` `background_detections`;
  `--no-clean-background` keeps the median as it is.
- **What is in front is decided in 3D.** Each animal gets a rough body: tapered
  capsules along its skeleton, radii measured from the labels. Midline nodes
  (tail base, trunk, neck, head) carry half the shoulder/haunch width or half the
  ear-to-ear width; nodes labeled on the body's edge (shoulders, haunches, ears,
  nose) a third of that — giving them the full radius made every mouse twice too
  wide at the shoulders. A tail is chained base to tip, even where the skeleton
  joins each tail node to the tail base. Bodies are rendered per camera into a
  depth map, and a node is **hidden** where another animal's body is in front of
  it. Then: animals are pasted far to near; a hidden node still places its
  animal's centroid and crop but is not asked to show (asking for it made the
  optimizer paint texture through the front animal); an animal showing less than
  `--min-visible` (0.3) of its nodes is not drawn in that camera; and a covered
  animal starts from two layers, its last mostly uncovered appearance under the
  previous frame's visible part, so its covered part never comes back as a hole.
  Tails hide nothing, and an animal hiding its own far side is not modelled.
  `--no-occlusion` turns all of this off. Check the body model on your own
  animals with `bodies` before trusting it; `--body-scale` adjusts it.
- **Truth is only what the image can show.** A node the generating models lack
  (a tail, for a pair trained without one) is drawn nowhere, so it is left out of
  the truth.
- **Video:** every frame a keyframe, no B-frames, yuv420p — frame-exact in every
  decoder, browsers included. Grey to limited-range YUV costs ~1.6 grey levels on
  average at `--crf 8`, which is why the round trip is re-measured on the decoded
  video. sleap-io cannot read a one-frame `.mp4`, so a one-frame scenario is held
  for two frames.

### Measured on the DPBox stress test (5 mice, 8 cameras)

- Triangulating the hand labels of 125 mice across the 4 top cameras reprojects
  at a 0.29 px median (worst per-view median 1.7 px).
- LUCID's own Track All + Triangulate All of `truth_session/` recovers the truth
  to 8.5e-8 mm, so LUCID reads the calibration exactly as the generator projects.
- Through the whole pipeline (generated video → sleap-nn predictions → LUCID
  Track All + Triangulate All), 2 mice, one frame: 4 top cameras 1.08 mm median /
  2.9 mm max 3D error; 8 cameras 0.81 / 5.6 mm.
- A 24-frame crossing, 8 cameras, the mice passing 40 mm apart at 0.48 m/s:
  identities kept on every frame; on nodes at least two cameras show, LUCID's
  Refined triangulation 0.92 / 2.8 / 16.9 mm (median / p95 / max), DLT
  1.08 / 6.1 / 67.6 mm. DLT's worst cases are one side camera misreading the
  back mouse's tail tip while the two overlap, which a robust loss discounts.

### Scoring in LUCID

The harness that loads a generated folder in LUCID, runs Track All and
Triangulate All, and scores grouping, 3D, 2D and identities against
`truth.json` lives in the LUCID repository, because it drives LUCID's own code in
a browser: `tests/e2e/_diag-synth-session.mjs`.

```bash
DATASET=OUT node tests/e2e/_diag-synth-session.mjs   # from a LUCID checkout
```

Known limit: LUCID's ID-switch checks need at least 60 s of tracking, so they
cannot be tested on clips this short.

## Comparing centroid models for hallucination

`sleap_negatives_compare.py` compares two or more centroid models, e.g. trained
with and without sleap-nn's `use_negative_frames`. Per model, on one video:

- detections on empty images (the median background, the `.slp`'s
  `is_negative` frames, `--empty-frames` ranges);
- recall and precision on labeled frames (frames in the model's own
  `labels_gt.train*.slp` reported apart), because a model that never fires is
  hard to fool for free;
- the effort to **fool** it: `sleap_backward.py`'s machinery changes a disc
  around an empty floor spot until sleap-nn's own centroid inference reports an
  animal anywhere in the disc, recording the RMS grey-level change it took. A
  model that holds out past `--max-change` gets a lower bound, not a pass.

Every model gets the same spots, start image, random views and step size. A
60-step budget was tried first and was wrong: unfinished optimization read as
"resisted". A holdout whose confidence never rose above 0.05 is reported as "gave
no gradient", never as resisted — a gradient probe cannot tell flat from robust.
Models may have different skeletons: each reads the labels in its own node order.
`--model NAME=DIR` splits at the first `=` whose right side is a folder, because
SLEAP folder names contain `=`.
