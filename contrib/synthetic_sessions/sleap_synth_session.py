#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["sleap-nn>=0.2.0"]
# ///
"""
Synthetic multi-camera sessions with EXACT ground truth, for testing LUCID.

The 3D poses are chosen, so they are known exactly; every camera's 2D is their
projection through the calibration, so it is known exactly too; and each camera's
image is drawn by `sleap_backward.py` — the SLEAP models run backward — so the
models read that image as that 2D. The result is a session folder LUCID loads
like a real one, plus the truth to score it against.

The steps, each its own subcommand:

    PY=~/.local/share/uv/tools/sleap-nn/bin/python

    # 1. Match a multi-camera .slp's hand labels across views (every
    #    assignment is tried; the one whose triangulation reprojects best wins)
    #    and triangulate them: real 3D poses, and the exemplar pool for step 3.
    $PY contrib/synthetic_sessions/sleap_synth_session.py triangulate \\
        --labels labels.slp --session SESSION_DIR --out labels3d.json

    # 1b. Optional: 2D for the cameras the labels lack, from the models'
    #     predictions on the labeled frames, so they get exemplars too (and the
    #     predictions' extra nodes, e.g. a tail, become nodes of the labels3d).
    $PY contrib/synthetic_sessions/sleap_synth_session.py attach --labels3d labels3d.json \\
        --centroid ... --instance ... --predictions preds/ --out labels3d_8cam.json

    # 1c. Optional: check the body model that decides what hides what, drawn
    #     over real labeled frames (outlines should hug the real animals).
    $PY contrib/synthetic_sessions/sleap_synth_session.py bodies --labels3d labels3d.json \\
        --frames 3355 --out bodies.png

    # 2. A scenario: the 3D pose of every animal on every frame. From a labeled
    #    frame (some of its animals), optionally moved, turned or animated.
    $PY contrib/synthetic_sessions/sleap_synth_session.py scenario \\
        --labels3d labels3d.json --frame 3355 --groups 0,2 --out scenario.json

    # 3. Draw every camera and package a LUCID session folder + the truth.
    $PY contrib/synthetic_sessions/sleap_synth_session.py generate \\
        --scenario scenario.json --labels3d labels3d.json --session SESSION_DIR \\
        --centroid models/x.centroid --instance models/y.centered_instance --out OUT

    # 4. Run the models FORWARD on the generated videos, as `sleap-nn track`
    #    does, into each camera folder: the predictions LUCID is tested on.
    $PY contrib/synthetic_sessions/sleap_synth_session.py predict --out OUT \\
        --centroid models/x.centroid --instance models/y.centered_instance

What `generate` writes in --out:

    session/               load in LUCID (File > Load Session Folder, per camera)
      calibration.toml     the generated cameras' sections, verbatim
      <camera>/<camera>.mp4
      <camera>/<camera>.predictions.slp     (after `predict`)
    truth_session/         the same videos with the EXACT 2D as user instances,
      ...                  one track per animal: LUCID's own grouping and
                           triangulation of it must reproduce truth.json
    truth.json             per frame, per animal: 3D, and 2D in every camera
    work/<camera>/         per frame: init, overlay, heatmaps, progress, report
    summary.json           per camera and frame: the models' round-trip error,
                           on the saved PNG and on the decoded .mp4 frame

Conventions are LUCID's (`pose/pose-data.js` `Camera`), which are OpenCV's:
`rotation` is a Rodrigues vector, x_cam = R X + t, then the k1 k2 p1 p2 k3 lens
model, then K. So `cv2.projectPoints` is LUCID's `reprojectPointCamera`.

The same animal starts from the same real mouse in every camera: an exemplar is
a triangulated label GROUP (one real mouse at one moment, seen by every camera),
picked per animal by how well its 2D fits the target in all cameras at once, so
its coat, size and markings agree across views. Later frames of a sequence start
from the previous generated frame instead, moved onto the new pose, so the
animal does not flicker.
"""

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import itertools
import json
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import tomllib

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sleap_backward as sb  # noqa: E402  (exits with a hint outside sleap-nn's env)

cv2, sio, torch = sb.cv2, sb.sio, sb.torch


# ---------------------------------------------------------------------------
# Calibration (anipose TOML), in LUCID's conventions
# ---------------------------------------------------------------------------


@dataclass
class Camera:
    name: str
    K: np.ndarray  # (3, 3)
    dist: np.ndarray  # (5,) k1 k2 p1 p2 k3
    rvec: np.ndarray  # (3,) Rodrigues
    tvec: np.ndarray  # (3,)
    size: tuple  # (W, H)
    section: str = ""  # its TOML section, verbatim

    @property
    def R(self):
        return cv2.Rodrigues(self.rvec)[0]

    @property
    def P(self):
        """Pinhole projection K [R | t] (no lens distortion)."""
        return self.K @ np.hstack([self.R, self.tvec[:, None]])

    @property
    def centre(self):
        return -self.R.T @ self.tvec

    def project(self, X):
        """(..., 3) world -> (..., 2) pixels with lens distortion. NaN in, NaN out."""
        X = np.asarray(X, float)
        flat = X.reshape(-1, 3)
        out = np.full((len(flat), 2), np.nan)
        ok = np.isfinite(flat).all(-1)
        if ok.any():
            uv, _ = cv2.projectPoints(flat[ok], self.rvec, self.tvec, self.K, self.dist)
            out[ok] = uv.reshape(-1, 2)
        return out.reshape(*X.shape[:-1], 2)

    def depth(self, X):
        """(..., 3) world -> (...,) distance in front of the camera (z of x_cam)."""
        X = np.asarray(X, float)
        return (X @ self.R.T + self.tvec)[..., 2]

    def undistort(self, uv):
        """(..., 2) distorted pixels -> ideal pinhole pixels (converged, not 5 steps)."""
        uv = np.asarray(uv, float)
        flat = uv.reshape(-1, 2)
        out = np.full_like(flat, np.nan)
        ok = np.isfinite(flat).all(-1)
        if ok.any():
            crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 200, 1e-14)
            und = cv2.undistortPointsIter(
                flat[ok].reshape(-1, 1, 2), self.K, self.dist, None, self.K, crit
            )
            out[ok] = und.reshape(-1, 2)
        return out.reshape(uv.shape)

    def visible(self, uv, X=None):
        """Which projected points land inside the image (and in front of the camera)."""
        W, H = self.size
        ok = np.isfinite(uv).all(-1) & (uv[..., 0] >= 0) & (uv[..., 0] <= W - 1)
        ok &= (uv[..., 1] >= 0) & (uv[..., 1] <= H - 1)
        if X is not None:
            ok &= self.depth(X) > 0
        return ok


def load_calibration(path):
    """Cameras of an anipose calibration.toml, with each section's text kept verbatim."""
    text = Path(path).read_text()
    data = tomllib.loads(text)
    heads = list(re.finditer(r"^\[([^\]]+)\]\s*$", text, flags=re.M))
    sections = {
        m.group(1): text[
            m.start() : heads[i + 1].start() if i + 1 < len(heads) else len(text)
        ]
        for i, m in enumerate(heads)
    }
    cams = []
    for key, c in data.items():
        if not key.startswith("cam_"):
            continue
        rot = np.asarray(c.get("rotation", [0, 0, 0]), float)
        rvec = cv2.Rodrigues(rot)[0].ravel() if rot.shape == (3, 3) else rot.ravel()
        dist = np.zeros(5)
        d = np.asarray(c.get("distortions", []), float).ravel()
        dist[: len(d)] = d[:5]
        cams.append(
            Camera(
                name=c.get("name", key),
                K=np.asarray(c["matrix"], float),
                dist=dist,
                rvec=rvec,
                tvec=np.asarray(c.get("translation", [0, 0, 0]), float).ravel(),
                size=tuple(int(v) for v in c.get("size", [0, 0])),
                section=sections.get(key, ""),
            )
        )
    return cams, sections.get("metadata", "")


def write_calibration(path, cams, metadata=""):
    """A calibration.toml of just these cameras, each section copied verbatim, so
    every number is bit-identical to the source file's."""
    parts = []
    for i, cam in enumerate(cams):
        body = re.sub(r"^\[[^\]]+\]", f"[cam_{i}]", cam.section.strip(), count=1)
        parts.append(body)
    if metadata.strip():
        parts.append(metadata.strip())
    Path(path).write_text("\n\n".join(parts) + "\n")


# ---------------------------------------------------------------------------
# Triangulation
# ---------------------------------------------------------------------------


def dlt(cams, uv):
    """Linear triangulation of one point. `uv` (V, 2) distorted pixels, NaN = unseen.

    Undistorted, then solved in a frame centred on the cameras and scaled to their
    spread (as LUCID's `triangulatePointDLT` is), so the answer does not depend on
    where the calibration put its origin. Returns (3,), NaN with fewer than 2 views.
    """
    ok = np.isfinite(uv).all(-1)
    if ok.sum() < 2:
        return np.full(3, np.nan)
    centres = np.array([c.centre for c in cams])
    mid = centres.mean(0)
    s = np.linalg.norm(centres - mid, axis=1).mean() or 1.0
    T = np.diag([s, s, s, 1.0])
    T[:3, 3] = mid  # normalized -> world
    rows = []
    for cam, p, good in zip(cams, uv, ok):
        if not good:
            continue
        u, v = cam.undistort(p)
        P = cam.P @ T
        rows += [u * P[2] - P[0], v * P[2] - P[1]]
    _, _, vt = np.linalg.svd(np.array(rows))
    x = T @ vt[-1]
    return x[:3] / x[3]


def refine(cams, uv, X0):
    """Gauss-Newton on the reprojection error (with lens distortion) from X0."""
    from scipy.optimize import least_squares

    ok = np.isfinite(uv).all(-1)
    if ok.sum() < 2 or not np.isfinite(X0).all():
        return X0
    used = [c for c, g in zip(cams, ok) if g]
    obs = uv[ok]

    def resid(X):
        return np.concatenate([c.project(X[None])[0] for c in used]) - obs.ravel()

    return least_squares(resid, X0, method="lm").x


def triangulate_pose(cams, uv, refine_lm=True):
    """(V, K, 2) -> (K, 3)."""
    out = np.full((uv.shape[1], 3), np.nan)
    for k in range(uv.shape[1]):
        X = dlt(cams, uv[:, k])
        out[k] = refine(cams, uv[:, k], X) if refine_lm else X
    return out


def reprojection_error(cams, uv, X):
    """(V, K) px between `uv` (V, K, 2) and the projection of X (K, 3)."""
    return np.stack(
        [np.linalg.norm(c.project(X) - p, axis=-1) for c, p in zip(cams, uv)]
    )


def match_views(cams, poses):
    """Which instance is which animal in every view, by triangulation.

    `poses[v]` is (N, K, 2): the N instances of view v. Every view must have the same
    N. Every assignment is tried (N!^(V-1)); its cost sums, over every PAIR of views
    and every animal, the median over nodes of the two-view triangulation's
    reprojection error. Returns (perms (V, N): perms[v][g] = view v's instance of
    animal g, with view 0 the identity; cost per animal).
    """
    V, N = len(poses), len(poses[0])
    if any(len(p) != N for p in poses):
        raise ValueError(
            f"instance counts differ between views: {[len(p) for p in poses]}"
        )
    if V > 4 or N > 6:
        raise ValueError("exhaustive matching is limited to 4 views of 6 animals")
    # Pairwise costs C[(v, w)][a, b]: instance a of view v with instance b of view w.
    C = {}
    for v, w in itertools.combinations(range(V), 2):
        c = np.full((N, N), 1e6)
        for a in range(N):
            for b in range(N):
                uv = np.stack([poses[v][a], poses[w][b]])
                X = triangulate_pose([cams[v], cams[w]], uv, refine_lm=False)
                err = reprojection_error([cams[v], cams[w]], uv, X)
                if np.isfinite(err).any():
                    c[a, b] = np.nanmedian(err)
        C[(v, w)] = c
    perms = np.array(list(itertools.permutations(range(N))))  # (P, N)
    g = np.arange(N)
    # Total cost over (p_1, ..., p_{V-1}), view 0 fixed to the identity.
    total = 0
    for v, w in C:
        if v == 0:
            term = C[(0, w)][g[None, :], perms].sum(1)  # (P,)
            shape = [1] * (V - 1)
            shape[w - 1] = len(perms)
            total = total + term.reshape(shape)
        else:
            term = C[(v, w)][perms[:, None, :], perms[None, :, :]].sum(-1)  # (P, P)
            shape = [1] * (V - 1)
            shape[v - 1] = shape[w - 1] = len(perms)
            total = total + term.reshape(shape)
    best = np.unravel_index(np.argmin(total), np.shape(total))
    out = np.zeros((V, N), int)
    out[0] = g
    for v in range(1, V):
        out[v] = perms[best[v - 1]]
    return out


def camera_for_video(filename, cams):
    """The calibration camera whose name appears in a video's path."""
    hits = [c for c in cams if c.name and c.name in str(filename)]
    return max(hits, key=lambda c: len(c.name)) if hits else None


def find_video(session_dir, cam_name):
    """The video in SESSION/<cam_name>/ (the one per-camera folder LUCID loads)."""
    d = Path(session_dir) / cam_name
    vids = sorted(
        p for p in d.glob("*") if p.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv")
    )
    vids = [p for p in vids if "calibration" not in p.name.lower()] or vids
    return vids[0] if vids else None


def cmd_triangulate(args):
    cal_path = Path(args.calibration or Path(args.session) / "calibration.toml")
    cams_all, _ = load_calibration(cal_path)
    labels = sio.load_slp(args.labels, open_videos=False)
    skel = labels.skeletons[0]
    nodes = list(skel.node_names)
    edges = [[nodes.index(a), nodes.index(b)] for a, b in skel.edge_names]
    vid_cams = [camera_for_video(v.filename, cams_all) for v in labels.videos]
    used = [(vi, c) for vi, c in enumerate(vid_cams) if c is not None]
    if len(used) < 2:
        raise SystemExit(
            f"fewer than 2 of the .slp's videos match a calibration camera: {vid_cams}"
        )
    print("views:", ", ".join(f"video {vi} = {c.name}" for vi, c in used))
    cams = [c for _, c in used]

    by_frame = {}
    for lf in labels.labeled_frames:
        vi = labels.videos.index(lf.video)
        by_frame.setdefault(lf.frame_idx, {})[vi] = [
            sb.instance_to_array(inst, nodes) for inst in lf.user_instances
        ]
    groups, skipped = [], []
    for fidx in sorted(by_frame):
        per = by_frame[fidx]
        if not all(vi in per and per[vi] for vi, _ in used):
            skipped.append((fidx, "not labeled in every view"))
            continue
        views = [np.stack(per[vi]) for vi, _ in used]
        try:
            perms = match_views(cams, views)
        except ValueError as e:
            skipped.append((fidx, str(e)))
            continue
        for g in range(len(perms[0])):
            uv = np.stack([views[v][perms[v][g]] for v in range(len(cams))])
            X = triangulate_pose(cams, uv)
            err = reprojection_error(cams, uv, X)
            groups.append(
                {
                    "frame": int(fidx),
                    "group": g,
                    "points3d": _nulls(X),
                    "instances": {c.name: int(perms[v][g]) for v, c in enumerate(cams)},
                    "points2d": {c.name: _nulls(uv[v]) for v, c in enumerate(cams)},
                    "reproj_px": {
                        c.name: _round(np.nanmedian(err[v])) for v, c in enumerate(cams)
                    },
                    "reproj_max_px": _round(np.nanmax(err)),
                }
            )
        errs = [max(gr["reproj_px"].values()) for gr in groups if gr["frame"] == fidx]
        print(
            f"  frame {fidx}: {len(perms[0])} animals, worst per-view median reprojection "
            f"{max(errs):.2f} px"
        )
    allerr = np.array([[gr["reproj_px"][c.name] for c in cams] for gr in groups])
    print(
        f"{len(groups)} groups on {len({g['frame'] for g in groups})} frames; per-view median "
        f"reprojection error: median {np.median(allerr):.2f} px, 95th pct "
        f"{np.percentile(allerr, 95):.2f} px, max {allerr.max():.2f} px"
    )
    for f, why in skipped:
        print(f"  skipped frame {f}: {why}")
    out = {
        "labels": str(Path(args.labels).resolve()),
        "calibration": str(cal_path.resolve()),
        "session": str(Path(args.session).resolve()) if args.session else None,
        "nodes": nodes,
        "edges": edges,
        "cameras": [c.name for c in cams],
        "video_index": {c.name: vi for vi, c in used},
        "groups": groups,
        "skipped": skipped,
    }
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"wrote {args.out}")


def _nulls(a):
    a = np.asarray(a, float)
    return [None if not np.isfinite(r).all() else [float(v) for v in r] for r in a]


def _arr(rows, dim):
    return np.array([[np.nan] * dim if r is None else r for r in rows], float)


def _round(x, digits=3):
    return None if x is None or not np.isfinite(x) else round(float(x), digits)


# ---------------------------------------------------------------------------
# Scenario
# ---------------------------------------------------------------------------


def floor_frame(groups, cams):
    """An up direction and a floor height, from where the labeled animals stand.

    Fits a plane through every labeled 3D node (the animals are on the floor, so
    their nodes hug a plane a few cm above it); `up` points from the plane toward
    the cameras. Returns (up (3,), a point on the plane (3,))."""
    pts = np.concatenate([_arr(g["points3d"], 3) for g in groups])
    pts = pts[np.isfinite(pts).all(-1)]
    c = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - c)
    up = vt[-1]
    if np.mean([(cam.centre - c) @ up for cam in cams]) < 0:
        up = -up
    return up, c


def rotation_about(axis, angle):
    """3x3 rotation by `angle` (radians) about unit `axis` (right-handed)."""
    return cv2.Rodrigues(np.asarray(axis, float) * angle)[0]


def floor_axes(up):
    """Two horizontal unit axes (e1, e2) with e1 x e2 = up: world x, flattened, then up x e1."""
    x = np.array([1.0, 0.0, 0.0]) if abs(up[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = x - (x @ up) * up
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(up, e1)


def heading(pose, nodes, up):
    """Unit floor direction the animal faces: tail base (or trunk) toward the nose (or head)."""

    def first(*names):
        for n in names:
            if n in nodes and np.isfinite(pose[nodes.index(n)]).all():
                return pose[nodes.index(n)]
        return None

    back, front = first("TTI", "Tail_Base", "Trunk"), first("Nose", "Head", "Neck")
    if back is None or front is None:
        return None
    h = front - back
    h = h - (h @ up) * up
    return h / np.linalg.norm(h)


def place(pose, up, yaw, shift):
    """A rigid move on the floor: turn by `yaw` (rad) about `up` through the pose's
    centroid, then shift by the 3-vector `shift`."""
    c = np.nanmean(pose, axis=0)
    return (pose - c) @ rotation_about(up, yaw).T + c + shift


def signed_angle(a, b, up):
    """Angle (rad) turning floor direction a onto b about up."""
    return float(np.arctan2(np.cross(a, b) @ up, a @ b))


def segment_penetration(points, prad, pose, edges, radii):
    """How deep `points` (N, 3) with radii `prad` (N,) sink into the capsules on
    `edges` of `pose`: (depth (N,) mm, >0 = inside; the axis point each is nearest
    to (N, 3)). Depth is (capsule radius there + point radius) - distance."""
    depth = np.full(len(points), -np.inf)
    near = np.full((len(points), 3), np.nan)
    for i, j in edges:
        a, b = pose[i], pose[j]
        if not (np.isfinite(a).all() and np.isfinite(b).all()):
            continue
        ab = b - a
        u = np.clip(((points - a) @ ab) / max(ab @ ab, 1e-12), 0, 1)
        q = a + u[:, None] * ab
        d = (
            (radii[i] + u * (radii[j] - radii[i]))
            + prad
            - np.linalg.norm(points - q, axis=1)
        )
        better = d > depth
        depth[better], near[better] = d[better], q[better]
    return depth, near


def separate_bodies(poses, nodes, edges, lab, up, margin=1.0, rounds=6):
    """Keep animals from passing THROUGH each other, which rigid motion allows.

    Measured with the body model the generator draws with (`body_radii`,
    `body_edges`). Tail nodes inside another animal's body are pushed out to its
    surface (plus `margin` mm) — sideways or up over its back, never down into the
    floor — and a tail SEGMENT still passing through takes its two nodes with it.
    Bodies (everything but tails) are only measured: moving them would change the
    scenario asked for, so the caller decides. On the 40 mm crossing, a back
    mouse's tail swept straight through the front mouse, and those nodes were
    hidden in all but at most one camera: truth no image could show.
    Returns (poses, {"tail_nodes_moved", "max_tail_move_mm", "body_overlap_mm"})."""
    radii = body_radii(nodes, lab)
    trunk, tail = body_edges(nodes, edges, lab)
    tail_nodes = sorted(
        {k for e in tail for k in e if nodes[k].lower().startswith("tail")}
    )
    names = list(poses)
    moved, max_move, overlap = set(), 0.0, 0.0
    poses = {n: p.copy() for n, p in poses.items()}
    for a in names:
        for b in names:
            if a == b:
                continue
            P, r = body_samples(poses[a], trunk, radii)
            if len(P):
                overlap = max(
                    overlap,
                    float(np.max(segment_penetration(P, r, poses[b], trunk, radii)[0])),
                )
            for _ in range(rounds):
                # Nodes first, then whatever of the tail chain still runs through.
                pts = poses[a][tail_nodes]
                ok = np.isfinite(pts).all(-1)
                depth, near = segment_penetration(
                    np.where(ok[:, None], pts, 0),
                    np.full(len(pts), TAIL_RADIUS),
                    poses[b],
                    trunk,
                    radii,
                )
                push = {}
                for idx, k in enumerate(tail_nodes):
                    if ok[idx] and depth[idx] > -margin:
                        push[k] = (near[idx], depth[idx] + margin)
                S, _ = body_samples(poses[a], tail, radii, step=1.0)
                if len(S):
                    sd, sn = segment_penetration(
                        S, np.full(len(S), TAIL_RADIUS), poses[b], trunk, radii
                    )
                    for m in np.nonzero(sd > -margin)[0]:
                        # the two tail nodes this sample lies between
                        for i, j in tail:
                            pa, pb = poses[a][i], poses[a][j]
                            ab = pb - pa
                            u = np.clip(((S[m] - pa) @ ab) / max(ab @ ab, 1e-12), 0, 1)
                            if np.linalg.norm(pa + u * ab - S[m]) < 1e-6:
                                for k in (i, j):
                                    if k in tail_nodes and k not in push:
                                        push[k] = (sn[m], sd[m] + margin)
                                break
                if not push:
                    break
                for k, (q, amount) in push.items():
                    v = poses[a][k] - q
                    if v @ up < 0:  # never down into the floor
                        v = v - (v @ up) * up
                    norm = np.linalg.norm(v)
                    v = v / norm if norm > 1e-9 else up
                    poses[a][k] = poses[a][k] + v * amount
                    moved.add((a, k))
                    max_move = max(max_move, float(amount))
    return poses, {
        "tail_nodes_moved": len(moved),
        "max_tail_move_mm": round(max_move, 2),
        "body_overlap_mm": round(max(overlap, 0.0), 2),
    }


def cmd_scenario(args):
    data = json.loads(Path(args.labels3d).read_text())
    groups = [g for g in data["groups"] if g["frame"] == args.frame]
    if not groups:
        raise SystemExit(f"no triangulated groups on frame {args.frame}")
    pick = (
        [int(i) for i in args.groups.split(",")]
        if args.groups
        else [g["group"] for g in groups]
    )
    chosen = [next(g for g in groups if g["group"] == i) for i in pick]
    names = [f"mouse_{i}" for i in range(len(chosen))]
    nodes = data["nodes"]
    base = [_arr(g["points3d"], 3) for g in chosen]
    T = args.frames
    cams, _ = load_calibration(data["calibration"])
    cams = [c for c in cams if c.name in data["cameras"]]
    up, _ = floor_frame(data["groups"], cams)
    e1, e2 = floor_axes(up)

    # Each animal: an initial turn (yaw0) and a total motion over the sequence
    # (shift, yaw), applied linearly in time.
    motion = {n: {"yaw0": 0.0, "shift": np.zeros(3), "yaw": 0.0} for n in names}
    for spec in args.move or []:
        name, vals = spec.split("=")
        v = [float(x) for x in vals.split(",")]
        motion[name]["shift"] = v[0] * e1 + v[1] * e2
        motion[name]["yaw"] = np.deg2rad(v[2]) if len(v) > 2 else 0.0
    if args.cross is not None:
        # The first two animals swap places, passing side by side `cross` mm apart
        # at the midpoint, each facing the way it walks for the whole sequence.
        if len(names) < 2:
            raise SystemExit("--cross needs two animals")
        c0, c1 = (np.nanmean(b, axis=0) for b in base[:2])
        d = c1 - c0
        d -= (d @ up) * up
        side = np.cross(up, d / np.linalg.norm(d))
        for k, (n, sign) in enumerate(((names[0], 1), (names[1], -1))):
            motion[n]["shift"] = sign * d + sign * args.cross * side
            h = heading(base[k], nodes, up)
            if h is not None:
                motion[n]["yaw0"] = signed_angle(
                    h, motion[n]["shift"] / np.linalg.norm(motion[n]["shift"]), up
                )
    swap = None
    if args.swap_at is not None:
        a, b = (args.swap or f"{names[0]},{names[1]}").split(",")
        swap = (args.swap_at, a, b)

    frames, contacts = [], []
    for t in range(T):
        s = t / (T - 1) if T > 1 else 0.0
        poses = {}
        for n, b in zip(names, base):
            m = motion[n]
            p = place(b, up, m["yaw0"], np.zeros(3))
            poses[n] = place(p, up, s * m["yaw"], s * m["shift"])
        if swap and t >= swap[0]:
            # A deliberate identity swap: from here on each animal is where the
            # other would have been. The paths stay continuous, the animals do not.
            poses[swap[1]], poses[swap[2]] = poses[swap[2]], poses[swap[1]]
        if len(poses) > 1 and not args.allow_overlap:
            poses, rep = separate_bodies(poses, nodes, data["edges"], data, up)
            contacts.append(dict(rep, frame=t))
        frames.append({"animals": {n: _nulls(p) for n, p in poses.items()}})
    if contacts:
        moved = [c for c in contacts if c["tail_nodes_moved"]]
        if moved:
            print(
                f"  tails pushed out of the other animal on {len(moved)} frame(s): "
                f"{sum(c['tail_nodes_moved'] for c in moved)} node move(s), up to "
                f"{max(c['max_tail_move_mm'] for c in moved):.1f} mm"
            )
        bad = [c for c in contacts if c["body_overlap_mm"] > args.max_overlap]
        if bad:
            raise SystemExit(
                f"bodies pass through each other by up to {max(c['body_overlap_mm'] for c in bad):.1f} mm "
                f"on frame(s) {[c['frame'] for c in bad][:12]} (tolerance --max-overlap {args.max_overlap} mm): "
                "move the animals further apart, or --allow-overlap"
            )
    steps = [
        np.linalg.norm(
            np.nanmean(_arr(frames[t]["animals"][n], 3), 0)
            - np.nanmean(_arr(frames[t - 1]["animals"][n], 3), 0)
        )
        for t in range(1, T)
        for n in names
    ]
    out = {
        "nodes": nodes,
        "edges": data["edges"],
        "fps": args.fps,
        "animals": names,
        "frames": frames,
        # Never start an animal from the very moment it was taken from.
        "exclude_exemplar_frames": [args.frame],
        "floor": {"up": up.tolist(), "e1": e1.tolist(), "e2": e2.tolist()},
        "swap": {"at": swap[0], "animals": [swap[1], swap[2]]} if swap else None,
        # Per frame: how far bodies overlap, and the tail nodes moved out of another body.
        "contacts": contacts,
        "source": {
            "labels3d": str(Path(args.labels3d).resolve()),
            "frame": args.frame,
            "groups": pick,
            "move": args.move,
            "cross": args.cross,
        },
    }
    Path(args.out).write_text(json.dumps(out, indent=1))
    msg = f"wrote {args.out}: {len(names)} animal(s) x {T} frame(s) from frame {args.frame} groups {pick}"
    if steps:
        msg += f"; centroid moves up to {max(steps):.1f} mm/frame ({max(steps) * args.fps / 1000:.2f} m/s)"
    print(msg)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


@dataclass
class View:
    cam: Camera
    video: Path
    reader: object = None
    bg: object = None  # (1, C, H, W) tensor in [0, 1]
    bg_u8: object = None
    frames: list = field(default_factory=list)  # generated (H, W, C) uint8, per frame


def pick_exemplars(
    scn_poses,
    views,
    pool,
    animals,
    lab_nodes,
    scn_nodes,
    log=print,
    max_zoom=0.5,
    zoom_px=20.0,
):
    """One triangulated label group per animal, fitted in every camera at once.

    `scn_poses[c]` is (A, K, 2): every animal's target 2D in camera c (NaN = off
    image). The cost of group g for animal a is the sum over cameras of the 2D
    shape mismatch after the best similarity (`procrustes`), as `exemplar_init`
    uses within one camera, plus `zoom_px` px per unit of |log scale| (2 px for
    10%): scaling blurs or coarsens the animal. A group that would need more than
    ±max_zoom in any camera is not a candidate — a hard ±15% per camera, as within
    one video, rejected every group for some animals with 8 cameras, since a side
    camera sees the same mouse 20-40% bigger or smaller as it walks. Animals get
    distinct groups (Hungarian).
    """
    from scipy.optimize import linear_sum_assignment

    A = len(animals)
    cost = np.full((A, len(pool)), np.inf)
    for j, g in enumerate(pool):
        for a in range(A):
            total = 0.0
            for c, view in enumerate(views):
                src = group_pose(g, view.cam.name, lab_nodes, scn_nodes)
                fit = sb.procrustes(src, scn_poses[c][a])
                if not sb.zoom_ok(fit, max_zoom):
                    total = np.inf
                    break
                total += fit[4] + zoom_px * abs(np.log(fit[1]))
            cost[a, j] = total
    ok = np.isfinite(cost)
    if not ok.any(axis=1).all():
        bad = [animals[a] for a in range(A) if not ok[a].any()]
        raise SystemExit(f"no labeled animal fits {bad} in every camera")
    rows, cols = linear_sum_assignment(np.where(ok, cost, 1e12))
    picks = {}
    for a, j in zip(rows, cols):
        g = pool[j]
        picks[animals[a]] = g
        log(
            f"  {animals[a]} starts as label group {g['group']} of frame {g['frame']} in every camera "
            f"(shape mismatch {cost[a, j] / len(views):.1f} px per camera)"
        )
    return picks


def background_detections(pair, bg_u8, device):
    """Where `sleap-nn track`'s centroid model finds animals in an image with none."""
    if pair.centroid is None:
        return np.zeros((0, 2))
    u8 = torch.from_numpy(np.ascontiguousarray(bg_u8)).permute(2, 0, 1)[None].to(device)
    found = sb.recover(pair, u8, torch.zeros(0, len(pair.node_names), 2, device=device))
    return found["centroids"] if found["centroids"] is not None else np.zeros((0, 2))


def clean_background(
    pair, bg, bg_u8, device, steps=150, radius=60, keep=20.0, log=print
):
    """The empty arena, changed only where the centroid model sees an animal in it.

    A median background is cleaner than any real frame, and on DPBox sideC that was
    enough for the 8-camera models to read two piles of white bedding as animals —
    on every generated frame, while real frames show no such detection — which put
    an animal in the predictions that the truth does not have, and LUCID grouped it
    into a real mouse. So: discs of `radius` px around each such detection are
    optimized until the centroid map there is empty, pulled toward the original by
    `keep`. Returns (bg tensor, bg uint8, detections before, detections after).
    """
    before = background_detections(pair, bg_u8, device)
    if len(before) == 0:
        return bg, bg_u8, before, before
    H, W = bg.shape[-2:]
    ys, xs = torch.meshgrid(
        torch.arange(H, dtype=torch.float32),
        torch.arange(W, dtype=torch.float32),
        indexing="ij",
    )
    mask = torch.zeros(H, W)
    for x, y in before:
        d = torch.sqrt((xs - float(x)) ** 2 + (ys - float(y)) ** 2)
        mask = torch.maximum(mask, torch.sigmoid((radius - d) / (radius / 6)))
    img = sb.PyramidImage(bg, 6, mask[None, None].to(device), 1)
    opt = torch.optim.Adam(img.parameters(), lr=0.02)
    first = None
    for _ in range(steps):
        x = img()
        sized, _ = sb.size_match(pair, x)
        pred, _ = sb.run_stage(pair.centroid, sized)
        mass = (pred.relu() ** 2).sum()
        first = first or mass.item()
        loss = mass / first + keep * ((x - bg) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    out = img().detach()
    out_u8 = (out * 255).round().to(torch.uint8)[0].permute(1, 2, 0).cpu().numpy()
    after = background_detections(pair, out_u8, device)
    log(
        f"  the empty arena read as {len(before)} animal(s) at {np.round(before).astype(int).tolist()}; "
        f"after cleaning {len(after)}"
    )
    return sb.to_tensor(out_u8, bg.shape[1], device), out_u8, before, after


# ---------------------------------------------------------------------------
# Bodies: which animal is in front, and which nodes it hides
# ---------------------------------------------------------------------------

TAIL_RADIUS = 2.5  # mm: a mouse tail is ~5 mm thick at the base and thinner after


def body_radii(nodes, lab, scale=1.0):
    """A body radius (mm) per node: a crude 3D body to say what hides what.

    MIDLINE nodes carry the body: torso ones half the labels' median
    shoulder-to-shoulder and haunch-to-haunch width, the head half the ear-to-ear
    width. Nodes labeled on the body's EDGE (shoulders, haunches, ears) and the
    nose get a third of that: they sit on the surface, so a full-radius ball
    around one reaches a whole body-width further out — that made every mouse
    ~2x too wide at the shoulders and had the 40 mm crossing's bodies "overlap"
    by 11 mm. Tail nodes take TAIL_RADIUS; any other node is torso. Measured from
    the label groups, so another rig's (or species') animals get their own size.
    """
    ln = lab["nodes"]

    def half_width(a, b):
        if a not in ln or b not in ln:
            return None
        ws = []
        for g in lab["groups"]:
            p = _arr(g["points3d"], 3)
            if np.isfinite(p[[ln.index(a), ln.index(b)]]).all():
                ws.append(np.linalg.norm(p[ln.index(a)] - p[ln.index(b)]))
        return 0.5 * float(np.median(ws)) if ws else None

    widths = [
        w
        for w in (
            half_width("Shoulder_left", "Shoulder_right"),
            half_width("Haunch_left", "Haunch_right"),
        )
        if w
    ]
    torso = float(np.mean(widths)) if widths else 12.0
    head = half_width("Ear_L", "Ear_R") or 0.7 * torso
    edge = (
        "Shoulder_left",
        "Shoulder_right",
        "Haunch_left",
        "Haunch_right",
        "Ear_L",
        "Ear_R",
        "Nose",
    )
    r = []
    for n in nodes:
        if n.lower().startswith("tail"):
            r.append(TAIL_RADIUS)
        elif n in edge:
            r.append((head if n in ("Ear_L", "Ear_R", "Nose") else torso) / 3)
        elif n == "Head":
            r.append(head)
        else:
            r.append(torso)
    return np.array(r) * scale


def body_edges(nodes, edges, lab):
    """(body, tail): the segments the body model is made of.

    `body` is every skeleton edge with no tail node. The tail is re-chained base to
    tip: a skeleton may join each tail node straight to the tail base (DPBox's
    does), which drew the tail as a fan of straight spokes and had them "hide"
    nodes the real, curved tail does not reach. Tail nodes are ordered by their
    median distance from the node they hang off, in the labels."""
    tails = [k for k, n in enumerate(nodes) if n.lower().startswith("tail")]
    body = [(i, j) for i, j in edges if i not in tails and j not in tails]
    hubs = {i if j in tails else j for i, j in edges if (i in tails) != (j in tails)}
    if not tails or not hubs:
        return body, []
    hub = min(hubs)
    ln = lab["nodes"]

    def dist(k):
        if nodes[k] not in ln or nodes[hub] not in ln:
            return k
        d = [
            np.linalg.norm(
                _arr(g["points3d"], 3)[ln.index(nodes[k])]
                - _arr(g["points3d"], 3)[ln.index(nodes[hub])]
            )
            for g in lab["groups"]
        ]
        d = [x for x in d if np.isfinite(x)]
        return float(np.median(d)) if d else float(k)

    chain = [hub] + sorted(tails, key=dist)
    return body, list(zip(chain[:-1], chain[1:]))


def body_samples(pose3d, edges, radii, step=2.0):
    """Points every `step` mm along each skeleton edge, with the body radius there
    (linear between the two nodes'): the axis of a chain of tapered capsules."""
    pts, rs = [], []
    for i, j in edges:
        a, b = pose3d[i], pose3d[j]
        if not (np.isfinite(a).all() and np.isfinite(b).all()):
            continue
        n = max(2, int(np.ceil(np.linalg.norm(b - a) / step)) + 1)
        s = np.linspace(0, 1, n)[:, None]
        pts.append(a + s * (b - a))
        rs.append(radii[i] + s[:, 0] * (radii[j] - radii[i]))
    for k, p in enumerate(pose3d):  # a node on no drawn edge is still body
        if np.isfinite(p).all():
            pts.append(p[None])
            rs.append(radii[k : k + 1])
    if not pts:
        return np.zeros((0, 3)), np.zeros(0)
    return np.concatenate(pts), np.concatenate(rs)


def render_depth(cam, pose3d, edges, radii):
    """(H, W) float32: how far the body's FRONT surface is from `cam` at each pixel
    (inf where the body is not). Discs drawn far to near, so the nearest wins; each
    disc is a capsule sample at its axis depth minus its radius."""
    W, H = cam.size
    depth = np.full((H, W), np.inf, np.float32)
    P, r = body_samples(pose3d, edges, radii)
    if len(P) == 0:
        return depth
    z = cam.depth(P)
    uv = cam.project(P)
    ok = (z > r) & np.isfinite(uv).all(-1)
    order = np.argsort(-z[ok])
    f = float(cam.K[0, 0])
    for (u, v), zz, rr in zip(uv[ok][order], z[ok][order], r[ok][order]):
        rad = f * rr / zz
        if -rad <= u < W + rad and -rad <= v < H + rad:
            cv2.circle(
                depth,
                (int(round(u * 16)), int(round(v * 16))),
                max(1, int(round(rad * 16))),
                float(zz - rr),
                -1,
                lineType=cv2.LINE_8,
                shift=4,
            )
    return depth


def occlusion(cam, X_t, uv_t, vis_t, body, radii):
    """Per animal in one camera at one frame: which nodes ANOTHER animal hides, and
    which animals are nearer than which. `body` is `body_edges`'s (body, tail).

    A node is hidden where another animal's body front lies between it and the
    camera at its pixel. A tail hides nothing (a 5 mm tail over a node leaves it
    readable), though it is part of the silhouette returned. Self-occlusion (a
    mouse's own far shoulder) is not modelled: that is ordinary for every labeled
    side view, and the models read those nodes anyway. Returns (hidden (A, K) bool,
    depth (A, H, W) of the whole body incl. tail, mean depth (A,)).
    """
    A = len(X_t)
    trunk, tail = body
    occluders = np.stack([render_depth(cam, X_t[a], trunk, radii) for a in range(A)])
    depths = np.stack(
        [
            np.minimum(occluders[a], render_depth(cam, X_t[a], tail, radii))
            if tail
            else occluders[a]
            for a in range(A)
        ]
    )
    W, H = cam.size
    hidden = np.zeros(vis_t.shape, bool)
    for a in range(A):
        if A < 2:
            break
        others = np.min(np.delete(occluders, a, axis=0), axis=0)
        z = cam.depth(X_t[a])
        for k in np.nonzero(vis_t[a])[0]:
            u, v = (int(round(c)) for c in uv_t[a, k])
            if 0 <= u < W and 0 <= v < H and others[v, u] < z[k]:
                hidden[a, k] = True
    mean_depth = np.array([np.nanmean(cam.depth(X_t[a])) for a in range(A)])
    return hidden, depths, mean_depth


def generate_frame(
    pair, view, targets, init, opt, device, work, log=print, node_mask=None
):
    """Optimize one camera's image for `targets` (A, K, 2); returns (uint8 image, report).

    `node_mask` (A, K) says which nodes the image must show (default: every finite
    one). A node outside it — hidden behind another animal — still places its
    animal's centroid and crop, as a hidden node's label does in training, but its
    heatmap is not asked for."""
    W, H = view.cam.size
    poses_t = torch.tensor(np.stack(targets), dtype=torch.float32, device=device)
    if node_mask is None:
        node_mask = torch.isfinite(poses_t).all(-1)
    else:
        node_mask = torch.as_tensor(
            np.asarray(node_mask), device=device
        ) & torch.isfinite(poses_t).all(-1)
    mask = sb.edit_mask(
        list(targets),
        (W, H),
        opt.margin,
        device,
        edges=pair.edges if opt.mask == "body" else None,
    )
    image, history, snapshots = sb.optimize(
        pair, init, poses_t, node_mask, opt, mask, log=log
    )
    saved_u8 = (image * 255).round().to(torch.uint8)
    report = round_trip(pair, saved_u8, poses_t, node_mask)
    report["final"] = history[-1] if history else None
    rgb = sb.to_uint8(saved_u8.float() / 255)
    work.mkdir(parents=True, exist_ok=True)
    overlay = rgb.copy()
    for p in targets:
        sb.draw_pose(overlay, p, pair.edges, sb.GREEN)
    for p in report.get("_reported", []):
        sb.draw_pose(overlay, p, pair.edges, sb.MAGENTA, radius=2)
    nm = node_mask.cpu().numpy()
    for p, m in zip(targets, nm):  # hidden target nodes: yellow rings
        for xy, shown in zip(p, m):
            if not shown and np.isfinite(xy).all():
                cv2.circle(overlay, sb._pt(xy), 5, (240, 220, 40), 1, cv2.LINE_AA)
    sb.save_rgb(work / "overlay.png", overlay)
    sb.save_rgb(work / "init.png", sb.to_uint8(init))
    sb.save_rgb(
        work / "heatmaps.png", sb.heatmap_sheet(pair, saved_u8.float() / 255, poses_t)
    )
    sb.save_rgb(work / "progress.png", sb.progress_sheet(snapshots))
    found = report.pop("_reported", None)
    report["history"] = history
    (work / "report.json").write_text(json.dumps(report, indent=1))
    report["found"] = np.stack(found) if found else None
    channels = saved_u8.shape[1]
    out = saved_u8[0].permute(1, 2, 0).cpu().numpy()
    return (out if channels > 1 else out[..., :1]), report


def round_trip(pair, image_u8, poses_t, node_mask):
    """What `sleap-nn track` reads from a uint8 (1, C, H, W) image, vs the targets."""
    found = sb.recover(pair, image_u8, poses_t)
    target = poses_t.cpu().numpy()
    mask = node_mask.cpu().numpy()
    with torch.no_grad():
        terms = sb.objective(pair, image_u8.float() / 255, poses_t, node_mask)
    rep = {
        "heatmap_error": {
            k: terms[k].item() for k in ("centroid", "instance") if k in terms
        }
    }
    if found.get("centroids") is not None:
        rep["animals_found"] = int(len(found["centroids"]))
        if found.get("centroid_dist") is not None:
            rep["centroid_error_px"] = [
                round(float(v), 3) for v in found["centroid_dist"]
            ]
    if "reported" in found:
        err = sb.node_errors(found["reported"], target, mask)
        rep["node_error_px"] = np.where(
            np.isfinite(err), np.round(err, 3), None
        ).tolist()
        rep["median_node_error_px"] = _round(np.nanmedian(err))
        rep["max_node_error_px"] = _round(np.nanmax(err))
        hidden = np.isfinite(target).all(-1) & ~mask
        if hidden.any():
            # Where the models put nodes the image hides: not an error of the image,
            # just what a reader of this view would get.
            herr = np.linalg.norm(found["reported"] - target, axis=-1)[hidden]
            rep["hidden_nodes"] = int(hidden.sum())
            rep["hidden_node_error_px"] = (
                _round(np.nanmedian(herr)) if np.isfinite(herr).any() else None
            )
        rep["node_scores"] = np.round(found["scores"], 3).tolist()
        rep["_reported"] = list(found["reported"])
    return rep


def encode_mp4(frames_dir, out_path, fps, crf):
    """PNG sequence -> H.264 every frame a keyframe, no B-frames (frame-exact in
    every decoder, browsers included), yuv420p (what WebCodecs plays)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(frames_dir / "%06d.png"),
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        str(crf),
        "-g",
        "1",
        "-bf",
        "0",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)


def cmd_generate(args):
    sys.stdout.reconfigure(line_buffering=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    scn = json.loads(Path(args.scenario).read_text())
    lab = json.loads(Path(args.labels3d).read_text())
    session = Path(args.session or lab["session"])
    cams_all, metadata = load_calibration(
        args.calibration or session / "calibration.toml"
    )
    by_name = {c.name: c for c in cams_all}
    names = args.cameras.split(",") if args.cameras else lab["cameras"]
    missing = [n for n in names if n not in by_name]
    if missing:
        raise SystemExit(f"cameras {missing} are not in the calibration")
    views = []
    for n in names:
        video = find_video(session, n)
        if video is None:
            raise SystemExit(f"no video in {session / n}")
        views.append(View(cam=by_name[n], video=video))

    device = sb.pick_device(args.device)
    torch.manual_seed(args.seed)
    print(f"Loading models on {device} ...")
    t0 = time.time()
    pair = sb.load_pair(args.centroid, args.instance, device)
    print(f"  loaded in {time.time() - t0:.0f} s; nodes: {', '.join(pair.node_names)}")

    # The truth is only what the image shows: a node the models do not have (a tail,
    # for a pair trained without one) is not drawn anywhere, so it is not truth.
    dropped = [n for n in scn["nodes"] if n not in pair.node_names]
    if dropped:
        print(f"note: the models have no {dropped}: left out of the truth")
        keep = [k for k, n in enumerate(scn["nodes"]) if n in pair.node_names]
        remap = {k: i for i, k in enumerate(keep)}
        scn["edges"] = [
            [remap[a], remap[b]] for a, b in scn["edges"] if a in remap and b in remap
        ]
        scn["nodes"] = [scn["nodes"][k] for k in keep]
        for f in scn["frames"]:
            f["animals"] = {a: [p[k] for k in keep] for a, p in f["animals"].items()}
    animals = scn["animals"]
    if len(scn["frames"]) == 1:
        # sleap-io (and so `sleap-nn track`) cannot read a one-frame .mp4.
        print(
            "note: a one-frame scenario is held for 2 frames (sleap-io cannot read a 1-frame .mp4)"
        )
        scn["frames"] = scn["frames"] * 2
    T = len(scn["frames"])
    X = np.stack(
        [np.stack([_arr(f["animals"][a], 3) for a in animals]) for f in scn["frames"]]
    )  # (T, A, K, 3)
    # The exact 2D: every node's projection. Off-image nodes stay in the truth, but
    # are "don't care" for the generator, which can only draw what is in frame.
    uv = np.stack([v.cam.project(X) for v in views])  # (C, T, A, K, 2)
    vis = np.stack([v.cam.visible(uv[c], X) for c, v in enumerate(views)])
    targets = np.where(vis[..., None], uv, np.nan)

    # Reorder the scenario's nodes into the model's channel order.
    order = [
        scn["nodes"].index(n) if n in scn["nodes"] else None for n in pair.node_names
    ]
    if any(o is None for o in order):
        print(
            f"  note: model nodes {[n for n, o in zip(pair.node_names, order) if o is None]} "
            "are not in the scenario: left unconstrained"
        )

    def to_model(arr):
        """(..., K_scenario, D) -> (..., K_model, D); NaN for a node the scenario lacks."""
        r = np.full(arr.shape[:-2] + (len(order), arr.shape[-1]), np.nan)
        for k, o in enumerate(order):
            if o is not None:
                r[..., k, :] = arr[..., o, :]
        return r

    targets_m = to_model(targets)
    channels = max(
        s.in_channels for s in (pair.centroid, pair.instance) if s is not None
    )

    # Exemplars: one real mouse per animal, the same in every camera.
    excl = set(scn.get("exclude_exemplar_frames", []))
    pool = [
        g
        for g in lab["groups"]
        if g["frame"] not in excl and all(n in g["points2d"] for n in names)
    ]
    print(f"Exemplar pool: {len(pool)} label groups ({len(excl)} frame(s) excluded)")
    pinned = {}
    for spec in args.exemplars.split(",") if args.exemplars else []:
        name, fg = spec.split("=")
        f, g = (int(v) for v in fg.split(":"))
        hit = [x for x in lab["groups"] if x["frame"] == f and x["group"] == g]
        if not hit:
            raise SystemExit(f"--exemplars: no label group {g} on frame {f}")
        pinned[name] = hit[0]
        print(f"  {name} starts as label group {g} of frame {f} (pinned)")
    free = [a for a in animals if a not in pinned]
    picks = dict(pinned)
    if args.init == "exemplar" and free:
        taken = {(g["frame"], g["group"]) for g in pinned.values()}
        idx = [animals.index(a) for a in free]
        picks.update(
            pick_exemplars(
                [targets[c, 0][idx] for c in range(len(views))],
                views,
                [g for g in pool if (g["frame"], g["group"]) not in taken],
                free,
                lab["nodes"],
                scn["nodes"],
            )
        )

    summary = {
        "cameras": names,
        "animals": animals,
        "frames": T,
        "device": device,
        "exemplars": {
            a: {"frame": g["frame"], "group": g["group"]} for a, g in picks.items()
        },
        "per_camera": {},
    }
    lab_nodes = lab["nodes"]
    radii = body_radii(scn["nodes"], lab, args.body_scale)
    body = body_edges(scn["nodes"], scn["edges"], lab)
    print(
        "Body radii, mm: "
        + ", ".join(f"{n} {r:.1f}" for n, r in zip(scn["nodes"], radii))
    )
    summary["body_radii_mm"] = dict(zip(scn["nodes"], np.round(radii, 2).tolist()))
    # Per camera and frame: which nodes another animal hides (scenario node order),
    # and which animals the image is asked to show at all.
    hidden = np.zeros(vis.shape, bool)  # (C, T, A, K)
    included = np.zeros(vis.shape[:3], bool)  # (C, T, A)
    for c, view in enumerate(views):
        cam = view.cam
        work = out / "work" / cam.name
        frames_dir = work / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{cam.name}] {view.video.name}")
        view.reader = sio.load_video(str(view.video))
        bg_png = work / "background.png"
        if bg_png.exists():
            bg_u8 = sb.load_image(bg_png)
        else:
            t1 = time.time()
            bg_u8 = sb.median_background(view.reader, args.background_frames)
            cv2.imwrite(
                str(bg_png), bg_u8[..., 0] if bg_u8.shape[2] == 1 else bg_u8[..., ::-1]
            )
            print(
                f"  median background of {args.background_frames} frames ({time.time() - t1:.0f} s)"
            )
        bg = sb.to_tensor(bg_u8, channels, device)
        if args.clean_background:
            bg, bg_u8, det0, det1 = clean_background(pair, bg, bg_u8, device)
            if len(det0):
                cv2.imwrite(
                    str(work / "background_clean.png"),
                    bg_u8[..., 0] if bg_u8.shape[2] == 1 else bg_u8[..., ::-1],
                )
        else:
            det0 = det1 = background_detections(pair, bg_u8, device)
            if len(det0):
                print(
                    f"  note: the empty arena reads as {len(det0)} animal(s) at "
                    f"{np.round(det0).astype(int).tolist()} (--clean-background is off)"
                )
        summary["background_detections"] = summary.get("background_detections", {})
        summary["background_detections"][cam.name] = {
            "before": np.round(det0, 1).tolist(),
            "after": np.round(det1, 1).tolist(),
        }
        # Each animal's appearance in this camera: an image showing it UNCOVERED, with
        # its pose there and the other animals' (for the cut). It starts as its real
        # exemplar and moves to each generated frame where nothing is in front of it —
        # never to one where another animal covers part of it, or the part covered
        # would be cut out with it and pasted as a hole once they separate.
        refs = {}
        if args.init == "exemplar":
            for a, name in enumerate(animals):
                g = picks[name]
                refs[a] = {
                    "img": sb.to_tensor(view.reader[g["frame"]], channels, device),
                    "pose": group_pose(g, cam.name, lab_nodes, pair.node_names),
                    "others": [
                        group_pose(o, cam.name, lab_nodes, pair.node_names)
                        for o in lab["groups"]
                        if o["frame"] == g["frame"]
                        and o["group"] != g["group"]
                        and cam.name in o["points2d"]
                    ],
                    "from": -1,
                }
        reports = []
        img = prev = None
        for t in range(T):
            if args.occlusion and len(animals) > 1:
                hid, depths, mdepth = occlusion(
                    cam, X[t], uv[c, t], vis[c, t], body, radii
                )
            else:
                hid = np.zeros(vis[c, t].shape, bool)
                depths, mdepth = None, np.zeros(len(animals))
            hidden[c, t] = hid
            shown = vis[c, t] & ~hid
            frac = shown.sum(-1) / np.maximum(vis[c, t].sum(-1), 1)
            inc = vis[c, t].any(-1) & (frac >= args.min_visible)
            included[c, t] = inc
            hid_m = to_model(hid[..., None].astype(float))[..., 0] == 1
            if (
                t > 0
                and np.array_equal(targets_m[c, t], targets_m[c, t - 1], equal_nan=True)
                and np.array_equal(hidden[c, t], hidden[c, t - 1])
            ):
                # Nothing moved in this camera: the same image, not a re-optimized one.
                cv2.imwrite(
                    str(frames_dir / f"{t:06d}.png"),
                    img[..., 0] if img.shape[2] == 1 else img[..., ::-1],
                )
                reports.append(dict(reports[-1], same_as_frame=t - 1, seconds=0))
                print(f"  [{cam.name} t={t}] same poses as frame {t - 1}: same image")
                continue
            # Paste far to near, so a nearer animal covers a farther one as it would.
            # A covered animal goes on in two layers: its last uncovered appearance,
            # then the previous frame's minus the pixels that were covered there — so
            # what shows follows frame to frame, and only what was hidden is older.
            init = bg.clone()
            for a in np.argsort(-mdepth, kind="stable"):
                if a in refs:
                    r = refs[a]
                    init, _ = sb.paste_animal(
                        init,
                        bg,
                        r["img"],
                        r["pose"],
                        r["others"],
                        targets_m[c, t, a],
                        pair.edges,
                        warp=args.paste,
                    )
                if (
                    prev is not None
                    and a in prev["drawn"]
                    and refs.get(a, {}).get("from") != t - 1
                ):
                    init, _ = sb.paste_animal(
                        init,
                        bg,
                        prev["img"],
                        prev["drawn"][a],
                        [prev["drawn"][b] for b in prev["drawn"] if b != a],
                        targets_m[c, t, a],
                        pair.edges,
                        warp=args.paste,
                        keep_out=prev["covered"].get(a),
                    )
            init = sb.to_tensor(
                sb.to_uint8(init.clamp(0, 1))[..., :channels], channels, device
            )
            opt = argparse.Namespace(**vars(args))
            if t > 0 and args.steps_next is not None:
                opt.steps = args.steps_next
            idx = np.nonzero(inc)[0]
            note = {
                "drawn": [animals[a] for a in idx],
                "hidden_nodes": {
                    animals[a]: int(hid[a].sum())
                    for a in range(len(animals))
                    if hid[a].any()
                },
                "start_from_frame": {animals[a]: refs[a]["from"] for a in refs},
            }
            t1 = time.time()
            if len(idx):
                img, rep = generate_frame(
                    pair,
                    view,
                    list(targets_m[c, t][idx]),
                    init,
                    opt,
                    device,
                    work / f"frame_{t:06d}",
                    log=lambda m: print(f"  [{cam.name} t={t}]{m}"),
                    node_mask=~hid_m[idx],
                )
            else:  # every animal off image or hidden: nothing to optimize
                u8 = (
                    (init * 255)
                    .round()
                    .to(torch.uint8)[0]
                    .permute(1, 2, 0)
                    .cpu()
                    .numpy()
                )
                img, rep = (u8 if channels > 1 else u8[..., :1]), {"found": None}
            rep.update(note)
            rep["seconds"] = round(time.time() - t1, 1)
            hn = sum(note["hidden_nodes"].values())
            print(
                f"  [{cam.name} t={t}] node error median {rep.get('median_node_error_px')} px, "
                f"max {rep.get('max_node_error_px')} px; {rep.get('animals_found')} animal(s) found"
                + (
                    f"; {hn} node(s) hidden, read at {rep.get('hidden_node_error_px')} px"
                    if hn
                    else ""
                )
                + (
                    f"; not drawn: {[animals[a] for a in range(len(animals)) if not inc[a]]}"
                    if not inc.all()
                    else ""
                )
                + f" ({rep['seconds']:.0f} s)"
            )
            cv2.imwrite(
                str(frames_dir / f"{t:06d}.png"),
                img[..., 0] if img.shape[2] == 1 else img[..., ::-1],
            )
            frame_t = sb.to_tensor(img, channels, device)
            found = rep.pop("found", None)
            # Each animal as drawn here: its targets, and where the models found the
            # nodes the target leaves free (a tail the labels lack), so the cut keeps them.
            drawn = {}
            for a in range(len(animals)):
                d = targets_m[c, t, a].copy()
                if a in idx and found is not None:
                    row = found[list(idx).index(a)]
                    d = np.where(np.isfinite(d), d, row)
                drawn[a] = d
            # Which pixels of each animal another one is in front of (dilated a little:
            # the body model is approximate), and whether that is enough to matter.
            cov = {}
            for a in idx:
                if depths is None:
                    break
                mine = np.isfinite(depths[a])
                front = np.zeros_like(mine)
                for b in range(len(animals)):
                    if b != a:
                        front |= depths[b] < depths[a]
                if front.any():
                    cov[a] = (
                        cv2.dilate(front.astype(np.uint8), np.ones((9, 9), np.uint8))
                        > 0
                    )
                if mine.any() and front.sum() / mine.sum() > args.covered:
                    continue
                refs[a] = {
                    "img": frame_t,
                    "pose": drawn[a],
                    "others": [drawn[b] for b in drawn if b != a],
                    "from": t,
                }
            prev = {"img": frame_t, "drawn": {a: drawn[a] for a in idx}, "covered": cov}
            reports.append({k: v for k, v in rep.items() if k != "history"})
        summary["per_camera"][cam.name] = {"png": reports}

    package(
        out,
        views,
        scn,
        X,
        uv,
        vis,
        pair,
        args,
        metadata,
        summary,
        channels,
        device,
        targets_m,
        hidden,
        included,
        to_model,
    )


def _reorder_named(pose, names_from, names_to):
    out = np.full((len(names_to), 2), np.nan)
    for k, n in enumerate(names_to):
        if n in names_from:
            out[k] = pose[names_from.index(n)]
    return out


def package(
    out,
    views,
    scn,
    X,
    uv,
    vis,
    pair,
    args,
    metadata,
    summary,
    channels,
    device,
    targets_m,
    hidden,
    included,
    to_model,
):
    """Encode the videos and write the session folders, the truth and the summary."""
    animals = scn["animals"]
    nodes = scn["nodes"]
    sess = out / "session"
    truth_sess = out / "truth_session"
    for d in (sess, truth_sess):
        d.mkdir(parents=True, exist_ok=True)
        write_calibration(d / "calibration.toml", [v.cam for v in views], metadata)
    skeleton = sio.Skeleton(
        nodes=list(nodes), edges=[(nodes[i], nodes[j]) for i, j in scn["edges"]]
    )
    tracks = [sio.Track(name=a) for a in animals]
    for c, view in enumerate(views):
        cam = view.cam
        mp4 = sess / cam.name / f"{cam.name}.mp4"
        encode_mp4(
            out / "work" / cam.name / "frames", mp4, scn.get("fps", 60), args.crf
        )
        truth_mp4 = truth_sess / cam.name / mp4.name
        truth_mp4.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(mp4, truth_mp4)
        # What the models read from the DECODED video — what `predict` will see.
        video = sio.load_video(str(mp4))
        decoded = []
        for t in range(len(scn["frames"])):
            fr = video[t]
            png = sb.load_image(out / "work" / cam.name / "frames" / f"{t:06d}.png")
            diff = np.abs(fr[..., :1].astype(int) - png[..., :1].astype(int))
            u8 = (
                torch.from_numpy(np.ascontiguousarray(fr[..., :channels]))
                .permute(2, 0, 1)[None]
                .to(device)
            )
            idx = np.nonzero(included[c, t])[0]
            if len(idx) == 0:
                decoded.append({"drawn": []})
                continue
            hid_m = to_model(hidden[c, t][..., None].astype(float))[..., 0] == 1
            poses_t = torch.tensor(
                targets_m[c, t][idx], dtype=torch.float32, device=device
            )
            nm = torch.as_tensor(~hid_m[idx], device=device) & torch.isfinite(
                poses_t
            ).all(-1)
            rep = round_trip(pair, u8, poses_t, nm)
            rep.pop("_reported", None)
            rep["decode_vs_png"] = {
                "max": int(diff.max()),
                "mean": round(float(diff.mean()), 3),
            }
            decoded.append(rep)
            print(
                f"  [{cam.name} t={t}] from the .mp4: node error median "
                f"{rep.get('median_node_error_px')} px (codec changed pixels by <= "
                f"{rep['decode_vs_png']['max']} grey levels, mean {rep['decode_vs_png']['mean']})"
            )
        summary["per_camera"][cam.name]["mp4"] = decoded
        # The truth as one camera's .slp: every in-frame node of every animal (off-image
        # nodes are in truth.json only), one track per animal.
        video = sio.load_video(str(truth_mp4))
        lfs = []
        for t in range(len(scn["frames"])):
            insts = []
            for a, name in enumerate(animals):
                pts = np.where(vis[c, t, a][:, None], uv[c, t, a], np.nan)
                if np.isfinite(pts).any():
                    insts.append(
                        sio.Instance.from_numpy(pts, skeleton=skeleton, track=tracks[a])
                    )
            lfs.append(sio.LabeledFrame(video=video, frame_idx=t, instances=insts))
        sio.Labels(
            labeled_frames=lfs, videos=[video], skeletons=[skeleton], tracks=tracks
        ).save(str(truth_mp4.with_suffix(".slp")), embed=False)
    truth = {
        "cameras": [v.cam.name for v in views],
        "nodes": nodes,
        "edges": scn["edges"],
        "fps": scn.get("fps", 60),
        "animals": animals,
        "units": "calibration world units (mm)",
        "frames": [
            {
                "frame": t,
                "animals": {
                    name: {
                        "points3d": _nulls(X[t, a]),
                        "points2d": {
                            v.cam.name: _nulls(uv[c, t, a]) for c, v in enumerate(views)
                        },
                        "in_frame": {
                            v.cam.name: vis[c, t, a].tolist()
                            for c, v in enumerate(views)
                        },
                        # In frame but behind another animal (its body model): the image
                        # does not show these, so a 2D reading of them is not scored.
                        "hidden": {
                            v.cam.name: hidden[c, t, a].tolist()
                            for c, v in enumerate(views)
                        },
                        # False where so little of it shows that it was not drawn at all.
                        "drawn": {
                            v.cam.name: bool(included[c, t, a])
                            for c, v in enumerate(views)
                        },
                    }
                    for a, name in enumerate(animals)
                },
            }
            for t in range(len(scn["frames"]))
        ],
        "scenario": scn.get("source"),
        # A staged identity swap: from frame `at` on, each listed animal is where the
        # other would have been. The checks should flag it there.
        "swap": scn.get("swap"),
        "exemplars": summary.get("exemplars"),
        "body_radii_mm": summary.get("body_radii_mm"),
        # Where the centroid model saw an animal in a camera's empty arena, before and
        # after --clean-background: an "after" here is a detection with no truth.
        "background_detections": summary.get("background_detections"),
    }
    (out / "truth.json").write_text(json.dumps(truth, indent=1))
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"wrote {sess}/, {truth_sess}/, truth.json, summary.json")


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


def group_pose(g, cam_name, lab_nodes, names):
    """A label group's 2D in one camera, in `names` order."""
    return _reorder_named(_arr(g["points2d"][cam_name], 2), lab_nodes, names)


def cmd_attach(args):
    """Give every label group a 2D pose in more cameras, from the models' predictions.

    The labeled cameras' 3D says where each group must appear in another camera;
    the models are run on the labeled frames of every camera, and each group takes
    the prediction nearest its labels (a labeled camera) or its projection (any
    other), by exhaustive assignment per frame, if it is within --max-px and scores
    at least --min-score. Nodes the predictions have and the labels do not (a tail)
    become nodes of the labels3d too: their 2D from the predictions in every matched
    camera, their 3D triangulated from all of those (`node_sources` says which nodes
    are predictions). The labeled nodes' 2D and 3D are left exactly as they were.
    """
    from scipy.optimize import linear_sum_assignment
    from sleap_nn.predict import run_inference

    lab = json.loads(Path(args.labels3d).read_text())
    session = Path(args.session or lab["session"])
    cams_all, _ = load_calibration(args.calibration or lab["calibration"])
    names = args.cameras.split(",") if args.cameras else [c.name for c in cams_all]
    nodes = lab["nodes"]
    frames = sorted({g["frame"] for g in lab["groups"]})
    pred_dir = Path(args.predictions)
    pred_dir.mkdir(parents=True, exist_ok=True)
    pnodes, pedges = None, []
    for cam in [c for c in cams_all if c.name in names]:
        dest = pred_dir / f"{cam.name}.labeled_frames.slp"
        if not dest.exists():
            video = find_video(session, cam.name)
            print(f"[{cam.name}] predicting {len(frames)} frames of {video.name}")
            # output_path is required: without it sleap-nn writes next to the input video.
            run_inference(
                data_path=str(video),
                model_paths=[args.centroid, args.instance],
                frames=frames,
                output_path=str(dest),
                device=sb.pick_device(args.device),
            )
        L = sio.load_slp(str(dest), open_videos=False)
        pnodes = list(L.skeletons[0].node_names)
        pedges = list(L.skeletons[0].edge_names)
        preds = {
            lf.frame_idx: [
                (sb.instance_to_array(i, pnodes), float(i.score))
                for i in lf.predicted_instances
            ]
            for lf in L.labeled_frames
        }
        labeled = cam.name in lab["cameras"]
        n_att, costs = 0, []
        for f in frames:
            gs = [g for g in lab["groups"] if g["frame"] == f]
            ps = [p for p in preds.get(f, []) if p[1] >= args.min_score]
            if not gs or not ps:
                continue
            # Where each group must be: its labels here, else its 3D projected here.
            want = [
                _reorder_named(_arr(g["points2d"][cam.name], 2), nodes, pnodes)
                if labeled
                else _reorder_named(cam.project(_arr(g["points3d"], 3)), nodes, pnodes)
                for g in gs
            ]
            cost = np.full((len(gs), len(ps)), np.inf)
            for i, w in enumerate(want):
                for k, (pose, _) in enumerate(ps):
                    d = np.linalg.norm(w - pose, axis=-1)
                    if np.isfinite(d).any():
                        cost[i, k] = np.nanmedian(d)
            rows, cols = linear_sum_assignment(np.where(np.isfinite(cost), cost, 1e9))
            for r, k in zip(rows, cols):
                if cost[r, k] > args.max_px:
                    continue
                g = gs[r]
                g.setdefault("_pred2d", {})[cam.name] = ps[k][0]
                if not labeled:
                    g.setdefault("attach_px", {})[cam.name] = _round(cost[r, k])
                n_att += 1
                costs.append(cost[r, k])
        if not labeled and n_att:
            lab["cameras"].append(cam.name)
        print(
            f"  {cam.name}: {n_att} of {len(lab['groups'])} groups matched a prediction "
            f"(median {np.median(costs) if costs else float('nan'):.1f} px from "
            f"{'its labels' if labeled else 'its projection'})"
        )

    # The predictions' extra nodes join the skeleton; the labeled nodes stay as labeled.
    new = [n for n in (pnodes or []) if n not in nodes]
    all_nodes = nodes + new
    sources = lab.get("node_sources") or {n: "labels" for n in nodes}
    sources.update({n: "predictions" for n in new})
    edges = [list(e) for e in lab["edges"]]
    for a, b in pedges:
        if (a in new or b in new) and a in all_nodes and b in all_nodes:
            edges.append([all_nodes.index(a), all_nodes.index(b)])
    by_name = {c.name: c for c in cams_all}
    errs = []
    for g in lab["groups"]:
        pred = g.pop("_pred2d", {})
        for cam_name in lab["cameras"]:
            old = (
                _arr(g["points2d"][cam_name], 2) if cam_name in g["points2d"] else None
            )
            p = (
                _reorder_named(pred[cam_name], pnodes, all_nodes)
                if cam_name in pred
                else None
            )
            if old is None and p is None:
                continue
            full = np.full((len(all_nodes), 2), np.nan)
            if old is not None:
                full[: len(nodes)] = (
                    old  # labeled nodes: the labels, never the predictions
                )
            elif p is not None:
                full[: len(nodes)] = p[: len(nodes)]
            if p is not None:
                full[len(nodes) :] = p[len(nodes) :]
            g["points2d"][cam_name] = _nulls(full)
        if new:
            cams = [by_name[n] for n in lab["cameras"] if n in g["points2d"]]
            uv = np.stack([_arr(g["points2d"][c.name], 2)[len(nodes) :] for c in cams])
            X = np.full((len(new), 3), np.nan)
            for k in range(len(new)):
                # Predictions are wrong now and then (a tail): drop the worst view
                # while it is off by more than --outlier-px and 3 views would remain.
                pts = uv[:, k].copy()
                while True:
                    X[k] = triangulate_pose(cams, pts[:, None])[0]
                    e = reprojection_error(cams, pts[:, None], X[k][None])[:, 0]
                    if not (
                        np.nanmax(e, initial=0) > args.outlier_px
                        and np.isfinite(e).sum() > 3
                    ):
                        break
                    pts[np.nanargmax(e)] = np.nan
                for c, cam in enumerate(cams):
                    if not np.isfinite(pts[c]).all() and np.isfinite(uv[c, k]).all():
                        # The view disagreed: keep the triangulated node's projection there,
                        # so every camera's 2D is consistent with the 3D.
                        uv[c, k] = cam.project(X[k][None])[0]
                errs.extend(e[np.isfinite(e)].tolist())
            for c, cam in enumerate(cams):
                p2 = _arr(g["points2d"][cam.name], 2)
                p2[len(nodes) :] = uv[c]
                g["points2d"][cam.name] = _nulls(p2)
            g["points3d"] = g["points3d"] + _nulls(X)
    lab["nodes"], lab["edges"], lab["node_sources"] = all_nodes, edges, sources
    if new:
        print(
            f"added nodes {new} from the predictions; their triangulation reprojects at a median "
            f"{np.median(errs):.1f} px, 95th pct {np.percentile(errs, 95):.1f} px"
        )
    Path(args.out).write_text(json.dumps(lab, indent=1))
    print(f"wrote {args.out}")


def cmd_bodies(args):
    """Draw the body model over REAL labeled frames, to check it before trusting it.

    Per camera: each label group's capsule outline in its own colour, its nodes as
    dots, and a red ring on every node the body model says another animal hides.
    The outlines should hug the real animals and the rings sit where one covers
    another; if they do not, scale the radii (`generate --body-scale`)."""
    lab = json.loads(Path(args.labels3d).read_text())
    cams, _ = load_calibration(args.calibration or lab["calibration"])
    names = (
        args.cameras.split(",")
        if args.cameras
        else [c.name for c in cams if c.name in lab["cameras"]]
    )
    nodes = lab["nodes"]
    radii = body_radii(nodes, lab, args.body_scale)
    body = body_edges(nodes, lab["edges"], lab)
    print("Body radii, mm: " + ", ".join(f"{n} {r:.1f}" for n, r in zip(nodes, radii)))
    colours = [
        (255, 200, 0),
        (0, 120, 255),
        (0, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
        (255, 255, 255),
    ]
    tiles = []
    for f in [int(x) for x in args.frames.split(",")]:
        gs = [g for g in lab["groups"] if g["frame"] == f]
        if not gs:
            raise SystemExit(f"no label groups on frame {f}")
        X = np.stack([_arr(g["points3d"], 3) for g in gs])
        for cam in [c for c in cams if c.name in names]:
            uv = cam.project(X)
            vis = cam.visible(uv, X)
            hid, depths, _ = occlusion(cam, X, uv, vis, body, radii)
            frame = sio.load_video(str(find_video(lab["session"], cam.name)))[f]
            img = (
                cv2.cvtColor(frame[..., 0], cv2.COLOR_GRAY2BGR)
                if frame.shape[2] == 1
                else frame[..., ::-1].copy()
            )
            for a in range(len(X)):
                col = colours[a % len(colours)]
                cs, _ = cv2.findContours(
                    np.isfinite(depths[a]).astype(np.uint8),
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_NONE,
                )
                cv2.drawContours(img, cs, -1, col, 1)
                for k in np.nonzero(vis[a])[0]:
                    p = sb._pt(uv[a, k])
                    if hid[a, k]:
                        cv2.circle(img, p, 5, (0, 0, 255), 2)
                    else:
                        cv2.circle(img, p, 2, col, -1)
            nz = np.argwhere(np.isfinite(depths).any(0))
            (y0, x0), (y1, x1) = np.maximum(nz.min(0) - 30, 0), nz.max(0) + 30
            crop = img[y0:y1, x0:x1].copy()
            cv2.putText(
                crop,
                f"{cam.name} frame {f}: hidden {hid.sum(1).tolist()}",
                (5, 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            tiles.append(
                cv2.resize(crop, (560, int(560 * crop.shape[0] / crop.shape[1])))
            )
            print(
                f"  frame {f} {cam.name}: hidden nodes per animal {hid.sum(1).tolist()}"
            )
    h = max(t.shape[0] for t in tiles)
    tiles = [np.pad(t, ((0, h - t.shape[0]), (0, 0), (0, 0))) for t in tiles]
    tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % 3)
    cv2.imwrite(
        args.out,
        np.concatenate(
            [np.concatenate(tiles[i : i + 3], 1) for i in range(0, len(tiles), 3)], 0
        ),
    )
    print(f"wrote {args.out}")


def cmd_predict(args):
    from sleap_nn.predict import run_inference

    sess = Path(args.out) / "session"
    for mp4 in sorted(sess.glob("*/*.mp4")):
        dest = mp4.with_suffix(".predictions.slp")
        print(f"[{mp4.parent.name}] -> {dest.name}")
        # output_path is REQUIRED here: without it sleap-nn writes next to the input.
        run_inference(
            data_path=str(mp4),
            model_paths=[args.centroid, args.instance],
            output_path=str(dest),
            device=sb.pick_device(args.device),
            peak_threshold=sb.PEAK_THRESHOLD,
            integral_patch_size=sb.INTEGRAL_PATCH,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[1],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser(
        "triangulate", help="match hand labels across views and triangulate them"
    )
    p.add_argument(
        "--labels", required=True, help="multi-camera .slp (one video per camera)"
    )
    p.add_argument(
        "--session", help="session folder: calibration.toml + <camera>/ video folders"
    )
    p.add_argument(
        "--calibration", help="calibration.toml (default: SESSION/calibration.toml)"
    )
    p.add_argument("--out", required=True, help="labels3d.json")
    p.set_defaults(func=cmd_triangulate)

    p = sub.add_parser(
        "scenario", help="3D poses per animal per frame, from triangulated labels"
    )
    p.add_argument("--labels3d", required=True)
    p.add_argument(
        "--frame", type=int, required=True, help="labeled frame to take the poses from"
    )
    p.add_argument(
        "--groups", help="which of its triangulated animals, e.g. 0,2 (default: all)"
    )
    p.add_argument("--frames", type=int, default=1, help="frames in the scenario")
    p.add_argument("--fps", type=float, default=60)
    p.add_argument(
        "--move",
        action="append",
        metavar="NAME=DX,DY[,YAW]",
        help="total motion of one animal over the sequence: mm along the floor's two "
        "axes (see floor in the output), degrees about up; linear in time. Repeatable",
    )
    p.add_argument(
        "--cross",
        type=float,
        metavar="MM",
        help="the first two animals swap places, passing MM apart, facing the way they walk",
    )
    p.add_argument(
        "--swap-at",
        type=int,
        metavar="T",
        help="from frame T on, two animals exchange poses: a deliberate identity swap",
    )
    p.add_argument("--swap", metavar="A,B", help="which two (default: the first two)")
    p.add_argument(
        "--max-overlap",
        type=float,
        default=5.0,
        metavar="MM",
        help="refuse a scenario whose bodies (the generator's capsule model) pass "
        "this far into each other; real mice squash a little where they touch",
    )
    p.add_argument(
        "--allow-overlap",
        action="store_true",
        help="keep the poses exactly as moved: no tail pushed out, no overlap check",
    )
    p.add_argument("--out", required=True, help="scenario.json")
    p.set_defaults(func=cmd_scenario)

    p = sub.add_parser(
        "generate", help="draw every camera; write the session folder and the truth"
    )
    p.add_argument("--scenario", required=True)
    p.add_argument(
        "--labels3d", required=True, help="from `triangulate`: the exemplar pool"
    )
    p.add_argument(
        "--session",
        help="the REAL session (videos for backgrounds and exemplars); "
        "default: the one in labels3d",
    )
    p.add_argument("--calibration", help="default: SESSION/calibration.toml")
    p.add_argument(
        "--cameras", help="comma-separated camera names (default: labels3d's)"
    )
    p.add_argument("--centroid", required=True)
    p.add_argument("--instance", required=True)
    p.add_argument("--init", choices=["exemplar", "background"], default="exemplar")
    p.add_argument(
        "--exemplars",
        metavar="NAME=FRAME:GROUP,...",
        help="pin animals' exemplars (label groups), e.g. to redo some cameras with the "
        "same mice as a previous run (its summary.json lists them)",
    )
    p.add_argument(
        "--paste",
        choices=["tps", "similarity"],
        default="tps",
        help="how a start animal is moved onto its target: bent by a thin-plate "
        "spline through every node (tps), or rotated/scaled/moved as one piece",
    )
    p.add_argument(
        "--margin",
        type=float,
        default=40,
        help="only pixels within this many px of an animal may change",
    )
    p.add_argument(
        "--mask",
        choices=["body", "disc"],
        default="body",
        help="'body': within --margin of the drawn skeleton; 'disc': of a disc around "
        "the centroid reaching past the farthest node (sleap_backward.py's default)",
    )
    p.add_argument("--background-frames", type=int, default=60)
    p.add_argument(
        "--no-occlusion",
        dest="occlusion",
        action="store_false",
        help="ignore which animal is in front: paste in any order and ask every "
        "in-frame node of every animal to show",
    )
    p.add_argument(
        "--body-scale",
        type=float,
        default=1.0,
        help="scale the body model's radii (measured from the labels) by this",
    )
    p.add_argument(
        "--covered",
        type=float,
        default=0.02,
        help="an animal more than this fraction of whose body is behind another "
        "keeps its older uncovered appearance for the hidden part",
    )
    p.add_argument(
        "--min-visible",
        type=float,
        default=0.3,
        help="an animal showing less than this fraction of its in-frame nodes is "
        "not drawn in that camera (it is behind another)",
    )
    p.add_argument(
        "--no-clean-background",
        dest="clean_background",
        action="store_false",
        help="keep the median background as it is, even where the centroid model "
        "sees an animal in it",
    )
    p.add_argument(
        "--steps-next",
        type=int,
        default=150,
        help="steps for frames after the first (they start from the previous frame)",
    )
    p.add_argument(
        "--crf",
        type=int,
        default=8,
        help="x264 quality (0 = lossless, but that needs the High 4:4:4 profile, "
        "which browsers do not play)",
    )
    p.add_argument("--out", required=True)
    sb.add_optimization_args(p)
    # Pulled toward the start image: on DPBox topB (frame 3355, 2 mice, spline start,
    # body mask) --keep 0 grew dark halos and blotches around both mice, --keep 20
    # mostly did not, at the same round trip (1.23 vs 1.16 px median).
    p.set_defaults(func=cmd_generate, keep=20.0)

    p = sub.add_parser(
        "attach", help="give label groups 2D in more cameras, from predictions"
    )
    p.add_argument("--labels3d", required=True)
    p.add_argument("--session", help="default: the one in labels3d")
    p.add_argument("--calibration", help="default: the one in labels3d")
    p.add_argument(
        "--cameras", help="cameras to attach (default: every calibrated camera)"
    )
    p.add_argument("--centroid", required=True)
    p.add_argument("--instance", required=True)
    p.add_argument(
        "--predictions", required=True, help="folder for the per-camera predictions"
    )
    p.add_argument(
        "--min-score", type=float, default=0.5, help="instance score to accept"
    )
    p.add_argument(
        "--max-px",
        type=float,
        default=15,
        help="median node distance from the group's projection to accept",
    )
    p.add_argument(
        "--outlier-px",
        type=float,
        default=8,
        help="an added node's view reprojecting further than this is dropped (>= 3 kept)",
    )
    p.add_argument("--device", default="auto")
    p.add_argument("--out", required=True, help="the extended labels3d.json")
    p.set_defaults(func=cmd_attach)

    p = sub.add_parser(
        "bodies", help="draw the body model (what hides what) over real labeled frames"
    )
    p.add_argument("--labels3d", required=True)
    p.add_argument("--frames", required=True, help="labeled frames, e.g. 3355,926")
    p.add_argument("--cameras", help="default: every camera with 2D in labels3d")
    p.add_argument("--calibration", help="default: the one in labels3d")
    p.add_argument("--body-scale", type=float, default=1.0)
    p.add_argument("--out", required=True, help="a .png")
    p.set_defaults(func=cmd_bodies)

    p = sub.add_parser("predict", help="run the models forward on OUT/session's videos")
    p.add_argument("--out", required=True)
    p.add_argument("--centroid", required=True)
    p.add_argument("--instance", required=True)
    p.add_argument("--device", default="auto")
    p.set_defaults(func=cmd_predict)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
