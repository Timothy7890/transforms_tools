"""
Standalone FastAPI-based 3D camera pose viewer for one or more NeRF
transforms.json files. Renders an interactive Plotly scene in the browser with
per-file HTML checkboxes so each transforms.json group can be toggled.

Part of the transforms_tools package. After installing the optional viz extra
(`pip install -e ".[viz]"`), run it with the `transforms_viz` command, or as a
module: `python -m transforms_tools.visualize ...`.

Conventions:
    Right-handed world frame, OpenGL camera (looks -Z, +Y up).
    Axes coloring: X=red, Y=green, Z=blue (kept the same for every file).
    Frustums and camera markers are colored per file to tell them apart.

AABB overlay:
    If a transforms.json carries top-level "aabb" / "aabb_scale", the "显示 AABB"
    button draws the subject box (green solid) and the trainable cube
    (orange dashed, side = aabb_scale * longest_aabb_edge), in world coords.

Usage:
    transforms_viz [--host 0.0.0.0] [--port 7007] \
        /path/a/transforms.json /path/b/transforms.json ...

Dependencies (installed via the "viz" extra):
    pip install fastapi uvicorn plotly
"""

import argparse
import json
import os
import re
import numpy as np
from typing import List, Tuple

import plotly.graph_objects as go
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse


FRUSTUM_SCALE = 0.05
AXIS_SCALE = 0.04
WORLD_AXIS_SCALE = 0.20

# 3D 场景里相机中心点的大小（保持小，不挡视野）
MARKER_SIZE = 3
# 图例里彩色球的大小（只影响图例，跟场景点解耦）
LEGEND_MARKER_SIZE = 22

PALETTE = [
    "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
    "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52",
]

# Camera id parsed from file names like "auto_cam01_angle0.png" -> "cam01".
CAM_RE = re.compile(r"cam0*(\d+)", re.IGNORECASE)

# Length of the drawn plane-normal arrow, as a fraction of the scene diagonal.
NORMAL_SCALE = 0.15


def parse_transforms(json_path: str) -> tuple[dict, list[np.ndarray], list[str], dict]:
    with open(json_path, "r") as f:
        data = json.load(f)

    required = ["fl_x", "fl_y", "cx", "cy", "w", "h"]
    missing = [k for k in required if k not in data]
    if missing:
        raise KeyError(f"transforms.json missing keys: {missing}")

    intr = {k: float(data[k]) for k in required}

    # Optional NGP scene bounds. `aabb` is the subject box (world/mm coords);
    # `aabb_scale` (power of 2) defines the trainable cube around it.
    aabb = None
    if isinstance(data.get("aabb"), list) and len(data["aabb"]) == 2:
        try:
            mn = [float(v) for v in data["aabb"][0]]
            mx = [float(v) for v in data["aabb"][1]]
            if len(mn) == 3 and len(mx) == 3:
                aabb = [mn, mx]
        except (TypeError, ValueError):
            aabb = None
    aabb_scale = data.get("aabb_scale")
    try:
        aabb_scale = float(aabb_scale) if aabb_scale is not None else None
    except (TypeError, ValueError):
        aabb_scale = None
    meta = {"aabb": aabb, "aabb_scale": aabb_scale}

    poses: list[np.ndarray] = []
    file_paths: list[str] = []
    for frame in data.get("frames", []):
        m = np.array(frame["transform_matrix"], dtype=np.float64)
        if m.shape != (4, 4):
            raise ValueError(f"transform_matrix must be 4x4, got {m.shape}")
        poses.append(m)
        file_paths.append(str(frame.get("file_path", "")))

    if not poses:
        raise ValueError(f"No frames found in {json_path}")

    return intr, poses, file_paths, meta


def group_centers_by_camera(
    poses: list[np.ndarray], file_paths: list[str]
) -> dict[str, list[list[float]]]:
    """Group camera centers by physical camera id parsed from file_path.

    Falls back to a single "all" group when names carry no cam id.
    """
    groups: dict[str, list[list[float]]] = {}
    for c2w, fp in zip(poses, file_paths):
        m = CAM_RE.search(os.path.basename(fp))
        key = f"cam{int(m.group(1)):02d}" if m else "all"
        t = c2w[:3, 3]
        groups.setdefault(key, []).append([float(t[0]), float(t[1]), float(t[2])])
    return groups


def _unique_labels(paths: list[str]) -> list[str]:
    """Use parent dir basename as label; de-duplicate with ' (n)' suffix."""
    base_names = [os.path.basename(os.path.dirname(p)) or p for p in paths]
    seen: dict[str, int] = {}
    out: list[str] = []
    for name in base_names:
        seen[name] = seen.get(name, 0) + 1
        if seen[name] == 1:
            out.append(name)
        else:
            out.append(f"{name} ({seen[name]})")
    return out


def _scene_diag(all_centers: np.ndarray) -> float:
    if all_centers.shape[0] < 2:
        return 1.0
    diag = float(np.linalg.norm(all_centers.max(axis=0) - all_centers.min(axis=0)))
    return diag if diag > 1e-6 else 1.0


def _axis_traces(
    origin: np.ndarray,
    rot: np.ndarray,
    size: float,
    legendgroup: str,
) -> List[go.Scatter3d]:
    traces: List[go.Scatter3d] = []
    for i, color in [(0, "red"), (1, "green"), (2, "blue")]:
        end = origin + rot[:, i] * size
        pts = np.stack([origin, end], axis=0)
        traces.append(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode="lines",
            line=dict(color=color, width=4),
            showlegend=False,
            legendgroup=legendgroup,
            hoverinfo="skip",
        ))
    return traces


def _frustum_trace(
    c2w: np.ndarray,
    intr: dict,
    depth: float,
    color: str,
    legendgroup: str,
) -> go.Scatter3d:
    fl_x, fl_y = intr["fl_x"], intr["fl_y"]
    cx, cy = intr["cx"], intr["cy"]
    w, h = intr["w"], intr["h"]

    def corner(u: float, v: float) -> list[float]:
        x = (u - cx) / fl_x * depth
        y = -(v - cy) / fl_y * depth
        z = -depth
        return [x, y, z, 1.0]

    pts_local = np.array([
        [0.0, 0.0, 0.0, 1.0],
        corner(0.0, 0.0),
        corner(w,   0.0),
        corner(w,   h),
        corner(0.0, h),
    ], dtype=np.float64)
    pts = (c2w @ pts_local.T).T[:, :3]

    edges = [(0, 1), (0, 2), (0, 3), (0, 4),
             (1, 2), (2, 3), (3, 4), (4, 1)]
    xs, ys, zs = [], [], []
    for a, b in edges:
        xs += [pts[a, 0], pts[b, 0], None]
        ys += [pts[a, 1], pts[b, 1], None]
        zs += [pts[a, 2], pts[b, 2], None]

    return go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode="lines",
        line=dict(color=color, width=2),
        showlegend=False,
        legendgroup=legendgroup,
        hoverinfo="skip",
    )


def _batched_axis_traces(
    poses: list[np.ndarray],
    size: float,
    legendgroup: str,
) -> List[go.Scatter3d]:
    """所有位姿的局部坐标轴，按颜色合并成 3 个 trace（X=红/Y=绿/Z=蓝）。

    用 None 分隔各段，避免每个位姿单独建 trace，大幅减少 trace 数量。
    """
    seg = {0: ([], [], []), 1: ([], [], []), 2: ([], [], [])}
    for c2w in poses:
        o = c2w[:3, 3]
        rot = c2w[:3, :3]
        for i in range(3):
            e = o + rot[:, i] * size
            xs, ys, zs = seg[i]
            xs += [o[0], e[0], None]
            ys += [o[1], e[1], None]
            zs += [o[2], e[2], None]
    colors = {0: "red", 1: "green", 2: "blue"}
    traces: List[go.Scatter3d] = []
    for i in range(3):
        xs, ys, zs = seg[i]
        traces.append(go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode="lines",
            line=dict(color=colors[i], width=4),
            showlegend=False,
            legendgroup=legendgroup,
            hoverinfo="skip",
        ))
    return traces


def _batched_frustum_trace(
    poses: list[np.ndarray],
    intr: dict,
    depth: float,
    color: str,
    legendgroup: str,
) -> go.Scatter3d:
    """把同一文件里所有相机的视锥合并成一个 trace。"""
    fl_x, fl_y = intr["fl_x"], intr["fl_y"]
    cx, cy = intr["cx"], intr["cy"]
    w, h = intr["w"], intr["h"]

    def corner(u: float, v: float) -> list[float]:
        x = (u - cx) / fl_x * depth
        y = -(v - cy) / fl_y * depth
        return [x, y, -depth, 1.0]

    pts_local = np.array([
        [0.0, 0.0, 0.0, 1.0],
        corner(0.0, 0.0),
        corner(w,   0.0),
        corner(w,   h),
        corner(0.0, h),
    ], dtype=np.float64)
    edges = [(0, 1), (0, 2), (0, 3), (0, 4),
             (1, 2), (2, 3), (3, 4), (4, 1)]

    xs, ys, zs = [], [], []
    for c2w in poses:
        pts = (c2w @ pts_local.T).T[:, :3]
        for a, b in edges:
            xs += [pts[a, 0], pts[b, 0], None]
            ys += [pts[a, 1], pts[b, 1], None]
            zs += [pts[a, 2], pts[b, 2], None]

    return go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode="lines",
        line=dict(color=color, width=2),
        showlegend=False,
        legendgroup=legendgroup,
        hoverinfo="skip",
    )


def build_figure(
    paths: list[str],
) -> Tuple[go.Figure, dict[str, list[int]], list[str], list[str], dict, dict, float, dict]:
    """Return (figure, label_to_trace_indices, used_labels, warnings,
    cam_data, label_colors, scene_diag, aabb_data)."""
    if not paths:
        raise ValueError("No transforms.json paths provided.")

    labels = _unique_labels(paths)

    parsed: list[Tuple[str, dict, list[np.ndarray], list[str], dict]] = []
    warnings: list[str] = []
    for label, p in zip(labels, paths):
        try:
            intr, poses, file_paths, meta = parse_transforms(p)
            parsed.append((label, intr, poses, file_paths, meta))
        except (FileNotFoundError, KeyError, ValueError) as exc:
            warnings.append(f"[SKIP] {label} ({p}): {exc}")

    if not parsed:
        raise ValueError("None of the transforms.json files could be loaded. "
                         + " ".join(warnings))

    all_centers = np.stack(
        [pose[:3, 3] for _, _, poses, _, _ in parsed for pose in poses],
        axis=0,
    )
    diag = _scene_diag(all_centers)
    axis_size = AXIS_SCALE * diag
    frustum_depth = FRUSTUM_SCALE * diag

    fig = go.Figure()

    for tr in _axis_traces(np.zeros(3), np.eye(3), WORLD_AXIS_SCALE * diag, legendgroup="__world__"):
        fig.add_trace(tr)

    label_to_indices: dict[str, list[int]] = {}
    used_labels: list[str] = []
    cam_data: dict[str, dict[str, list[list[float]]]] = {}
    label_colors: dict[str, str] = {}
    aabb_data: dict[str, dict] = {}

    for gi, (label, intr, poses, file_paths, meta) in enumerate(parsed):
        used_labels.append(label)
        color = PALETTE[gi % len(PALETTE)]
        label_colors[label] = color
        cam_data[label] = group_centers_by_camera(poses, file_paths)
        aabb_data[label] = meta
        indices: list[int] = []

        # 合并 trace：每文件 3 根轴 + 1 个视锥，而不是每位姿 4 个 trace。
        for tr in _batched_axis_traces(poses, axis_size, legendgroup=label):
            indices.append(len(fig.data))
            fig.add_trace(tr)

        fr = _batched_frustum_trace(poses, intr, frustum_depth, color=color, legendgroup=label)
        indices.append(len(fig.data))
        fig.add_trace(fr)

        cc = np.stack([c2w[:3, 3] for c2w in poses], axis=0)
        marker = go.Scatter3d(
            x=cc[:, 0], y=cc[:, 1], z=cc[:, 2],
            mode="markers",
            marker=dict(size=MARKER_SIZE, color=color),
            legendgroup=label,
            showlegend=False,
            hovertext=[f"{label} :: frame {i}" for i in range(len(cc))],
            hoverinfo="text",
        )
        indices.append(len(fig.data))
        fig.add_trace(marker)

        # 图例专用的大彩色球（不画进场景，只为图例放大显示）
        legend_marker = go.Scatter3d(
            x=[None], y=[None], z=[None],
            mode="markers",
            marker=dict(size=LEGEND_MARKER_SIZE, color=color),
            name=label,
            legendgroup=label,
            showlegend=True,
            hoverinfo="skip",
        )
        indices.append(len(fig.data))
        fig.add_trace(legend_marker)

        label_to_indices[label] = indices

    n_cams_total = sum(len(poses) for _, _, poses, _, _ in parsed)
    fig.update_layout(
        title=f"Camera Poses: {len(parsed)} file(s), {n_cams_total} cameras  |  diag={diag:.3f}",
        scene=dict(
            xaxis_title="X (red)",
            yaxis_title="Y (green)",
            zaxis_title="Z (blue)",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        template="plotly_white",
        legend=dict(itemclick="toggle", itemdoubleclick="toggleothers"),
    )
    return fig, label_to_indices, used_labels, warnings, cam_data, label_colors, diag, aabb_data


app = FastAPI(title="Camera Pose Viewer")

# Populated from CLI args in main().
TRANSFORMS_LIST: list[str] = []


def _render_page(fig: go.Figure,
                 label_to_indices: dict[str, list[int]],
                 labels: list[str],
                 warnings: list[str],
                 paths: list[str],
                 cam_data: dict,
                 label_colors: dict,
                 diag: float,
                 aabb_data: dict) -> str:
    fig_dict = fig.to_dict()
    fig_json = json.dumps(fig_dict, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else o)
    map_json = json.dumps(label_to_indices)
    cam_data_json = json.dumps(cam_data)
    label_colors_json = json.dumps(label_colors)
    aabb_data_json = json.dumps(aabb_data)
    normal_len = NORMAL_SCALE * diag

    checkboxes_html = "\n".join(
        f'  <label class="cb"><input type="checkbox" data-label="{lbl}" checked> {lbl}</label>'
        for lbl in labels
    )
    src_html = "".join(f"<span class='src-line'>{lbl}: {p}</span>"
                       for lbl, p in zip(labels, paths))
    warn_html = ""
    if warnings:
        warn_html = "<div class='warn'>" + "<br>".join(warnings) + "</div>"

    return f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8"/>
<title>Camera Poses</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  html, body {{ margin: 0; padding: 0; height: 100%; font-family: ui-sans-serif, system-ui, sans-serif; }}
  #topbar {{ padding: 10px 14px; border-bottom: 1px solid #eee; display: flex; flex-wrap: wrap; gap: 14px; align-items: center; }}
  #topbar button {{ padding: 4px 10px; cursor: pointer; }}
  #checkboxes {{ display: flex; flex-wrap: wrap; gap: 10px 14px; padding: 8px 14px; border-bottom: 1px solid #eee; }}
  .cb {{ user-select: none; cursor: pointer; }}
  .src-details > summary {{ padding: 6px 14px; cursor: pointer; user-select: none; color: #555; font-size: 12px; list-style: revert; }}
  .src-details > summary:hover {{ color: #222; }}
  .src {{ padding: 6px 14px; color: #777; font-size: 12px; word-break: break-all; border-bottom: 1px solid #f4f4f4; }}
  .src-line {{ display: block; margin: 2px 0; }}
  .warn {{ padding: 6px 14px; color: #b15a00; background: #fff5e6; font-size: 12px; }}
  .fitinfo {{ display: none; padding: 6px 14px; color: #2b5; background: #f3fbf5; font-size: 12px; font-family: ui-monospace, monospace; word-break: break-all; border-bottom: 1px solid #eef; }}
  .measinfo {{ display: none; padding: 6px 14px; color: #1864ab; background: #eef6ff; font-size: 12px; font-family: ui-monospace, monospace; word-break: break-all; border-bottom: 1px solid #eef; }}
  .circinfo {{ display: none; padding: 6px 14px; color: #ae3ec9; background: #fbf0ff; font-size: 12px; font-family: ui-monospace, monospace; word-break: break-all; border-bottom: 1px solid #eef; }}
  .aabbinfo {{ display: none; padding: 6px 14px; color: #e8590c; background: #fff4e6; font-size: 12px; font-family: ui-monospace, monospace; word-break: break-all; border-bottom: 1px solid #eef; }}
  button.on {{ background: #2b8a3e; color: #fff; border-color: #2b8a3e; }}
  #measBtn.on {{ background: #1864ab; border-color: #1864ab; }}
  #circBtn.on {{ background: #ae3ec9; border-color: #ae3ec9; }}
  #aabbBtn.on {{ background: #e8590c; border-color: #e8590c; }}
  #plot {{ width: 100%; height: calc(100vh - 180px); min-height: 480px; }}
</style>
</head><body>

<div id="topbar">
  <strong>Camera Pose Viewer</strong>
  <button onclick="setAll(true)">全选</button>
  <button onclick="setAll(false)">全不选</button>
  <button id="fitBtn" onclick="toggleFit()">开始拟合平面</button>
  <button id="circBtn" onclick="toggleCircle()">拟合圆平面</button>
  <button id="measBtn" onclick="toggleMeasure()">测量相邻平面距离</button>
  <button id="aabbBtn" onclick="toggleAabb()">显示 AABB</button>
  <span style="color:#888;font-size:12px;">点击图例项也能单独切换 / 双击只看这一个</span>
</div>

<div id="checkboxes">
{checkboxes_html}
</div>

<details class="src-details">
  <summary>实际路径 ({len(paths)})</summary>
  <div class="src">{src_html}</div>
</details>
<div id="fitinfo" class="fitinfo"></div>
<div id="circinfo" class="circinfo"></div>
<div id="measinfo" class="measinfo"></div>
<div id="aabbinfo" class="aabbinfo"></div>
{warn_html}

<div id="plot"></div>

<script>
const LABEL_TO_INDICES = {map_json};
const FIG = {fig_json};
const CAM_DATA = {cam_data_json};
const LABEL_COLORS = {label_colors_json};
const AABB_DATA = {aabb_data_json};
const NORMAL_LEN = {normal_len};

Plotly.newPlot('plot', FIG.data, FIG.layout, {{responsive: true}});

// Dynamically-added overlay traces (planes / distance) are tagged with a
// `meta` object so they can always be located by scanning gd.data, which is
// robust to index shifts when overlays are added/removed independently.
function dynIndicesForLabel(label) {{
  const gd = document.getElementById('plot');
  const out = [];
  gd.data.forEach((tr, i) => {{ if (tr.meta && tr.meta.dyn && tr.meta.label === label) out.push(i); }});
  return out;
}}

function indicesByKind(kind) {{
  const gd = document.getElementById('plot');
  const out = [];
  gd.data.forEach((tr, i) => {{ if (tr.meta && tr.meta.kind === kind) out.push(i); }});
  return out;
}}

function labelVisible(label) {{
  const cb = document.querySelector(`#checkboxes input[data-label="${{label}}"]`);
  return cb ? cb.checked : true;
}}

function toggle(label, visible) {{
  const idxs = (LABEL_TO_INDICES[label] || []).concat(dynIndicesForLabel(label));
  if (idxs.length === 0) return;
  Plotly.restyle('plot', {{visible: visible ? true : 'legendonly'}}, idxs);
}}

// ---- front-end plane fitting -----------------------------------------
let planesOn = false;

// Cyclic Jacobi eigen-decomposition for a symmetric 3x3 matrix.
// Returns {{values:[3], vectors:[[3],[3],[3]]}} where vectors[i][k] is the
// i-th component of the k-th eigenvector (eigenvectors are columns).
function eig3sym(A) {{
  const a = A.map(r => r.slice());
  const v = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  const offs = [[0, 1], [0, 2], [1, 2]];
  for (let sweep = 0; sweep < 60; sweep++) {{
    let p = 0, q = 1, mx = Math.abs(a[0][1]);
    for (const [i, j] of offs) {{
      if (Math.abs(a[i][j]) > mx) {{ mx = Math.abs(a[i][j]); p = i; q = j; }}
    }}
    if (mx < 1e-14) break;
    const phi = 0.5 * Math.atan2(2 * a[p][q], a[q][q] - a[p][p]);
    const c = Math.cos(phi), s = Math.sin(phi);
    for (let k = 0; k < 3; k++) {{
      const akp = a[k][p], akq = a[k][q];
      a[k][p] = c * akp - s * akq;
      a[k][q] = s * akp + c * akq;
    }}
    for (let k = 0; k < 3; k++) {{
      const apk = a[p][k], aqk = a[q][k];
      a[p][k] = c * apk - s * aqk;
      a[q][k] = s * apk + c * aqk;
    }}
    for (let k = 0; k < 3; k++) {{
      const vkp = v[k][p], vkq = v[k][q];
      v[k][p] = c * vkp - s * vkq;
      v[k][q] = s * vkp + c * vkq;
    }}
  }}
  return {{ values: [a[0][0], a[1][1], a[2][2]], vectors: v }};
}}

function fitOnePlane(pts) {{
  const n = pts.length;
  const c = [0, 0, 0];
  for (const p of pts) {{ c[0] += p[0]; c[1] += p[1]; c[2] += p[2]; }}
  c[0] /= n; c[1] /= n; c[2] /= n;

  const cov = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (const p of pts) {{
    const d = [p[0] - c[0], p[1] - c[1], p[2] - c[2]];
    for (let i = 0; i < 3; i++)
      for (let j = 0; j < 3; j++) cov[i][j] += d[i] * d[j];
  }}
  for (let i = 0; i < 3; i++)
    for (let j = 0; j < 3; j++) cov[i][j] /= n;

  const {{ values, vectors }} = eig3sym(cov);
  const order = [0, 1, 2].sort((x, y) => values[x] - values[y]);
  const col = k => [vectors[0][k], vectors[1][k], vectors[2][k]];
  const normal = col(order[0]);   // smallest spread -> plane normal
  const u = col(order[2]);        // largest in-plane direction
  const v = col(order[1]);
  const rms = Math.sqrt(Math.max(values[order[0]], 0));

  let umin = Infinity, umax = -Infinity, vmin = Infinity, vmax = -Infinity;
  for (const p of pts) {{
    const d = [p[0] - c[0], p[1] - c[1], p[2] - c[2]];
    const su = d[0] * u[0] + d[1] * u[1] + d[2] * u[2];
    const sv = d[0] * v[0] + d[1] * v[1] + d[2] * v[2];
    umin = Math.min(umin, su); umax = Math.max(umax, su);
    vmin = Math.min(vmin, sv); vmax = Math.max(vmax, sv);
  }}
  const pad = 1.12;
  umin *= pad; umax *= pad; vmin *= pad; vmax *= pad;
  const corner = (su, sv) => [
    c[0] + su * u[0] + sv * v[0],
    c[1] + su * u[1] + sv * v[1],
    c[2] + su * u[2] + sv * v[2],
  ];
  const corners = [corner(umin, vmin), corner(umax, vmin),
                   corner(umax, vmax), corner(umin, vmax)];
  return {{ centroid: c, normal, corners, rms, n, u, v }};
}}

// Algebraic (Kåsa) least-squares circle fit on in-plane 2D coords.
// P: array of [u, v]. Returns {{cu, cv, r, rms}} (center in u/v coords).
function fitCircle2D(P) {{
  const n = P.length;
  let mu = 0, mv = 0;
  for (const p of P) {{ mu += p[0]; mv += p[1]; }}
  mu /= n; mv /= n;
  let Suu = 0, Svv = 0, Suv = 0, Suuu = 0, Svvv = 0, Suvv = 0, Svuu = 0;
  for (const p of P) {{
    const u = p[0] - mu, v = p[1] - mv;
    Suu += u*u; Svv += v*v; Suv += u*v;
    Suuu += u*u*u; Svvv += v*v*v; Suvv += u*v*v; Svuu += v*u*u;
  }}
  const C1 = 0.5*(Suuu + Suvv), C2 = 0.5*(Svvv + Svuu);
  const det = Suu*Svv - Suv*Suv;
  let uc = 0, vc = 0;
  if (Math.abs(det) > 1e-12) {{
    uc = (C1*Svv - C2*Suv) / det;
    vc = (Suu*C2 - Suv*C1) / det;
  }}
  const r = Math.sqrt(Math.max(uc*uc + vc*vc + (Suu + Svv)/n, 0));
  const cu = uc + mu, cv = vc + mv;
  let se = 0;
  for (const p of P) {{
    const d = Math.hypot(p[0] - cu, p[1] - cv) - r;
    se += d*d;
  }}
  return {{ cu, cv, r, rms: Math.sqrt(se / n) }};
}}

// Fit every camera (>=3 poses) in a file. Returns a list of
// {{cam, centroid, normal, corners, rms, n}} sorted by id.
function fitLabelPlanes(label) {{
  const cams = CAM_DATA[label] || {{}};
  const out = [];
  for (const cam of Object.keys(cams).sort()) {{
    const pts = cams[cam];
    if (pts.length < 3) continue;
    const f = fitOnePlane(pts);
    f.cam = cam;
    out.push(f);
  }}
  return out;
}}

function toggleFit() {{
  const gd = document.getElementById('plot');
  const btn = document.getElementById('fitBtn');
  const infoEl = document.getElementById('fitinfo');

  if (!planesOn) {{
    const newTraces = [];
    const info = [];
    for (const label in CAM_DATA) {{
      const color = LABEL_COLORS[label] || '#888888';
      const vis = labelVisible(label) ? true : 'legendonly';
      for (const f of fitLabelPlanes(label)) {{
        const nstr = f.normal.map(x => x.toFixed(4)).join(', ');
        newTraces.push({{
          type: 'mesh3d',
          x: f.corners.map(p => p[0]), y: f.corners.map(p => p[1]), z: f.corners.map(p => p[2]),
          i: [0, 0], j: [1, 2], k: [2, 3],
          color: color, opacity: 0.35, flatshading: true,
          name: label + ' ' + f.cam, legendgroup: label,
          showlegend: false, hoverinfo: 'skip', visible: vis,
          meta: {{ dyn: true, kind: 'fit', label: label }},
        }});
        const e = [f.centroid[0] + f.normal[0] * NORMAL_LEN,
                   f.centroid[1] + f.normal[1] * NORMAL_LEN,
                   f.centroid[2] + f.normal[2] * NORMAL_LEN];
        newTraces.push({{
          type: 'scatter3d', mode: 'lines',
          x: [f.centroid[0], e[0]], y: [f.centroid[1], e[1]], z: [f.centroid[2], e[2]],
          line: {{ color: color, width: 6 }},
          name: label + ' ' + f.cam + ' normal', legendgroup: label,
          showlegend: false, hoverinfo: 'text', visible: vis,
          hovertext: `${{label}} ${{f.cam}}<br>normal=(${{nstr}})<br>rms=${{f.rms.toExponential(2)}}  n=${{f.n}}`,
          meta: {{ dyn: true, kind: 'fit', label: label }},
        }});
        info.push(`${{label}} ${{f.cam}}: n=${{f.n}}  normal=(${{nstr}})  rms=${{f.rms.toExponential(2)}}`);
      }}
    }}
    if (newTraces.length === 0) {{
      infoEl.textContent = '没有可拟合的相机（每个相机至少需要 3 个位姿）。';
      infoEl.style.display = 'block';
      return;
    }}
    Plotly.addTraces('plot', newTraces);
    infoEl.innerHTML = info.join('<br>');
    infoEl.style.display = 'block';
    planesOn = true;
    btn.textContent = '隐藏拟合平面';
    btn.classList.add('on');
  }} else {{
    const idx = indicesByKind('fit');
    if (idx.length) Plotly.deleteTraces('plot', idx);
    infoEl.style.display = 'none';
    infoEl.innerHTML = '';
    planesOn = false;
    btn.textContent = '开始拟合平面';
    btn.classList.remove('on');
  }}
}}

// ---- front-end circle fitting (orbit circles) ------------------------
let circlesOn = false;

function toggleCircle() {{
  const gd = document.getElementById('plot');
  const btn = document.getElementById('circBtn');
  const infoEl = document.getElementById('circinfo');

  if (!circlesOn) {{
    const newTraces = [];
    const info = [];
    const SEG = 96;
    for (const label in CAM_DATA) {{
      const color = LABEL_COLORS[label] || '#888888';
      const vis = labelVisible(label) ? true : 'legendonly';
      const cams = CAM_DATA[label];
      for (const cam of Object.keys(cams).sort()) {{
        const pts = cams[cam];
        if (pts.length < 3) continue;
        const f = fitOnePlane(pts);
        // Project points into the plane's (u, v) coordinate frame.
        const P = pts.map(p => {{
          const d = [p[0] - f.centroid[0], p[1] - f.centroid[1], p[2] - f.centroid[2]];
          return [d[0]*f.u[0] + d[1]*f.u[1] + d[2]*f.u[2],
                  d[0]*f.v[0] + d[1]*f.v[1] + d[2]*f.v[2]];
        }});
        const ci = fitCircle2D(P);
        const diameter = 2 * ci.r;
        // Circle center back in 3D.
        const center3 = [
          f.centroid[0] + ci.cu*f.u[0] + ci.cv*f.v[0],
          f.centroid[1] + ci.cu*f.u[1] + ci.cv*f.v[1],
          f.centroid[2] + ci.cu*f.u[2] + ci.cv*f.v[2],
        ];
        // Sample the circle polyline.
        const cx = [], cy = [], cz = [];
        for (let s = 0; s <= SEG; s++) {{
          const t = 2*Math.PI*s/SEG;
          const a = ci.r*Math.cos(t), b = ci.r*Math.sin(t);
          cx.push(center3[0] + a*f.u[0] + b*f.v[0]);
          cy.push(center3[1] + a*f.u[1] + b*f.v[1]);
          cz.push(center3[2] + a*f.u[2] + b*f.v[2]);
        }}
        newTraces.push({{
          type: 'scatter3d', mode: 'lines',
          x: cx, y: cy, z: cz,
          line: {{ color: color, width: 4 }},
          name: label + ' ' + cam + ' circle', legendgroup: label,
          showlegend: false, hoverinfo: 'text', visible: vis,
          hovertext: `${{label}} ${{cam}}<br>diameter=${{diameter.toFixed(4)}}<br>radius=${{ci.r.toFixed(4)}}  fitRMS=${{ci.rms.toExponential(2)}}`,
          meta: {{ dyn: true, kind: 'circle', label: label }},
        }});
        // Filled disk (triangle fan from center) so the plane interior is shaded.
        const dvx = [center3[0]], dvy = [center3[1]], dvz = [center3[2]];
        for (let s = 0; s <= SEG; s++) {{ dvx.push(cx[s]); dvy.push(cy[s]); dvz.push(cz[s]); }}
        const dfi = [], dfj = [], dfk = [];
        for (let s = 0; s < SEG; s++) {{ dfi.push(0); dfj.push(1 + s); dfk.push(2 + s); }}
        newTraces.push({{
          type: 'mesh3d',
          x: dvx, y: dvy, z: dvz,
          i: dfi, j: dfj, k: dfk,
          color: color, opacity: 0.25, flatshading: true,
          name: label + ' ' + cam + ' disk', legendgroup: label,
          showlegend: false, hoverinfo: 'skip', visible: vis,
          meta: {{ dyn: true, kind: 'circle', label: label }},
        }});
        // Center marker + diameter label.
        newTraces.push({{
          type: 'scatter3d', mode: 'markers+text',
          x: [center3[0]], y: [center3[1]], z: [center3[2]],
          marker: {{ size: 4, color: color, symbol: 'x' }},
          text: ['⌀' + diameter.toFixed(3)],
          textposition: 'top center',
          textfont: {{ size: 12, color: color }},
          name: label + ' ' + cam + ' center', legendgroup: label,
          showlegend: false, hoverinfo: 'skip', visible: vis,
          meta: {{ dyn: true, kind: 'circle', label: label }},
        }});
        info.push(`${{label}} ${{cam}}: ⌀=${{diameter.toFixed(4)}}  r=${{ci.r.toFixed(4)}}  fitRMS=${{ci.rms.toExponential(2)}}`);
      }}
    }}
    if (newTraces.length === 0) {{
      infoEl.textContent = '没有可拟合的相机（每个相机至少需要 3 个位姿）。';
      infoEl.style.display = 'block';
      return;
    }}
    Plotly.addTraces('plot', newTraces);
    infoEl.innerHTML = '圆轨迹拟合（平面内最小二乘，⌀=直径）:<br>' + info.join('<br>');
    infoEl.style.display = 'block';
    circlesOn = true;
    btn.textContent = '隐藏圆平面';
    btn.classList.add('on');
  }} else {{
    const idx = indicesByKind('circle');
    if (idx.length) Plotly.deleteTraces('plot', idx);
    infoEl.style.display = 'none';
    infoEl.innerHTML = '';
    circlesOn = false;
    btn.textContent = '拟合圆平面';
    btn.classList.remove('on');
  }}
}}

// ---- adjacent-plane distance measurement -----------------------------
let measOn = false;

function toggleMeasure() {{
  const gd = document.getElementById('plot');
  const btn = document.getElementById('measBtn');
  const infoEl = document.getElementById('measinfo');

  if (!measOn) {{
    const newTraces = [];
    const info = [];
    for (const label in CAM_DATA) {{
      const color = LABEL_COLORS[label] || '#888888';
      const vis = labelVisible(label) ? true : 'legendonly';
      const planes = fitLabelPlanes(label);
      if (planes.length < 2) continue;

      // Mean normal (sign-aligned to the first plane) defines the stacking axis.
      const ref = planes[0].normal;
      let mn = [0, 0, 0];
      for (const p of planes) {{
        const dot = p.normal[0]*ref[0] + p.normal[1]*ref[1] + p.normal[2]*ref[2];
        const sgn = dot < 0 ? -1 : 1;
        mn[0] += sgn*p.normal[0]; mn[1] += sgn*p.normal[1]; mn[2] += sgn*p.normal[2];
      }}
      const mlen = Math.hypot(mn[0], mn[1], mn[2]) || 1;
      mn = [mn[0]/mlen, mn[1]/mlen, mn[2]/mlen];

      for (const p of planes) {{
        p.proj = p.centroid[0]*mn[0] + p.centroid[1]*mn[1] + p.centroid[2]*mn[2];
      }}
      planes.sort((a, b) => a.proj - b.proj);

      // Polyline through centroids in stacking order.
      newTraces.push({{
        type: 'scatter3d', mode: 'lines+markers',
        x: planes.map(p => p.centroid[0]), y: planes.map(p => p.centroid[1]), z: planes.map(p => p.centroid[2]),
        line: {{ color: color, width: 4, dash: 'dot' }},
        marker: {{ size: 4, color: color }},
        name: label + ' spacing', legendgroup: label,
        showlegend: false, hoverinfo: 'text', visible: vis,
        hovertext: planes.map(p => `${{label}} ${{p.cam}}`),
        meta: {{ dyn: true, kind: 'meas', label: label }},
      }});

      // Distance labels at the midpoint of each adjacent pair.
      const tx = [], ty = [], tz = [], tt = [];
      for (let i = 0; i + 1 < planes.length; i++) {{
        const a = planes[i], b = planes[i + 1];
        const d = Math.abs(b.proj - a.proj);
        tx.push((a.centroid[0] + b.centroid[0]) / 2);
        ty.push((a.centroid[1] + b.centroid[1]) / 2);
        tz.push((a.centroid[2] + b.centroid[2]) / 2);
        tt.push(d.toFixed(4));
        info.push(`${{label}} ${{a.cam}}→${{b.cam}}: ${{d.toFixed(4)}}`);
      }}
      newTraces.push({{
        type: 'scatter3d', mode: 'text',
        x: tx, y: ty, z: tz, text: tt,
        textfont: {{ size: 13, color: color }},
        name: label + ' dist', legendgroup: label,
        showlegend: false, hoverinfo: 'skip', visible: vis,
        meta: {{ dyn: true, kind: 'meas', label: label }},
      }});
    }}
    if (newTraces.length === 0) {{
      infoEl.textContent = '需要每个文件至少 2 个相机平面（每个平面 >=3 个位姿）。';
      infoEl.style.display = 'block';
      return;
    }}
    Plotly.addTraces('plot', newTraces);
    infoEl.innerHTML = '相邻平面间距（沿平均法向投影）:<br>' + info.join('<br>');
    infoEl.style.display = 'block';
    measOn = true;
    btn.textContent = '隐藏距离测量';
    btn.classList.add('on');
  }} else {{
    const idx = indicesByKind('meas');
    if (idx.length) Plotly.deleteTraces('plot', idx);
    infoEl.style.display = 'none';
    infoEl.innerHTML = '';
    measOn = false;
    btn.textContent = '测量相邻平面距离';
    btn.classList.remove('on');
  }}
}}

// ---- AABB / aabb_scale boxes -----------------------------------------
let aabbOn = false;

// 12-edge wireframe box from min/max corners as one scatter3d line trace.
function boxTrace(mn, mx, color, dash, width, label, name) {{
  const c = [
    [mn[0], mn[1], mn[2]], [mx[0], mn[1], mn[2]], [mx[0], mx[1], mn[2]], [mn[0], mx[1], mn[2]],
    [mn[0], mn[1], mx[2]], [mx[0], mn[1], mx[2]], [mx[0], mx[1], mx[2]], [mn[0], mx[1], mx[2]],
  ];
  const edges = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]];
  const xs = [], ys = [], zs = [];
  for (const [a, b] of edges) {{
    xs.push(c[a][0], c[b][0], null);
    ys.push(c[a][1], c[b][1], null);
    zs.push(c[a][2], c[b][2], null);
  }}
  return {{
    type: 'scatter3d', mode: 'lines', x: xs, y: ys, z: zs,
    line: {{ color: color, width: width, dash: dash }},
    name: name, legendgroup: label, showlegend: false,
    hoverinfo: 'text', hovertext: name,
    meta: {{ dyn: true, kind: 'aabb', label: label }},
  }};
}}

function toggleAabb() {{
  const btn = document.getElementById('aabbBtn');
  const infoEl = document.getElementById('aabbinfo');

  if (!aabbOn) {{
    const newTraces = [];
    const info = [];
    for (const label in AABB_DATA) {{
      const a = AABB_DATA[label];
      if (!a || !a.aabb) continue;
      const vis = labelVisible(label) ? true : 'legendonly';
      const mn = a.aabb[0], mx = a.aabb[1];

      // 绿色实线：植物区 aabb
      const t1 = boxTrace(mn, mx, '#2b8a3e', 'solid', 5, label, label + ' aabb(植物区)');
      t1.visible = vis; newTraces.push(t1);

      // 橙色虚线：可训练区 = 以 aabb 中心为心、边长 aabb_scale*最长边 的立方体
      const sc = a.aabb_scale || 1;
      const center = [(mn[0]+mx[0])/2, (mn[1]+mx[1])/2, (mn[2]+mx[2])/2];
      const L = Math.max(mx[0]-mn[0], mx[1]-mn[1], mx[2]-mn[2]);
      const half = 0.5 * sc * L;
      const cmn = [center[0]-half, center[1]-half, center[2]-half];
      const cmx = [center[0]+half, center[1]+half, center[2]+half];
      const t2 = boxTrace(cmn, cmx, '#e8590c', 'dash', 3, label,
                          label + ' aabb_scale=' + sc + '(可训练区)');
      t2.visible = vis; newTraces.push(t2);

      const f3 = v => v.map(x => x.toFixed(1)).join(',');
      info.push(`${{label}}: aabb=[${{f3(mn)}}]~[${{f3(mx)}}]  aabb_scale=${{sc}}  ` +
                `可训练立方体边长=${{(sc*L).toFixed(1)}} (最长边 L=${{L.toFixed(1)}})`);
    }}
    if (newTraces.length === 0) {{
      infoEl.textContent = '当前 transforms.json 没有 "aabb" 字段，无法可视化（请先在 transforms.json 顶层加 aabb / aabb_scale）。';
      infoEl.style.display = 'block';
      return;
    }}
    Plotly.addTraces('plot', newTraces);
    infoEl.innerHTML = 'AABB（<b style="color:#2b8a3e">绿实线=植物区 aabb</b>，' +
                       '<b style="color:#e8590c">橙虚线=可训练区 aabb_scale</b>）:<br>' + info.join('<br>');
    infoEl.style.display = 'block';
    aabbOn = true;
    btn.textContent = '隐藏 AABB';
    btn.classList.add('on');
  }} else {{
    const idx = indicesByKind('aabb');
    if (idx.length) Plotly.deleteTraces('plot', idx);
    infoEl.style.display = 'none';
    infoEl.innerHTML = '';
    aabbOn = false;
    btn.textContent = '显示 AABB';
    btn.classList.remove('on');
  }}
}}

document.querySelectorAll('#checkboxes input').forEach(cb => {{
  cb.addEventListener('change', () => toggle(cb.dataset.label, cb.checked));
}});

function setAll(state) {{
  document.querySelectorAll('#checkboxes input').forEach(cb => {{
    cb.checked = state;
    toggle(cb.dataset.label, state);
  }});
}}

window.addEventListener('resize', () => Plotly.Plots.resize('plot'));
</script>

</body></html>"""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    try:
        fig, label_to_indices, labels, warnings, cam_data, label_colors, diag, aabb_data = build_figure(TRANSFORMS_LIST)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return _render_page(fig, label_to_indices, labels, warnings, TRANSFORMS_LIST,
                        cam_data, label_colors, diag, aabb_data)


def main() -> None:
    parser = argparse.ArgumentParser(description="Web 3D viewer for NeRF transforms.json camera poses.")
    parser.add_argument("paths", nargs="+", help="One or more transforms.json file paths.")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default 0.0.0.0).")
    parser.add_argument("--port", type=int, default=7007, help="Bind port (default 7007).")
    args = parser.parse_args()

    global TRANSFORMS_LIST
    TRANSFORMS_LIST = args.paths

    print(f"[INFO] Sources ({len(TRANSFORMS_LIST)}):")
    for p in TRANSFORMS_LIST:
        print(f"         {p}")
    print(f"[INFO] Serving on http://{args.host}:{args.port}")
    print(f"[INFO] Local SSH forward example:")
    print(f"         ssh -L {args.port}:localhost:{args.port} <user>@<remote>")
    print(f"       then open http://localhost:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
