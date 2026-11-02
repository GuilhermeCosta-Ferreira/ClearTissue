"""Build a self-contained 3D HTML viewer of a BrainGlobe cord atlas.

Renders every annotated region of a reference atlas as its own translucent
surface in one portable ``.html`` file (Plotly, plotly.js from CDN). A search
box typed into the page matches an acronym, name, or structure id as you type
and highlights the matching region; because the match is resolved against the
full ontology tree, a *collapsed parent* that never appears in the annotation
array itself (e.g. "grey matter") highlights the union of all its descendant
leaf regions that do appear.

Everything is precomputed into the page — no server, no mouse data; open the
file by double-clicking. All logic lives in this script; it reads the atlas
through BrainGlobe and writes a single HTML file.

Main components:
    build_region_meshes: annotation volume -> one surface mesh per region.
    build_searchable_index: ontology -> every acronym/name/id that can
        highlight something, mapped to the leaf traces it covers.
    render_html: assemble the Plotly figure and the search UI into one file.
"""

# ================================================================
# 0. Section: IMPORTS
# ================================================================
import json
import argparse
import webbrowser

from pathlib import Path
from dataclasses import dataclass

import numpy as np

from numpy.typing import NDArray
from scipy.ndimage import find_objects, zoom
from skimage.measure import marching_cubes  # pylint: disable=no-name-in-module
from brainglobe_atlasapi import BrainGlobeAtlas

# ================================================================
# 1. Section: Constants
# ================================================================
_BACKGROUND_LABEL: int = 0

# Per-trace opacities: very faint context, fully opaque when highlighted so
# even small regions stand out against the dimmed rest.
_BASE_OPACITY: float = 0.06
_HIGH_OPACITY: float = 1.0
_FAINT_OPACITY: float = 0.02

# Edge contour drawn on a region only while it is selected.
_CONTOUR_COLOR: str = "#111111"
_CONTOUR_WIDTH: int = 4

_PLOT_ID: str = "atlas-plot"

# Isotropic render grid; the main lever on file size. 60 µm keeps laminae
# distinguishable while staying small.
_VOXEL_UM: float = 60.0

# Marching-cubes decimation; higher = coarser surfaces and a smaller file.
_MESH_STEP: int = 1

# Fallback colour for regions whose ontology entry has no rgb_triplet.
_FALLBACK_RGB: tuple[int, int, int] = (150, 150, 150)


# ================================================================
# 2. Section: Types
# ================================================================
@dataclass(frozen=True)
class RegionMesh:
    """One region's surface and identity, ready to become a Plotly trace.

    Attributes:
        label: annotation id of the region (always a leaf present in the array).
        acronym: region acronym, used as the trace/hover name.
        name: full region name.
        rgb: display colour as an ``(r, g, b)`` 0-255 triplet.
        verts: ``(V, 3)`` vertex coordinates, already scaled to microns.
        faces: ``(F, 3)`` triangle vertex indices.
    """

    label: int
    acronym: str
    name: str
    rgb: tuple[int, int, int]
    verts: NDArray
    faces: NDArray


# ================================================================
# 3. Section: Mesh + index building
# ================================================================
def build_region_meshes(
    atlas: BrainGlobeAtlas,
    voxel_um: float,
    step: int,
) -> list[RegionMesh]:
    """Extract one surface mesh per annotated region on an isotropic grid.

    The annotation is first resampled (nearest-neighbour, so labels are kept)
    to an isotropic ``voxel_um`` grid — the single biggest lever on output file
    size — then each region is cropped to its bounding box and surfaced with
    marching cubes. Vertices come back in microns, mapped to a cord-friendly
    frame (rostro-caudal along x, left-right along y, dorso-ventral up along z).

    Args:
        voxel_um: isotropic voxel size of the render grid; larger is coarser
            and yields a much smaller file.
        step: marching-cubes step size; larger decimates each surface further.
    """
    annotation = _resample_isotropic(
        np.asarray(atlas.annotation), atlas.resolution, voxel_um
    )

    present = np.unique(annotation)
    present = present[present != _BACKGROUND_LABEL]

    # Compact 1..K relabel so find_objects returns a short bounding-box list.
    compact = np.zeros_like(annotation, dtype=np.int32)
    for compact_id, label in enumerate(present, start=1):
        compact[annotation == label] = compact_id
    boxes = find_objects(compact)

    meshes: list[RegionMesh] = []
    for compact_id, label in enumerate(present, start=1):
        box = boxes[compact_id - 1]
        if box is None:
            continue

        mesh = _mesh_one_region(annotation, int(label), box, voxel_um, step, atlas)
        if mesh is not None:
            meshes.append(mesh)

    return meshes


def build_searchable_index(
    atlas: BrainGlobeAtlas,
    meshes: list[RegionMesh],
) -> list[dict]:
    """Map every searchable region to the leaf traces it highlights.

    Walks the ontology and, for each structure that has at least one present
    descendant, records the set of mesh trace indices covered. Leaf regions map
    to themselves; collapsed parents (absent from the annotation array) map to
    all their present descendants, which is what lets them be highlighted.

    Returns:
        One dict per searchable region with ``id``, ``acronym``, ``name``,
        ``collapsed`` (not directly in the array) and ``traces`` (trace indices).
    """
    trace_of_label = {mesh.label: index for index, mesh in enumerate(meshes)}
    present_labels = set(trace_of_label)

    descendants: dict[int, list[int]] = {}
    for label in present_labels:
        for ancestor in atlas.structures[label]["structure_id_path"]:
            descendants.setdefault(int(ancestor), []).append(label)

    index: list[dict] = []
    for region_id, leaf_labels in descendants.items():
        structure = atlas.structures[region_id]
        index.append(
            {
                "id": region_id,
                "acronym": structure["acronym"],
                "name": structure["name"],
                "collapsed": region_id not in present_labels,
                "traces": sorted(trace_of_label[label] for label in leaf_labels),
            }
        )

    index.sort(key=lambda entry: entry["acronym"].lower())
    return index


# ──────────────────────────────────────────────────────
# 3.1 Subsection: Helper Functions
# ──────────────────────────────────────────────────────
def _resample_isotropic(
    annotation: NDArray,
    resolution: tuple[float, float, float],
    voxel_um: float,
) -> NDArray:
    factors = np.asarray(resolution, dtype=float) / voxel_um
    if np.allclose(factors, 1.0):
        return annotation
    return zoom(annotation, factors, order=0)  # order 0 keeps integer labels


def _mesh_one_region(
    annotation: NDArray,
    label: int,
    box: tuple[slice, slice, slice],
    voxel_um: float,
    step: int,
    atlas: BrainGlobeAtlas,
) -> RegionMesh | None:
    # Pad by one voxel so surfaces on the bounding-box edge still close.
    padded = tuple(slice(max(s.start - 1, 0), s.stop + 1) for s in box)
    sub = annotation[padded] == label
    if sub.sum() < 2 or any(dim < 2 for dim in sub.shape):
        return None

    sub = np.pad(sub, 1, mode="constant", constant_values=False)
    try:
        verts, faces, _, _ = marching_cubes(
            sub.astype(np.float32), level=0.5, step_size=step
        )
    except (ValueError, RuntimeError):
        return None

    # Undo the two paddings, add the bounding-box origin, scale to microns.
    origins = np.array([s.start - 2 for s in padded], dtype=np.float64)
    verts = (verts + origins) * voxel_um

    # Map annotation axes (ap, si, rl) to display (x=ap, y=rl, z=si-up).
    display = (
        np.column_stack([verts[:, 0], verts[:, 2], verts[:, 1]])
        .round()
        .astype(np.int32)
    )

    structure = atlas.structures[label]
    rgb = tuple(structure.get("rgb_triplet") or _FALLBACK_RGB)
    return RegionMesh(
        label=label,
        acronym=str(structure["acronym"]),
        name=str(structure["name"]),
        rgb=rgb,  # type: ignore[arg-type]
        verts=display,
        faces=faces.astype(np.int32),
    )


# ================================================================
# 4. Section: HTML rendering
# ================================================================
def render_html(meshes: list[RegionMesh], index: list[dict], title: str) -> str:
    """Assemble the Plotly figure and the search UI into one HTML document."""
    import plotly.graph_objects as go

    figure = go.Figure()
    for mesh in meshes:
        r, g, b = mesh.rgb
        figure.add_trace(
            go.Mesh3d(
                x=mesh.verts[:, 0],
                y=mesh.verts[:, 1],
                z=mesh.verts[:, 2],
                i=mesh.faces[:, 0],
                j=mesh.faces[:, 1],
                k=mesh.faces[:, 2],
                color=f"rgb({r},{g},{b})",
                opacity=_BASE_OPACITY,
                name=mesh.acronym,
                hovertext=f"{mesh.acronym} — {mesh.name}",
                hoverinfo="text",
                flatshading=True,
                showscale=False,
                contour=dict(show=False, color=_CONTOUR_COLOR, width=_CONTOUR_WIDTH),
            )
        )

    figure.update_layout(
        title=title,
        showlegend=False,
        autosize=True,
        scene=dict(
            xaxis=dict(title="rostro-caudal (µm)"),
            yaxis=dict(title="left-right (µm)"),
            zaxis=dict(title="dorso-ventral (µm)"),
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
    )

    plot_div = figure.to_html(
        full_html=False,
        include_plotlyjs="cdn",
        div_id=_PLOT_ID,
        default_width="100vw",
        default_height="100vh",
        config={"responsive": True},
    )
    return _compose_document(plot_div, index, len(meshes), title)


# ──────────────────────────────────────────────────────
# 4.1 Subsection: Helper Functions
# ──────────────────────────────────────────────────────
def _compose_document(
    plot_div: str,
    index: list[dict],
    n_traces: int,
    title: str,
) -> str:
    data_script = (
        "<script>\n"
        f"const SEARCHABLE = {json.dumps(index)};\n"
        f"const NTRACES = {n_traces};\n"
        f"const PLOT_ID = {json.dumps(_PLOT_ID)};\n"
        f"const BASE_OPACITY = {_BASE_OPACITY};\n"
        f"const HIGH_OPACITY = {_HIGH_OPACITY};\n"
        f"const FAINT_OPACITY = {_FAINT_OPACITY};\n"
        "</script>"
    )

    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>{title}</title>{_STYLE}</head><body>"
        f"{_PANEL_HTML}{plot_div}{data_script}{_APP_SCRIPT}"
        "</body></html>"
    )


# ================================================================
# 5. Section: Static page assets
# ================================================================
_STYLE = """<style>
  html, body { height: 100%; margin: 0; }
  body { font-family: system-ui, sans-serif; overflow: hidden; }
  #atlas-plot, .plotly-graph-div { width: 100vw !important; height: 100vh !important; }
  #panel { position: fixed; top: 12px; left: 12px; z-index: 10; width: 320px;
    background: rgba(255,255,255,0.95); border: 1px solid #ccc; border-radius: 8px;
    padding: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.15); }
  #panel h1 { font-size: 14px; margin: 0 0 8px; }
  #region-search { width: 100%; box-sizing: border-box; padding: 6px 8px;
    font-size: 14px; border: 1px solid #bbb; border-radius: 6px; }
  #controls { display: flex; gap: 8px; margin-top: 6px; align-items: center; }
  #clear-btn { padding: 5px 10px; font-size: 13px; cursor: pointer; }
  #status { font-size: 12px; color: #555; }
  #candidates { list-style: none; margin: 8px 0 0; padding: 0; max-height: 320px;
    overflow-y: auto; }
  #candidates li { padding: 5px 7px; font-size: 13px; border-radius: 5px; cursor: pointer; }
  #candidates li:hover { background: #eef3ff; }
  #candidates li .group { color: #a15; font-size: 11px; }
</style>"""

_PANEL_HTML = """<div id='panel'>
  <h1>Atlas region search</h1>
  <input id='region-search' placeholder='Acronym, name, or id…' autocomplete='off'>
  <div id='controls'><button id='clear-btn'>Clear</button><span id='status'></span></div>
  <ul id='candidates'></ul>
</div>"""

_APP_SCRIPT = """<script>
  const gd = document.getElementById(PLOT_ID);
  const input = document.getElementById('region-search');
  const list = document.getElementById('candidates');
  const status = document.getElementById('status');

  function setOpacities(highlight) {
    const hi = new Set(highlight);
    const ops = [], contours = [], idx = [];
    for (let t = 0; t < NTRACES; t++) {
      idx.push(t);
      ops.push(hi.size ? (hi.has(t) ? HIGH_OPACITY : FAINT_OPACITY) : BASE_OPACITY);
      contours.push(hi.has(t));  // edge contour only on selected regions
    }
    Plotly.restyle(gd, {opacity: ops, 'contour.show': contours}, idx);
  }

  function matchesFor(query) {
    const q = query.trim().toLowerCase();
    if (!q) return [];
    return SEARCHABLE.filter(r =>
      r.acronym.toLowerCase().includes(q) ||
      r.name.toLowerCase().includes(q) ||
      String(r.id).includes(q)
    ).sort((a, b) => a.acronym.length - b.acronym.length);
  }

  function renderCandidates(query) {
    const matches = matchesFor(query);
    list.innerHTML = '';
    if (!query.trim()) { status.textContent = 'Type an acronym, name, or id'; return matches; }
    matches.slice(0, 60).forEach(r => {
      const li = document.createElement('li');
      const group = r.collapsed ? " <span class='group'>(group · " + r.traces.length + ")</span>" : '';
      li.innerHTML = r.acronym + ' — ' + r.name + " <span style='color:#999'>#" + r.id + '</span>' + group;
      li.onclick = () => selectRegion(r);
      list.appendChild(li);
    });
    status.textContent = matches.length + ' match(es)';
    return matches;
  }

  function selectRegion(r) {
    setOpacities(r.traces);
    const n = r.traces.length;
    status.textContent = 'Showing ' + r.acronym + ' — ' + r.name +
      ' (' + n + ' region' + (n > 1 ? 's' : '') + ')';
  }

  input.addEventListener('input', () => renderCandidates(input.value));
  input.addEventListener('keydown', e => {
    if (e.key === 'Enter') { const m = renderCandidates(input.value); if (m.length) selectRegion(m[0]); }
  });
  document.getElementById('clear-btn').onclick = () => {
    input.value = ''; list.innerHTML = ''; setOpacities([]); status.textContent = 'Cleared';
  };
  setOpacities([]);
</script>"""


# ================================================================
# 6. Section: CLI + Main
# ================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a 3D HTML atlas viewer with region search.",
    )
    parser.add_argument(
        "--atlas",
        default="allen_cord_20um",
        help="BrainGlobe atlas name (default: allen_cord_20um)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output HTML path (default: renders/atlas/<atlas>.html)",
    )
    parser.add_argument(
        "--voxel-um",
        type=float,
        default=_VOXEL_UM,
        help=f"Isotropic render voxel size in µm (default: {_VOXEL_UM})",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=_MESH_STEP,
        help=f"Marching-cubes decimation step (default: {_MESH_STEP})",
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Do not open the file in a browser after building",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    atlas = BrainGlobeAtlas(args.atlas)
    meshes = build_region_meshes(atlas, voxel_um=args.voxel_um, step=args.step)
    index = build_searchable_index(atlas, meshes)

    out_path = args.out or Path("renders") / "atlas" / f"{args.atlas}.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    html = render_html(meshes, index, title=f"{args.atlas} — region viewer")
    out_path.write_text(html, encoding="utf-8")

    print(f"Built {len(meshes)} region meshes, {len(index)} searchable regions")
    print(f"Wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")

    if not args.no_open:
        webbrowser.open(out_path.resolve().as_uri())


if __name__ == "__main__":
    main()
