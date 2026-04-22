import json
from pathlib import Path
from clearbrain import load_points
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import os
from math import ceil

# ====================== CONFIG ======================
main_dir = "/Users/guilhermec.f/Documents/EPFL/ClearBrain/data/32B"
plot_dir = os.path.join(main_dir, "plot")
axial_dir = os.path.join(plot_dir, "axial_slice")
os.makedirs(axial_dir, exist_ok=True)

X_scale = 2.22
Y_scale = 1.0
Z_scale = 1.0

prism_half_width = 1000        # 2000×2000 square
prism_half_thickness = 250     # 500 units thick
n_cuts = 10
density_radius_2d = 40         # radius for local density heatmap

json_files = ["raw_points_sc.json"]

print("Creating SINGLE PNG with 10 axial cuts (points inside prisms + density heatmap)...\n")

for filename in json_files:
    filepath = os.path.join(main_dir, filename)
    if not os.path.exists(filepath):
        continue

    sample_name = os.path.splitext(filename)[0]
    print(f"Processing {sample_name}...")

    # Load points
    with open(filepath, "r") as f:
        points_full = np.array(json.load(f))
    points_full[:, 0] *= X_scale
    points_full[:, 1] *= Y_scale
    points_full[:, 2] *= Z_scale

    # Load Spinal Cord Orientation Line
    orientation_path = os.path.join(main_dir, "centerline_sc.json")
    if not os.path.exists(orientation_path):
        print(f"❌ Orientation line missing! Run the previous script first {orientation_path}.")
        continue

    orientation_line = load_points(Path(orientation_path))

    # Select 10 positions further from extremities
    start_idx = int(0.05 * len(orientation_line))
    end_idx   = int(0.95 * len(orientation_line))
    indices = np.linspace(start_idx, end_idx, n_cuts, dtype=int)

    # Prepare grid (2 rows × 5 columns for 10 cuts)
    fig, axes = plt.subplots(2, 5, figsize=(20, 8), dpi=300)
    axes = axes.flatten()

    for i, idx in enumerate(indices):
        ax = axes[i]
        P = orientation_line[idx]

        # Tangent T
        if idx < len(orientation_line) - 1:
            T = orientation_line[idx + 1] - P
        else:
            T = P - orientation_line[idx - 1]
        T /= np.linalg.norm(T)

        # Perpendicular U, V
        arbitrary = np.array([1., 0., 0.])
        if abs(np.dot(T, arbitrary)) > 0.99:
            arbitrary = np.array([0., 1., 0.])
        U = np.cross(T, arbitrary)
        U /= np.linalg.norm(U)
        V = np.cross(T, U)

        # Filter points inside the prism
        vec = points_full - P
        dist_along_T = np.dot(vec, T)
        inside_thickness = np.abs(dist_along_T) <= prism_half_thickness

        proj_U = np.dot(vec, U)
        proj_V = np.dot(vec, V)
        inside_square = (np.abs(proj_U) <= prism_half_width) & (np.abs(proj_V) <= prism_half_width)

        mask = inside_thickness & inside_square
        slice_pts = points_full[mask]

        if len(slice_pts) < 10:
            ax.axis('off')
            ax.set_title(f'Axial Cut {i+1}\n(no points)', fontsize=10)
            continue

        # Project to 2D local coordinates (U-V plane)
        local_u = np.dot(slice_pts - P, U)
        local_v = np.dot(slice_pts - P, V)

        # Density heatmap
        tree2d = KDTree(np.column_stack((local_u, local_v)))
        densities = np.array([len(tree2d.query_ball_point(p, r=density_radius_2d)) - 1
                              for p in np.column_stack((local_u, local_v))])

        sc = ax.scatter(local_u, local_v, c=densities, s=8, cmap='hot', alpha=0.9)
        ax.set_title(f'Axial Cut {i+1}\nY-pos ≈ {orientation_line[idx,1]:.0f}', fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
        ax.set_xlabel("Local U")
        ax.set_ylabel("Local V")
        ax.set_aspect('equal')

    # Turn off any unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    fig.suptitle(f"{sample_name} — 10 Axial Cuts (2000×2000 × 500 thick)\nDensity Heatmap of points inside each prism", fontsize=16, y=0.98)
    plt.tight_layout()

    # SAVE SINGLE PNG
    output_png = os.path.join(axial_dir, f"{sample_name}_10_AXIAL_CUTS_2000x2000_500thick_DENSITY.png")
    fig.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"   ✅ BIG PNG with all 10 heatmaps saved → {output_png}\n")

print("🎉 DONE!")
print("Open the new PNG files in plot/axial_slice/")
print("Each of the 10 panels shows ONLY the points inside that exact prism, projected to 2D, with density heatmap.")
print("The prisms are 2000×2000 × 500 thick and placed away from the extremities.")
