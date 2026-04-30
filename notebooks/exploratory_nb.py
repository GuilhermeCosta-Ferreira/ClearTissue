import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    from matplotlib import pyplot as plt

    from pathlib import Path
    from typing import cast
    from scipy.spatial import KDTree

    from clearbrain.sections.sections import get_tangent_vector, get_basis_vector

    from clearbrain import (
        load_points,
        plot_3d_clear_points
    )
    from clearbrain.preprocess import(
        scale_points,
        filter_low_density_points,
    )
    from clearbrain.centerline import (
        get_centerline,
        smooth_centerline,
        add_centerline
    )
    from clearbrain.sections import get_spinal_sections, add_spinal_sections
    from clearbrain.save import save_to_json

    from clearbrain.sections.projection import get_2d_sections
    from clearbrain.sections.plot_projections import plot_section_2d
    from clearbrain.sections.pca import get_image_pca_components

    return (
        KDTree,
        Path,
        add_centerline,
        add_spinal_sections,
        cast,
        get_2d_sections,
        get_basis_vector,
        get_centerline,
        get_image_pca_components,
        get_spinal_sections,
        get_tangent_vector,
        load_points,
        mo,
        np,
        plot_3d_clear_points,
        plot_section_2d,
        plt,
        scale_points,
        smooth_centerline,
    )


@app.cell
def _():
    #MOUSE = mo.ui.text(placeholder="32B")
    MOUSE = "32B"
    return (MOUSE,)


@app.cell(hide_code=True)
def _(MOUSE, mo):
    mo.md(rf"""
    # Exploratory Notebook
    This notebokok will serve to view the clear tissue data and learn which preprocessing will be needed to obtain proper results

    Firs let's import all the important data. We can pick which mouse we want to inspect and visualize for this step. As a default let's start with `32B`, but you can pick another one here: {MOUSE}
    """)
    return


@app.cell
def _(MOUSE, Path):
    # IO Settings
    ROOT_FOLDER: Path = Path(__file__).resolve().parents[1]
    DATA_FOLDER: Path = ROOT_FOLDER / "data"
    #MOUSE: str = "32B"

    # Scaling Setting
    SCALING: tuple[float, float, float] = (2.22, 1.0, 1.0)

    # Filtering Setting
    DENSITY_RADIUS: int = 50
    MIN_DENSITY: int = 25 #20 [20, 50[, [20, 30]

    # Plotting Settings
    PLOT_SUBSAMPLE: int = 80  # Get's every X points

    BIN_WIDTH: int = 500
    HIGHLIGHT_CENTERLINE: bool = True # makes sure the line is drawn on top of it

    # Centerline Smoothing Settings
    SPLINE_SMOOTHING: float = 5000.0  # ← extremely smooth (as requested)
    N_POINTS_ON_LINE: float = 4000  # more points = perfectly smooth visual

    PRISM_HALF_WIDTH: int = 1000        # ← 2000×2000 square base (as requested)
    PRISM_HALF_THICKNESS: int = 250     # ← 500 units thick
    N_CUTS: int = 10 

    #filepath = DATA_FOLDER / MOUSE.value / "raw_points_sc.json"
    filepath = DATA_FOLDER / MOUSE / "raw_points_sc.json"
    return (
        BIN_WIDTH,
        DENSITY_RADIUS,
        HIGHLIGHT_CENTERLINE,
        N_CUTS,
        N_POINTS_ON_LINE,
        PLOT_SUBSAMPLE,
        PRISM_HALF_THICKNESS,
        PRISM_HALF_WIDTH,
        SCALING,
        SPLINE_SMOOTHING,
        filepath,
    )


@app.cell
def _(
    PLOT_SUBSAMPLE: int,
    SCALING: tuple[float, float, float],
    filepath,
    load_points,
    plot_3d_clear_points,
    scale_points,
):
    points = load_points(filepath)
    points = scale_points(points, SCALING)
    #points = filter_low_density_points(points, DENSITY_RADIUS, MIN_DENSITY)

    # 4. Generate the 3D plot
    fig, ax = plot_3d_clear_points(points, PLOT_SUBSAMPLE)
    return fig, points


@app.cell
def _(
    BIN_WIDTH: int,
    HIGHLIGHT_CENTERLINE: bool,
    N_POINTS_ON_LINE: float,
    PLOT_SUBSAMPLE: int,
    SPLINE_SMOOTHING: float,
    add_centerline,
    get_centerline,
    plot_3d_clear_points,
    points,
    smooth_centerline,
):
    centerline = get_centerline(points, BIN_WIDTH)
    centerline = smooth_centerline(centerline, SPLINE_SMOOTHING, N_POINTS_ON_LINE)

    # 3. Generate the 3D plot
    fig2, ax2 = plot_3d_clear_points(points, PLOT_SUBSAMPLE)
    ax2 = add_centerline(ax2, centerline, HIGHLIGHT_CENTERLINE)
    return centerline, fig2


@app.cell
def _(
    HIGHLIGHT_CENTERLINE: bool,
    N_CUTS: int,
    PLOT_SUBSAMPLE: int,
    PRISM_HALF_THICKNESS: int,
    PRISM_HALF_WIDTH: int,
    add_centerline,
    add_spinal_sections,
    centerline,
    get_spinal_sections,
    plot_3d_clear_points,
    points,
):
    spinal_sections, section_points, section_centers = get_spinal_sections(
        points,
        centerline,
        N_CUTS,
        PRISM_HALF_WIDTH,
        PRISM_HALF_THICKNESS
    )

    # 3. Generate the 3D plot
    fig3, ax3 = plot_3d_clear_points(points, PLOT_SUBSAMPLE)
    ax3 = add_centerline(ax3, centerline, HIGHLIGHT_CENTERLINE)
    ax3 = add_spinal_sections(ax3, spinal_sections)
    return fig3, section_centers, section_points


@app.cell
def _(fig, fig2, fig3, mo):
    tabs = {
        "Raw Data": mo.mpl.interactive(fig),
        "Centerline": mo.mpl.interactive(fig2),
        "Spinal Sections": mo.mpl.interactive(fig3)
    }

    mo.ui.tabs(tabs)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    With the provided sections we can observe now the different cfos densities across a given spinal section
    """)
    return


@app.cell
def _(
    DENSITY_RADIUS: int,
    KDTree,
    cast,
    centerline,
    get_basis_vector,
    get_tangent_vector,
    mo,
    np,
    plt,
    section_centers,
    section_points,
):
    def _():
        fig, axes = plt.subplots(2, 5, figsize=(10, 5))
        axes = axes.flatten()
        for i, prism_points in enumerate(section_points):
            ax = axes[i]
            centerline_pos = section_centers[i]
            idx = cast(int, np.argmin(np.linalg.norm(centerline - centerline_pos, axis=1)))

            # 2.2. Get the tangent vector
            tagent_vector = get_tangent_vector(centerline, idx, centerline_pos)

            # 2.3 Get the perpendicular basis vectors
            u, v = get_basis_vector(tagent_vector)

            # Project to 2D local coordinates (U-V plane)
            local_u = np.dot(prism_points - centerline_pos, u)
            local_v = np.dot(prism_points - centerline_pos, v)

            # Density heatmap
            tree2d = KDTree(np.column_stack((local_u, local_v)))
            densities = np.array([len(tree2d.query_ball_point(p, r=DENSITY_RADIUS)) - 1
                                  for p in np.column_stack((local_u, local_v))])

            sc = ax.scatter(local_u, local_v, c=densities, s=1, cmap='hot', alpha=0.9)
            ax.grid(True, alpha=0.3)
            #plt.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
            ax.set_xlabel("Local U")
            ax.set_ylabel("Local V")
            ax.set_aspect('equal')

        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        return fig, axes


    fig4, ax4 = _()

    mo.mpl.interactive(fig4)
    return


@app.cell
def _(
    centerline,
    get_2d_sections,
    get_image_pca_components,
    mo,
    plot_section_2d,
    section_centers,
    section_points,
):
    prism_imgs = get_2d_sections(section_points, section_centers, centerline)

    print(get_image_pca_components(prism_imgs[0]))

    fig5, ax5 = plot_section_2d(prism_imgs)
    mo.mpl.interactive(fig5)
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
