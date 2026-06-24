# ================================================================
# 0. Section: IMPORTS
# ================================================================
import os
import trimesh

import numpy as np

from skimage import measure
from numpy.typing import NDArray

from cleartissue.domain_model.data import TissueType
from cleartissue.service.ClearTissueProject import ClearTissueProject
import cleartissue.domain_model.transformations as tr



# ================================================================
# 0. Section: IMPORTS
# ================================================================
def separate_brain_parts(brain_labels: np.ndarray) -> NDArray:
    brain_parts = []
    for label in np.unique(brain_labels):
        if(label == 0):
            continue  # Skip background
        part = np.where(brain_labels == label, label, 0)
        brain_parts.append(part)
    return np.array(brain_parts)

def build_mesh(voxel_array: NDArray) -> trimesh.Trimesh:
    # Prepare Data
    binary = np.where(voxel_array > 0, 1, 0)

    # Extract surface mesh
    verts, faces, _, _ = measure.marching_cubes(binary, level=0.5)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)

    return mesh




# ================================================================
# 3. Section: MAIN
# ================================================================
if __name__ == '__main__':
    project = ClearTissueProject.load(
        mouse="32B",
        tissue_type=TissueType.SPINAL_CORD,
    )
    raw_batch = project.load_raw()

    pipeline = project.init_pipeline("Cell Counts")

    input("Setup the config. Press enter when ready")

    pipeline.add_list([
        tr.RegularizeSample,
        tr.OrientSample,
        tr.StretchSample,
        tr.UntwistSample,
        tr.RotateSample,
        tr.CylindricalMaskSample,
        tr.EmptySpaceTrimSample,
        tr.SizeMatchedAtlasRegistration,
    ])

    final_batch = project.run_pipeline(pipeline, raw_batch)

    print(np.unique(final_batch.atlas.data).size)
    separate_atlas = separate_brain_parts(final_batch.atlas.data)

    for part in separate_atlas:
        build_mesh(part)
