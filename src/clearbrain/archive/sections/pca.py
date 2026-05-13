# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np


# ================================================================
# 1. Section: Functions
# ================================================================
def get_image_pca_components(volume: np.ndarray) -> np.ndarray:
    """Compute the PCA components of the brain MRI data. 1PC is AP, 2PC is LR, 3PC is IS."""
    mri_mask = np.where(volume > 0, 1, 0)

    coords = np.column_stack(np.where(mri_mask > 0))
    coords_mean = np.mean(coords, axis=0)
    centered_coords = coords - coords_mean

    cov_matrix = np.cov(centered_coords, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    sorted_indices = np.argsort(eigenvalues)[::-1]
    principal_components = eigenvectors[:, sorted_indices]

    return principal_components
