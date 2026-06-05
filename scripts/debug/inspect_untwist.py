# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np
from matplotlib import pyplot as plt

from PIL import Image
from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from clearbrain.tissue import TissueType
from reportlab.lib.utils import ImageReader
from clearbrain.data import TissueLoader, TissueSource



# ================================================================
# 1. Section: INPUTS
# ================================================================
DATA_FOLDER: Path = Path("data")
MOUSE: str = "32B"
TISSUE_TYPE: TissueType = TissueType.SPINAL_COORD


# ================================================================
# 2. Section: FUNCTIONS
# ================================================================
def _to_pil_image(img: np.ndarray) -> Image.Image:
    # Remove singleton dimensions
    img = np.squeeze(img)

    # Normalize floats or non-uint8 arrays to uint8
    if img.dtype != np.uint8:
        img = img.astype(np.float32)
        min_val = np.nanmin(img)
        max_val = np.nanmax(img)

        if max_val > min_val:
            img = (img - min_val) / (max_val - min_val)
        else:
            img = np.zeros_like(img)

        img = (255 * img).clip(0, 255).astype(np.uint8)

    # Grayscale: H x W
    if img.ndim == 2:
        return Image.fromarray(img, mode="L").convert("RGB")

    # RGB / RGBA: H x W x C
    if img.ndim == 3:
        if img.shape[-1] == 1:
            return Image.fromarray(img[..., 0], mode="L").convert("RGB")
        if img.shape[-1] == 3:
            return Image.fromarray(img, mode="RGB")
        if img.shape[-1] == 4:
            return Image.fromarray(img, mode="RGBA").convert("RGB")

    raise ValueError(f"Unsupported image shape: {img.shape}")


def images_to_grid_pdf(
    images,
    output_path="images_grid.pdf",
    n_cols=5,
    n_rows=7,
    page_size=A4,
    margin=24,
    padding=6,
):
    c = canvas.Canvas(output_path, pagesize=page_size)
    page_width, page_height = page_size

    usable_width = page_width - 2 * margin
    usable_height = page_height - 2 * margin

    cell_width = (usable_width - padding * (n_cols - 1)) / n_cols
    cell_height = (usable_height - padding * (n_rows - 1)) / n_rows

    images_per_page = n_cols * n_rows

    for idx, img in enumerate(images):
        pos_on_page = idx % images_per_page

        if idx > 0 and pos_on_page == 0:
            c.showPage()

        row = pos_on_page // n_cols
        col = pos_on_page % n_cols

        pil_img = _to_pil_image(img)
        img_width, img_height = pil_img.size
        aspect = img_width / img_height

        # Fit image inside cell while preserving aspect ratio
        draw_width = cell_width
        draw_height = draw_width / aspect

        if draw_height > cell_height:
            draw_height = cell_height
            draw_width = draw_height * aspect

        x_cell = margin + col * (cell_width + padding)
        y_cell = page_height - margin - (row + 1) * cell_height - row * padding

        x = x_cell + (cell_width - draw_width) / 2
        y = y_cell + (cell_height - draw_height) / 2

        c.drawImage(
            ImageReader(pil_img),
            x,
            y,
            width=draw_width,
            height=draw_height,
            preserveAspectRatio=True,
            mask="auto",
        )

    c.save()


# ================================================================
# 3. Section: MAIN
# ================================================================
if __name__ == '__main__':
    source = TissueSource(mouse=MOUSE, tissue_type=TISSUE_TYPE, base_path=DATA_FOLDER)
    loader = TissueLoader(source)
    twisting_data = loader.load_twisting_data()

    angles = np.asarray([result.transform.GetParameters()[0] for result in twisting_data])
    timesteps = np.asarray([result.elapsed_time for result in twisting_data])
    images = np.asarray([result.registered_image for result in twisting_data])

    """ images_to_grid_pdf(
        images,
        output_path="registered_images.pdf",
        n_cols=6,
        n_rows=8,
    ) """

    plt.figure()
    plt.plot(angles)
    #plt.plot(timesteps)
    plt.xlabel("Coronal Axis")
    plt.ylabel("Angle")
    plt.show(block=False)

    # gradient
    plt.figure()
    plt.plot(angles[1:] - angles[:-1])
    plt.show(block=False)

    plt.figure()
    plt.subplot(2, 3, 1)
    plt.imshow(images[13])
    plt.subplot(2, 3, 2)
    plt.imshow(images[14])
    plt.subplot(2, 3, 3)
    plt.imshow(images[15])
    plt.subplot(2, 3, 4)
    plt.imshow(images[16])
    plt.subplot(2, 3, 5)
    plt.imshow(images[17])
    plt.subplot(2, 3, 6)
    plt.imshow(images[18])
    plt.show(block=False)

    input("Press Enter to close...")
    plt.close('all')
