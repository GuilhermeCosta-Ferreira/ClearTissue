# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np
import polars as pl

from pathlib import Path
from numpy.typing import NDArray
from dataclasses import dataclass, field
from brainglobe_atlasapi import BrainGlobeAtlas
from tqdm import tqdm

from cleartissue.tissue import TissueType, ClearVolume
from cleartissue.data import TissueLoader, TissueSource



# ================================================================
# 1. Section: INPUTS
# ================================================================
DATA_FOLDER: Path = Path("data")
MOUSE: str = "32B"
TISSUE_TYPE: TissueType = TissueType.SPINAL_COORD

BACKGROUND_LABEL: int = 0



# ================================================================
# 2. Section: FUNCTIONS
# ================================================================
@dataclass
class SegmentData:
    segment_id: int
    segment_name: str
    segment_acronym: str
    cell_count: int
    segment_volume: int
    cell_density: float
    side: int #0: is total, 1: is left and 2 is right

@dataclass
class AtlasData:
    left_segments: list[SegmentData] = field(default_factory=list)
    right_segments: list[SegmentData] = field(default_factory=list)

    @property
    def total_segments(self) -> list[SegmentData]:
        total_segments = []
        for idx, l_seg in enumerate(self.left_segments):
            r_seg = self.right_segments[idx]

            if l_seg.segment_id != r_seg.segment_id:
                raise ValueError(f"Segment ids do not match: {l_seg.segment_id} != {r_seg.segment_id}")

            total_count = l_seg.cell_count + r_seg.cell_count
            segment_volume = l_seg.segment_volume + r_seg.segment_volume
            cell_density = total_count / segment_volume if segment_volume != 0 else 0

            total_data = SegmentData(
                segment_id=l_seg.segment_id,
                segment_name=l_seg.segment_name,
                segment_acronym=l_seg.segment_acronym,
                cell_count=total_count,
                segment_volume=segment_volume,
                cell_density=cell_density,
                side=0)
            total_segments.append(total_data)

        return total_segments

    @property
    def segment_names(self) -> list[str]:
        return [seg.segment_name for seg in self.total_segments]

    @property
    def segment_acronyms(self) -> list[str]:
        return [seg.segment_acronym for seg in self.total_segments]

    @property
    def segment_ids(self) -> list[int]:
        return [seg.segment_id for seg in self.total_segments]

    @property
    def total_cell_densities(self) -> list[float]:
        return [seg.cell_density for seg in self.total_segments]

    @property
    def left_cell_densities(self) -> list[float]:
        return [seg.cell_density for seg in self.left_segments]

    @property
    def right_cell_densities(self) -> list[float]:
        return [seg.cell_density for seg in self.right_segments]

    @property
    def dataframe_density(self) -> pl.DataFrame:
        return pl.DataFrame({
            "segment_name": self.segment_names,
            "segment_acronym": self.segment_acronyms,
            "segment_id": self.segment_ids,
            "left_cell_density": self.left_cell_densities,
            "right_cell_density": self.right_cell_densities,
            "total_cell_density": self.total_cell_densities,
        })


def get_lateralized_data(atlas: BrainGlobeAtlas, cells: ClearVolume, lateralized_volume: NDArray, label: int, side: int):
    structure = atlas.structures[int(label)]
    name = structure["name"]
    acronym = structure["acronym"]

    label_mask = np.where(lateralized_volume == label + side * 0.1, 1, 0)
    cells_in_label = cells.volume * label_mask

    cell_count = np.sum(cells_in_label)
    segment_volume = np.sum(label_mask)

    cell_density = cell_count / segment_volume if segment_volume != 0 else 0

    seg_data = SegmentData(
        segment_id=label,
        segment_name=name,
        segment_acronym=acronym,
        cell_count=cell_count,
        segment_volume=segment_volume,
        cell_density=cell_density,
        side=side)
    return seg_data


# ================================================================
# 3. Section: MAIN
# ================================================================
if __name__ == '__main__':
    source = TissueSource(mouse=MOUSE, tissue_type=TISSUE_TYPE, base_path=DATA_FOLDER)
    loader = TissueLoader(source)
    cells = loader.load_volume(suffix="_cells_untwisted_cleaned")
    reg_atlas = loader.load_volume(suffix="_registered_atlas")
    reg_hemisphere = loader.load_volume(suffix="_registered_hemisphere")

    atlas = BrainGlobeAtlas("allen_cord_20um", check_latest=False)

    present_labels = np.unique(reg_atlas.volume)
    lateralized_volume = reg_atlas.volume + reg_hemisphere.volume * 0.1

    atlas_data = AtlasData()
    for present_lb in tqdm(present_labels):
        if present_lb == BACKGROUND_LABEL:
            continue

        left_seg_data = get_lateralized_data(atlas, cells, lateralized_volume, present_lb, side=1)
        right_seg_data = get_lateralized_data(atlas, cells, lateralized_volume, present_lb, side=2)
        atlas_data.left_segments.append(left_seg_data)
        atlas_data.right_segments.append(right_seg_data)

    print(atlas_data.dataframe_density)

    atlas_data.dataframe_density.write_csv("output.csv")
