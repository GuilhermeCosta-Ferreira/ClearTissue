# ================================================================
# 0. Section: IMPORTS
# ================================================================
import json
import pytest

from clearbrain.data.downloader.json_downloader import download_json


# ================================================================
# 1. Section: Functions
# ================================================================
class TestJsonDownloader:
    def test_download_json_creates_file(self, tmp_path):
        data = {"mouse": "M001", "age": 12}
        source_filepath = tmp_path / "source_file.txt"

        output_path = download_json(
            data=data,
            source_filepath=source_filepath,
            to_update=False,
            suffix="_metadata",
        )

        assert output_path == tmp_path / "source_file_metadata.json"
        assert output_path.exists()

    def test_download_json_writes_correct_data(self, tmp_path):
        data = {"mouse": "M001", "values": [1, 2, 3]}
        source_filepath = tmp_path / "brain_volume.tif"

        output_path = download_json(
            data=data,
            source_filepath=source_filepath,
            to_update=False,
            suffix="_metadata",
        )

        with output_path.open("r", encoding="utf-8") as f:
            saved_data = json.load(f)

        assert saved_data == data

    def test_download_json_overwrites_if_update_is_true(self, tmp_path):
        old_data = {"old": True}
        new_data = {"new": True}

        source_filepath = tmp_path / "brain_volume.tif"
        existing_file = tmp_path / "brain_volume_metadata.json"

        existing_file.write_text(json.dumps(old_data), encoding="utf-8")

        output_path = download_json(
            data=new_data,
            source_filepath=source_filepath,
            to_update=True,
            suffix="_metadata",
        )

        with output_path.open("r", encoding="utf-8") as f:
            saved_data = json.load(f)

        assert saved_data == new_data

    def test_download_json_raises_if_file_exists_and_update_is_false(self, tmp_path):
        data = {"mouse": "M001"}
        source_filepath = tmp_path / "brain_volume.tif"

        existing_file = tmp_path / "brain_volume_metadata.json"
        existing_file.write_text("{}", encoding="utf-8")

        with pytest.raises(FileExistsError):
            download_json(
                data=data,
                source_filepath=source_filepath,
                to_update=False,
                suffix="_metadata",
            )
