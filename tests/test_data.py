import json
import importlib.util
import hashlib
from pathlib import Path
import runpy
import sys

import numpy as np
import pandas as pd
import pytest

from cellselect.data import (
    CHANNELS,
    assign_group_split,
    load_prepared,
    make_cohort,
    training_statistics,
)


def metadata_fixture(n_fields=10):
    n = n_fields * 3
    index = pd.DataFrame(
        {
            "global_index": np.arange(n),
            "batch": 0,
            "antibodies": "all",
            "data_path": "antibodies_all/batch_0/slide_replicate_0",
            "slide_replicate": 0,
            "fov_center_x": np.repeat(np.arange(n_fields), 3),
            "fov_center_y": 0,
            "fov_center_z": 1,
            "position_in_fov_x_pix": np.tile([500, 1000, 1500], n_fields),
            "position_in_fov_y_pix": 1000,
        }
    ).set_index("global_index")
    markers = pd.DataFrame(index=index.index)
    for gate in range(10):
        markers[f"gate{gate}_selection_gated"] = False
    for row in range(n):
        markers.loc[row, f"gate{[0, 1, 7][row % 3]}_selection_gated"] = True
    markers["CD16_full_model_unmixed"] = np.arange(n, dtype=float) + 1
    markers["CD45_full_model_unmixed"] = np.arange(n, dtype=float) + 2
    return index, markers


def test_cohort_does_not_invent_missing_or_ambiguous_labels():
    index, markers = metadata_fixture()
    markers.loc[0, "CD16_full_model_unmixed"] = np.nan
    markers.loc[1, "CD45_full_model_unmixed"] = np.nan
    markers.loc[2, "gate0_selection_gated"] = True  # conflicting broad classes
    cohort, qc = make_cohort(index, markers)
    assert 0 not in cohort.index and 2 not in cohort.index
    assert 1 in cohort.index and np.isnan(cohort.loc[1, "CD45"])
    assert qc["excluded_missing_or_nonfinite_CD16"] == 1
    assert qc["ambiguous_across_class_gates_global"] == 1


def test_fields_include_physical_slide_but_not_focus_plane():
    index, markers = metadata_fixture()
    index.loc[1, "fov_center_z"] = 99
    index.loc[2, "data_path"] = "antibodies_all/batch_0/slide_replicate_1"
    cohort, _ = make_cohort(index, markers)
    assert cohort.loc[0, "fov_id"] == cohort.loc[1, "fov_id"]
    assert cohort.loc[0, "fov_id"] != cohort.loc[2, "fov_id"]


def test_group_split_is_disjoint_and_outcome_independent():
    index, markers = metadata_fixture(20)
    cohort, _ = make_cohort(index, markers)
    first = assign_group_split(cohort)
    modified = cohort.copy()
    modified["CD16"] *= 1e6
    modified["class_id"] = 2 - modified["class_id"]
    pd.testing.assert_series_equal(first, assign_group_split(modified))
    assert cohort.assign(split=first).groupby("fov_id")["split"].nunique().max() == 1
    assert first.value_counts().to_dict() == {"train": 42, "val": 9, "test": 9}


def test_normalization_ignores_all_validation_and_test_values():
    images = np.arange(6 * 4 * 2 * 2, dtype=np.float32).reshape(6, 4, 2, 2)
    frame = pd.DataFrame(
        {
            "split": ["train"] * 3 + ["val", "test", "test"],
            "CD16": [1.0, 2.0, 3.0, 50.0, 60.0, 70.0],
            "CD45": [2.0, 4.0, 6.0, 50.0, 60.0, 70.0],
        }
    )
    first = training_statistics(images, frame, chunk_size=2)
    images[3:] *= -1e6
    frame.loc[3:, ["CD16", "CD45"]] = -10000
    assert first == training_statistics(images, frame, chunk_size=2)
    assert first["markers"]["CD16"]["mean"] == 2.0


def test_preparation_end_to_end_preserves_amplitude_and_locks_split(tmp_path, monkeypatch):
    import zarr

    root = tmp_path / "raw" / "BSCCMNIST"
    root.mkdir(parents=True)
    index, markers = metadata_fixture(10)
    index.to_csv(root / "BSCCM_index.csv")
    markers.to_csv(root / "BSCCM_surface_markers.csv")
    # Noncanonical channel order verifies channel-name selection.
    channel_names = ["Brightfield", "DPC_Top", "DPC_Right", "DPC_Bottom", "DPC_Left"]
    (root / "BSCCM_global_metadata.json").write_text(
        json.dumps({"name": "BSCCMNIST", "led_array": {"channel_names": channel_names}})
    )
    store = zarr.open(str(root / "BSCCM_images.zarr"), mode="w")
    for row, entry in index.iterrows():
        image = (np.arange(5 * 28 * 28).reshape(5, 28, 28) + row * 5000).astype(np.float32)
        store.create_dataset(f"{entry.data_path}/led_array/cell_{row}", data=image)
    output = tmp_path / "prepared"
    script = Path(__file__).resolve().parents[1] / "scripts" / "prepare_data.py"
    monkeypatch.setattr(
        sys,
        "argv",
        [str(script), "--source", str(root), "--output", str(output), "--data-root", str(tmp_path)],
    )
    runpy.run_path(str(script), run_name="__main__")
    data = load_prepared(output, verify_hashes=True)
    assert data.images.shape == (30, 4, 28, 28)
    assert data.images[0, 0, 0, 0] == 4 * 28 * 28
    assert data.images[1, 0, 0, 0] - data.images[0, 0, 0, 0] == 5000
    assert data.manifest["input_channels"] == list(CHANNELS)
    assert len(data.rows("train")) == 21
    assert data.metadata.groupby("fov_id")["split"].nunique().max() == 1
    with pytest.raises(RuntimeError, match="already locked"):
        runpy.run_path(str(script), run_name="__main__")


def test_chunked_archive_extraction_keeps_every_eligible_cell(tmp_path):
    import tarfile
    import zarr

    source = tmp_path / "source" / "BSCCMNIST"
    source.mkdir(parents=True)
    index, markers = metadata_fixture(10)
    index.to_csv(source / "BSCCM_index.csv")
    markers.to_csv(source / "BSCCM_surface_markers.csv")
    channel_names = ["Brightfield", *CHANNELS]
    (source / "BSCCM_global_metadata.json").write_text(
        json.dumps({"name": "BSCCMNIST", "led_array": {"channel_names": channel_names}})
    )
    arrays = zarr.open(str(source / "BSCCM_images.zarr"), mode="w")
    for row, entry in index.iterrows():
        image = np.full((5, 28, 28), row + 1, dtype=np.uint16)
        arrays.create_dataset(
            f"{entry.data_path}/led_array/cell_{row}", data=image, chunks=(1, 96, 96)
        )
    archive_path = tmp_path / "archive.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(source, arcname="BSCCMNIST")
    compressed = archive_path.read_bytes()
    halfway = len(compressed) // 2
    parts = [tmp_path / "chunk0.bin", tmp_path / "chunk1.bin"]
    parts[0].write_bytes(compressed[:halfway])
    parts[1].write_bytes(compressed[halfway:])
    script = Path(__file__).resolve().parents[1] / "scripts" / "download_data.py"
    spec = importlib.util.spec_from_file_location("download_data_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    destination = tmp_path / "unpacked"
    result = module.extract(parts, destination, all_contrasts=False)
    assert result["cohort_qc"]["eligible_cells"] == 30
    reopened = zarr.open(str(destination / "BSCCMNIST" / "BSCCM_images.zarr"), mode="r")
    for row, entry in index.iterrows():
        path = f"{entry.data_path}/led_array/cell_{row}"
        np.testing.assert_array_equal(reopened[path][1:5], row + 1)
        assert not (destination / "BSCCMNIST" / "BSCCM_images.zarr" / path / "0.0.0").exists()


def test_browser_zip_ingestion_preserves_original_and_verifies_chunks(tmp_path):
    import zipfile

    script = Path(__file__).resolve().parents[1] / "scripts" / "download_data.py"
    spec = importlib.util.spec_from_file_location("download_zip_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = tmp_path / "browser" / "download.zip"
    original.parent.mkdir()
    payloads = {
        "BSCCMNIST.tar.gz_chunk00000.bin": b"abc",
        "BSCCMNIST.tar.gz_chunk00001.bin": b"defghi",
    }
    records = [
        {"path": name, "size": len(value), "digest": hashlib.sha256(value).hexdigest()}
        for name, value in payloads.items()
    ]
    with zipfile.ZipFile(original, "w") as archive:
        for name, value in payloads.items():
            archive.writestr(name, value)
        archive.writestr("../../unrelated.txt", b"must not be unpacked")
    output = tmp_path / "data"
    result = module.ingest_public_zip(original, output, records)
    assert original.is_file()
    assert result["method"] == "official_public_selected_files_zip"
    for name, value in payloads.items():
        assert (output / "archives" / name).read_bytes() == value
    assert not (tmp_path / "unrelated.txt").exists()
    bad_records = [dict(record) for record in records]
    bad_records[0]["digest"] = "0" * 64
    with pytest.raises(ValueError, match="SHA-256"):
        module.ingest_public_zip(original, tmp_path / "bad", bad_records)
