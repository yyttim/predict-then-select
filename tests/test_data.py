"""Synthetic data checks; these fixtures are not biological results."""

import importlib.util
import io
import json
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import zarr

from cellselect.data import (
    CHANNELS,
    TARGET_COLUMN,
    PreparedData,
    assign_partitions,
    eligible_metadata,
    load_data,
    normalization,
    sha256_file,
    threshold_from_quarter,
)


def source_frames():
    rows = []
    for i in range(23):
        batch, antibody = (0, "CD16") if i < 20 else ((1, "CD16") if i < 22 else (0, "all"))
        rows.append(
            {
                "global_index": 100 + i if i < 20 else (i - 20 if i < 22 else 200),
                "batch": batch,
                "antibodies": antibody,
                "data_path": f"antibodies_{antibody}/batch_{batch}/slide_replicate_0",
                "slide_replicate": 0,
                "imaging_date": "synthetic",
                "fov_center_x": i // 2,
                "fov_center_y": 0.0,
                "fov_center_z": 0.0,
                "position_in_fov_x_pix": 500 + i % 2,
                "position_in_fov_y_pix": 1000,
            }
        )
    index = pd.DataFrame(rows).set_index("global_index")
    markers = pd.DataFrame({TARGET_COLUMN: np.arange(23, dtype=float) + 1}, index=index.index)
    markers.loc[119, TARGET_COLUMN] = 1e9  # Valid tails must not be trimmed.
    return index, markers


def test_single_antibody_cohort_keeps_tails_and_excludes_other_conditions():
    index, markers = source_frames()
    frame, qc = eligible_metadata(index, markers)
    assert frame.index.tolist() == list(range(100, 120))
    assert frame.target.max() == 1e9
    assert qc["excluded_nonfinite_target"] == 0
    markers.loc[101, TARGET_COLUMN] = np.nan
    frame, qc = eligible_metadata(index, markers)
    assert 101 not in frame.index and qc["excluded_nonfinite_target"] == 1
    external, _ = eligible_metadata(index, markers, batch=1)
    assert external.index.tolist() == [0, 1]
    with pytest.raises(ValueError, match="Only batch 0"):
        assign_partitions(external)


def test_fields_include_slide_identity_and_group_repeat_focus_planes():
    index, markers = source_frames()
    index.loc[101, "fov_center_z"] = 42
    frame, _ = eligible_metadata(index, markers)
    assert frame.loc[100, "fov_id"] == frame.loc[101, "fov_id"]
    index.loc[101, "data_path"] = "antibodies_CD16/batch_0/slide_replicate_1"
    changed, _ = eligible_metadata(index, markers)
    assert changed.loc[100, "fov_id"] != changed.loc[101, "fov_id"]


def test_split_is_outcome_independent_and_threshold_uses_quarter_only():
    index, markers = source_frames()
    frame, _ = eligible_metadata(index, markers)
    first = assign_partitions(frame)
    second = assign_partitions(frame.assign(target=-frame.target))
    pd.testing.assert_series_equal(first["split"], second["split"])
    pd.testing.assert_series_equal(first["train_quarter"], second["train_quarter"])
    assert first.groupby("fov_id")["split"].nunique().max() == 1
    assert first.groupby("fov_id")["train_quarter"].nunique().max() == 1
    assert first.loc[first.train_quarter, "split"].eq("train").all()
    assert first.train_quarter.sum() == 4
    threshold = threshold_from_quarter(first)
    second = first.copy()
    second.loc[~second.train_quarter, "target"] *= 1e6
    assert threshold_from_quarter(second) == threshold
    assert threshold["value"] == np.quantile(
        first.loc[first.train_quarter, "target"], 0.9, method="linear"
    )
    assert threshold["operator"] == ">=" and threshold["source"] == "quarter_training"


def test_budget_rows_and_training_only_normalization():
    index, markers = source_frames()
    metadata, _ = eligible_metadata(index, markers)
    metadata = assign_partitions(metadata).reset_index()
    rng = np.random.default_rng(0)
    images = rng.normal(size=(len(metadata), 4, 28, 28)).astype(np.float32)
    data = PreparedData(images, metadata, {})
    full, quarter = data.rows("train"), data.rows("train", fraction=0.25)
    assert set(quarter) < set(full)
    first = normalization(data, quarter)
    excluded = np.setdiff1d(np.arange(len(metadata)), quarter)
    images[excluded] *= 10000
    metadata.loc[excluded, "target"] *= 10000
    assert normalization(data, quarter) == first
    assert set(first) == {"image_mean", "image_std", "target_mean", "target_std"}
    assert all(isinstance(first[k], list) for k in ["image_mean", "image_std"])
    assert all(isinstance(first[k], float) for k in ["target_mean", "target_std"])
    with pytest.raises(ValueError, match="training rows only"):
        normalization(data, data.rows("val"))
    with pytest.raises(ValueError, match="evaluation pools stay complete"):
        data.rows("test", fraction=0.25)


def test_external_rows_cannot_change_threshold_or_training_normalization():
    index, markers = source_frames()
    main, _ = eligible_metadata(index, markers)
    main = assign_partitions(main)
    threshold = threshold_from_quarter(main)
    external, _ = eligible_metadata(index, markers, batch=1)
    external["split"] = "external"
    external["train_quarter"] = False
    metadata = pd.concat([main, external]).sort_index().reset_index()
    rng = np.random.default_rng(9)
    data = PreparedData(rng.normal(size=(len(metadata), 4, 28, 28)), metadata, {})
    first = normalization(data, data.rows("train", 0.25))
    assert metadata.iloc[data.rows("external")].global_index.tolist() == [0, 1]
    assert not set(data.rows("external")) & set(data.rows("train"))
    data.images[data.rows("external")] *= 1e6
    data.metadata.loc[data.metadata.split == "external", "target"] *= 1e8
    assert threshold_from_quarter(data.metadata) == threshold
    assert normalization(data, data.rows("train", 0.25)) == first
    with pytest.raises(ValueError, match="training rows only"):
        normalization(data, data.rows("external"))
    with pytest.raises(ValueError, match="evaluation pools stay complete"):
        data.rows("external", 0.25)


def preparation_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "prepare_data.py"
    spec = importlib.util.spec_from_file_location("prepare_data_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_archive(tmp_path, duplicate=False, missing_chunk=False):
    root = tmp_path / "source" / "BSCCMNIST"
    root.mkdir(parents=True)
    index, markers = source_frames()
    index.to_csv(root / "BSCCM_index.csv")
    markers.to_csv(root / "BSCCM_surface_markers.csv")
    channels = ["DPC_Bottom", "Brightfield", "DPC_Top", "DPC_Right", "DPC_Left"]
    (root / "BSCCM_global_metadata.json").write_text(
        json.dumps({"name": "BSCCMNIST", "led_array": {"channel_names": channels}})
    )
    store = zarr.open(str(root / "BSCCM_images.zarr"), mode="w")
    for row, (cell, entry) in enumerate(index.iterrows()):
        offset = 0 if duplicate and row == 1 else row
        image = (np.arange(5 * 28 * 28).reshape(5, 28, 28) + offset * 1000).astype(np.uint16)
        store.create_dataset(
            f"{entry.data_path}/led_array/cell_{cell}", data=image, chunks=(1, 96, 96)
        )
    archives = tmp_path / "archives"
    archives.mkdir()
    full = tmp_path / "fixture.tar.gz"
    with tarfile.open(full, "w:gz") as archive:

        def select(member):
            if missing_chunk and member.name.endswith("/cell_100/0.0.0"):
                return None
            return member

        archive.add(root, arcname="BSCCMNIST", filter=select)
    content = full.read_bytes()
    midpoint = len(content) // 2
    records = []
    for number, block in enumerate([content[:midpoint], content[midpoint:]]):
        path = archives / f"chunk{number}.bin"
        path.write_bytes(block)
        records.append({"name": path.name, "bytes": len(block), "sha256": sha256_file(path)})
    return root, archives, records


def test_full_preparation_preserves_native_images_and_locks_data(tmp_path):
    module = preparation_module()
    source, archives, records = source_archive(tmp_path)
    before = {
        name: sha256_file(source / name)
        for name in ["BSCCM_index.csv", "BSCCM_surface_markers.csv", "BSCCM_global_metadata.json"]
    }
    output = tmp_path / "prepared"
    manifest = module.prepare(source, archives, output, archive_specs=records)
    data = load_data(output)
    assert data.images.shape == (22, 4, 28, 28)
    positions = dict(zip(data.metadata.global_index, range(len(data.metadata))))
    assert data.images[positions[100], 0, 0, 0] == 4 * 28 * 28
    assert data.images[positions[101], 0, 0, 0] - data.images[positions[100], 0, 0, 0] == 1000
    assert data.metadata.target.max() == 1e9
    assert data.root == output.resolve() and data.directory == data.root
    assert data.manifest_sha256 == sha256_file(output / "manifest.json")
    assert manifest["input_channels"] == list(CHANNELS)
    assert set(data.rows("train", 0.25)) < set(data.rows("train"))
    assert (tmp_path / "raw_cd16/BSCCMNIST/BSCCM_images.zarr/antibodies_CD16/batch_1").exists()
    assert data.metadata.iloc[data.rows("external")].global_index.tolist() == [0, 1]
    main, _ = eligible_metadata(*source_frames())
    expected_main = assign_partitions(main).reset_index()
    actual_main = data.metadata.loc[data.metadata.batch == 0].reset_index(drop=True)
    pd.testing.assert_frame_equal(actual_main, expected_main, check_dtype=False)
    assert manifest["threshold"] == threshold_from_quarter(expected_main)
    assert all(sha256_file(source / name) == digest for name, digest in before.items())
    data.images._mmap.close()
    with pytest.raises(FileExistsError, match="already locked"):
        module.prepare(source, archives, output, archive_specs=records)
    with (output / "metadata.csv").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="checksum"):
        load_data(output)


def test_prepare_from_chunks_only_recovers_metadata_and_preserves_native_arrays(tmp_path):
    module = preparation_module()
    original, archives, records = source_archive(tmp_path)
    source = tmp_path / "downloaded_metadata"
    output = tmp_path / "prepared"
    manifest = module.prepare(source, archives, output, archive_specs=records)
    for name in module.SOURCE_NAMES:
        assert (source / name).read_bytes() == (original / name).read_bytes()
    assert {path.name for path in source.iterdir()} == set(module.SOURCE_NAMES)
    data = load_data(output)
    assert data.images.shape == (22, 4, 28, 28)
    row = int(np.flatnonzero(data.metadata.global_index.to_numpy() == 100)[0])
    np.testing.assert_array_equal(
        data.images[row, 0], np.arange(4 * 28 * 28, 5 * 28 * 28).reshape(28, 28)
    )
    assert len(data.rows("external")) == 2
    assert manifest["image_qc"]["resized_or_per_image_scaled"] is False
    data.images._mmap.close()


def test_partial_existing_metadata_is_validated_and_never_overwritten(tmp_path):
    module = preparation_module()
    original, archives, records = source_archive(tmp_path)
    source = tmp_path / "partial_source"
    source.mkdir()
    existing = source / "BSCCM_index.csv"
    existing.write_bytes((original / existing.name).read_bytes())
    initial_mtime = existing.stat().st_mtime_ns
    module.prepare(source, archives, tmp_path / "prepared", archive_specs=records)
    assert existing.stat().st_mtime_ns == initial_mtime
    assert all(
        (source / name).read_bytes() == (original / name).read_bytes()
        for name in module.SOURCE_NAMES
    )


def test_conflicting_partial_metadata_fails_without_overwriting_or_filling_files(tmp_path):
    module = preparation_module()
    _, archives, records = source_archive(tmp_path)
    source = tmp_path / "partial_source"
    source.mkdir()
    existing = source / "BSCCM_index.csv"
    existing.write_bytes(b"existing file must remain unchanged")
    with pytest.raises(ValueError, match="Existing metadata differs"):
        module.prepare(source, archives, tmp_path / "prepared", archive_specs=records)
    assert existing.read_bytes() == b"existing file must remain unchanged"
    assert [p.name for p in source.iterdir()] == [existing.name]


def test_archive_hashes_are_checked_before_any_metadata_extraction(tmp_path):
    module = preparation_module()
    _, archives, records = source_archive(tmp_path)
    records[0]["sha256"] = "0" * 64
    source = tmp_path / "missing_source"
    with pytest.raises(ValueError, match="integrity check"):
        module.prepare(source, archives, tmp_path / "prepared", archive_specs=records)
    assert not source.exists()
    assert not (tmp_path / "raw_cd16").exists()


@pytest.mark.parametrize("member_name", ["../BSCCM_index.csv", "C:/BSCCM_index.csv"])
def test_metadata_member_paths_cannot_escape(tmp_path, member_name):
    module = preparation_module()
    archive_path = tmp_path / "bad_metadata.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        content = b"unsafe"
        member = tarfile.TarInfo(member_name)
        member.size = len(content)
        archive.addfile(member, io.BytesIO(content))
    source = tmp_path / "metadata"
    with pytest.raises(ValueError, match="Unsafe archive path"):
        module.ensure_metadata([archive_path], source)
    assert not (tmp_path / "BSCCM_index.csv").exists()


def test_missing_zarr_chunk_is_not_silently_replaced_with_zero(tmp_path):
    module = preparation_module()
    source, archives, records = source_archive(tmp_path, missing_chunk=True)
    output = tmp_path / "prepared"
    with pytest.raises(ValueError, match="Missing image chunk"):
        module.prepare(source, archives, output, archive_specs=records)
    assert not (output / "manifest.json").exists()


def test_duplicate_images_are_reported_before_split_or_cohort_changes(tmp_path):
    module = preparation_module()
    source, archives, records = source_archive(tmp_path, duplicate=True)
    output = tmp_path / "prepared"
    with pytest.raises(ValueError, match="Exact duplicate images"):
        module.prepare(source, archives, output, archive_specs=records)
    assert not (output / "manifest.json").exists()
    assert not (output / "metadata.csv").exists()
    assert json.loads((output / "duplicate_image_audit.json").read_text())["pairs"] == [[100, 101]]


@pytest.mark.parametrize("name", ["../escape", "/absolute", "C:/outside", "nested/../../outside"])
def test_archive_paths_cannot_escape(tmp_path, name):
    with pytest.raises(ValueError, match="Unsafe archive path"):
        preparation_module().safe_destination(tmp_path, name)
