import json
from pathlib import Path
import runpy
import sys
import tarfile

import numpy as np
import pandas as pd
import pytest
import zarr

from cellselect.data import CHANNELS, sha256_file
from cellselect.external_data import load_external, make_external_cohort


def source_frames():
    rows = []
    for cell, batch, antibody, slide in [
        (0, 0, "all", 0),
        (1, 0, "all", 0),
        (10, 1, "all", 0),
        (11, 1, "all", 0),
        (12, 1, "all", 1),
        (13, 1, "all", 1),
        (20, 1, "CD16", 0),
        (21, 1, "unstained", 0),
    ]:
        rows.append(
            {
                "global_index": cell,
                "batch": batch,
                "antibodies": antibody,
                "slide_replicate": slide,
                "imaging_date": f"date_{batch}_{slide}",
                "data_path": f"antibodies_{antibody}/batch_{batch}/slide_replicate_{slide}",
                "fov_center_x": cell % 2,
                "fov_center_y": 0,
            }
        )
    index = pd.DataFrame(rows).set_index("global_index")
    markers = pd.DataFrame(index=index.index)
    markers["CD16_full_model_unmixed"] = np.arange(len(index), dtype=float) + 1
    markers["CD45_full_model_unmixed"] = np.arange(len(index), dtype=float) + 2
    markers["selection_gated_0"] = False
    markers.loc[[0, 1], "selection_gated_0"] = True
    return index, markers


def test_external_eligibility_is_stain_specific_and_keeps_unlabelled_cells():
    index, markers = source_frames()
    cohort, qc = make_external_cohort(index, markers)
    assert cohort.index.tolist() == [10, 11, 12, 13]
    assert (cohort["class_id"] == -1).all()
    assert not cohort["class_label_available"].any()
    assert qc["slides"] == 2 and qc["fovs"] == 4
    assert cohort.loc[10, "fov_id"] != cohort.loc[12, "fov_id"]
    assert (cohort["pool_id"] == cohort["slide_id"]).all()


def test_external_cohort_does_not_make_optional_marker_missingness_a_primary_gate():
    index, markers = source_frames()
    markers.loc[10, "CD16_full_model_unmixed"] = np.nan
    markers.loc[11, "CD45_full_model_unmixed"] = np.nan
    cohort, qc = make_external_cohort(index, markers)
    assert 10 not in cohort.index and 11 in cohort.index
    assert not cohort.loc[11, "CD45_available"]
    assert qc["excluded_nonfinite_CD16"] == 1
    markers.loc[12, "selection_gated_0"] = True
    with pytest.raises(ValueError, match="unexpectedly has published type gates"):
        make_external_cohort(index, markers)


def test_external_preparation_from_existing_archive_is_isolated(tmp_path, monkeypatch):
    source = tmp_path / "raw" / "BSCCMNIST"
    source.mkdir(parents=True)
    index, markers = source_frames()
    index.to_csv(source / "BSCCM_index.csv")
    markers.to_csv(source / "BSCCM_surface_markers.csv")
    names = ["Brightfield", *CHANNELS]
    (source / "BSCCM_global_metadata.json").write_text(
        json.dumps({"name": "BSCCMNIST", "led_array": {"channel_names": names}})
    )
    arrays = zarr.open(str(source / "BSCCM_images.zarr"), mode="w")
    raw = {}
    for cell, row in index.iterrows():
        image = np.arange(5 * 28 * 28, dtype=np.uint16).reshape(5, 28, 28) + cell * 1000
        raw[cell] = image
        arrays.create_dataset(
            f"{row.data_path}/led_array/cell_{cell}", data=image, chunks=(1, 96, 96)
        )
    main = tmp_path / "prepared" / "bsccmnist_cd16_v1"
    main.mkdir(parents=True)
    index.loc[[0, 1]].reset_index().to_csv(main / "metadata.csv", index=False)
    np.save(main / "images.npy", np.stack([raw[0][1:], raw[1][1:]]).astype(np.float32))
    source_files = ["BSCCM_index.csv", "BSCCM_surface_markers.csv", "BSCCM_global_metadata.json"]
    main_manifest = {
        "source_sha256": {name: sha256_file(source / name) for name in source_files},
        "prepared_sha256": {
            name: sha256_file(main / name) for name in ["metadata.csv", "images.npy"]
        },
    }
    (main / "manifest.json").write_text(json.dumps(main_manifest))
    original_main_hash = sha256_file(main / "manifest.json")
    archive_path = tmp_path / "archives" / "BSCCMNIST.tar.gz_chunk00000.bin"
    archive_path.parent.mkdir()
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(source, arcname="BSCCMNIST")
    download = {
        "files": [
            {
                "path": archive_path.name,
                "size": archive_path.stat().st_size,
                "digest": sha256_file(archive_path),
                "digestType": "sha-256",
            }
        ]
    }
    (tmp_path / "download_manifest.json").write_text(json.dumps(download))
    script = Path(__file__).resolve().parents[1] / "scripts" / "prepare_external_data.py"
    monkeypatch.setattr(sys, "argv", [str(script), "--data-root", str(tmp_path), "--workers", "2"])
    runpy.run_path(str(script), run_name="__main__")
    output = tmp_path / "prepared" / "bsccmnist_batch1_all_v1"
    data = load_external(output)
    assert data.images.shape == (4, 4, 28, 28)
    assert data.metadata["global_index"].tolist() == [10, 11, 12, 13]
    np.testing.assert_array_equal(data.images[0], raw[10][1:])
    assert data.manifest["normalization"]["fitted_on_external"] is False
    assert data.manifest["external_model_inference_performed"] is False
    assert len(data.pool_rows(data.metadata.iloc[0].pool_id)) == 2
    assert data.manifest["cohort_qc"]["exact_input_duplicates_with_main"] == 0
    assert sha256_file(main / "manifest.json") == original_main_hash
    for name, expected in main_manifest["source_sha256"].items():
        assert sha256_file(source / name) == expected
    with pytest.raises(FileExistsError, match="already frozen"):
        runpy.run_path(str(script), run_name="__main__")
    changed = data.metadata.copy()
    changed.loc[0, "class_id"] = 0
    changed.to_csv(output / "metadata.csv", index=False)
    with pytest.raises(ValueError, match="checksum"):
        load_external(output)
    with pytest.raises(ValueError, match="type labels"):
        load_external(output, verify_hashes=False)
