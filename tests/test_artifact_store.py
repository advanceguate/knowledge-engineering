import pytest

from ke.artifact_store import ArtifactStore


def test_artifact_store_round_trips_json_and_cache(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "workspace")
    artifact = store.write_json(store.source_dir("source-1") / "manifest.json", {"ok": True})

    assert store.read_json(artifact) == {"ok": True}
    assert store.get_cached_json("extract", "key") is None

    store.put_cached_json("extract", "key", {"cached": True})

    assert store.get_cached_json("extract", "key") == {"cached": True}


def test_corrupt_cache_entry_is_treated_as_a_miss(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "workspace")
    cache_path = store.cache_path("extract", "broken")
    cache_path.write_text("{truncated", encoding="utf-8")

    assert store.get_cached_json("extract", "broken") is None


def test_managed_child_symlink_cannot_escape_workspace(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "workspace")
    outside = tmp_path / "outside"
    outside.mkdir()
    (store.ontologies / "ontology-id").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        store.ontology_dir("ontology-id")


def test_nested_symlink_cannot_escape_workspace(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "workspace")
    outside = tmp_path / "outside"
    outside.mkdir()
    bundle = store.source_dir("source-1")
    bundle.mkdir()
    (bundle / "fragments").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        store.write_json(bundle / "fragments" / "escaped.json", {"unsafe": True})

    assert not (outside / "escaped.json").exists()
