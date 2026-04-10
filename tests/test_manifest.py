"""Tests for the asset manifest."""

from pathlib import Path

from sprite_me.storage.manifest import Asset, AssetManifest


def test_add_and_get_asset(tmp_path: Path):
    manifest_path = tmp_path / "manifest.json"
    m = AssetManifest(path=manifest_path)

    asset = Asset(prompt="knight", width=512, height=512, seed=123)
    asset.filename = f"{asset.asset_id}.png"
    m.add(asset)

    retrieved = m.get(asset.asset_id)
    assert retrieved is not None
    assert retrieved.prompt == "knight"
    assert retrieved.seed == 123


def test_manifest_persists_across_instances(tmp_path: Path):
    manifest_path = tmp_path / "manifest.json"
    m1 = AssetManifest(path=manifest_path)

    asset = Asset(prompt="slime")
    asset.filename = f"{asset.asset_id}.png"
    m1.add(asset)

    # New instance loads from disk
    m2 = AssetManifest(path=manifest_path)
    retrieved = m2.get(asset.asset_id)
    assert retrieved is not None
    assert retrieved.prompt == "slime"


def test_list_assets(tmp_path: Path):
    m = AssetManifest(path=tmp_path / "manifest.json")
    for i in range(3):
        a = Asset(prompt=f"item {i}")
        a.filename = f"{a.asset_id}.png"
        m.add(a)

    assets = m.list_assets()
    assert len(assets) == 3


def test_delete_asset(tmp_path: Path):
    m = AssetManifest(path=tmp_path / "manifest.json")
    a = Asset(prompt="test")
    a.filename = f"{a.asset_id}.png"
    m.add(a)

    assert m.delete(a.asset_id) is True
    assert m.get(a.asset_id) is None
    assert m.delete("nonexistent") is False


def test_update_asset(tmp_path: Path):
    m = AssetManifest(path=tmp_path / "manifest.json")
    a = Asset(prompt="original")
    a.filename = f"{a.asset_id}.png"
    m.add(a)

    updated = m.update(a.asset_id, prompt="updated", name="new name")
    assert updated is not None
    assert updated.prompt == "updated"
    assert updated.name == "new name"
