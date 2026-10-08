"""A world's map and the art it names are frozen together, or not at all.

Freezing the .tmj alone is a half-freeze: the map addresses its tilesets by path,
and those live in a template directory the author keeps editing. A world built
months ago would then render its own geometry through today's art — and a tile
that moved within a sheet renders as garbage, silently, only in old worlds.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from interaction.api.map_artifact import with_asset_urls
from world.initializer import WorldInitializer
from worlds.tiled import CAST_ART_FIELDS, TiledWorldConfig


@pytest.mark.asyncio
async def test_every_image_the_map_names_is_frozen_with_it(container) -> None:
    config = TiledWorldConfig(template="changan_iso")
    render_map = config.render_map()
    assert render_map is not None
    await WorldInitializer(container)._freeze_map_assets("w1", config, render_map)

    named = [str(ts["image"]) for ts in render_map["tilesets"] if ts.get("image")]
    assert len(named) == 15  # or this test proves nothing about a 15-tileset map
    for image in named:
        frozen = await container.snapshot.load_world_asset("w1", image)
        assert frozen is not None, f"{image} was not frozen"
        assert frozen == (config.render_assets_dir() / image).read_bytes()


@pytest.mark.asyncio
async def test_every_file_the_cast_names_is_frozen_and_served(container) -> None:
    """Sheets, descriptors and portraits alike: what the freeze copies and what the route
    addresses come from one field list, so a portrait can't be served from a world that never
    froze it."""
    config = TiledWorldConfig(template="metro")
    manifest = config.render_characters()
    assert manifest is not None
    await WorldInitializer(container)._freeze_character_assets("w3", config)

    named = [
        (key, field, str(entry[field]))
        for key, entry in manifest["atlases"].items()
        for field in CAST_ART_FIELDS
        if entry.get(field)
    ]
    assert any(field == "portrait" for _, field, _ in named)  # or this proves nothing about portraits
    served = with_asset_urls(manifest, "/base")["atlases"]
    for key, field, path in named:
        frozen = await container.snapshot.load_world_asset("w3", path)
        assert frozen == (config.render_assets_dir() / path).read_bytes(), f"{path} was not frozen"
        assert served[key][field] == f"/base/{path}"


@pytest.mark.asyncio
async def test_a_config_without_art_freezes_nothing_and_does_not_raise(container) -> None:
    """A hand-coded substrate carries no art. Building such a world must still work."""
    config = SimpleNamespace(render_assets_dir=lambda: None)
    await WorldInitializer(container)._freeze_map_assets("w2", config, {"tilesets": []})
    assert await container.snapshot.load_world_asset("w2", "anything.png") is None


@pytest.mark.asyncio
async def test_an_unreadable_image_is_skipped_rather_than_failing_the_build(container) -> None:
    """A world that builds is worth more than a perfectly frozen one: a missing
    image falls back to the live template, which is what every pre-freeze world
    already does. The rest of the art must still be frozen."""
    config = TiledWorldConfig(template="changan_iso")
    render_map = {
        "tilesets": [
            {"image": "tilesets/base/ground/ground.png"},
            {"image": "tilesets/does/not/exist.png"},
        ]
    }
    await WorldInitializer(container)._freeze_map_assets("w3", config, render_map)

    assert await container.snapshot.load_world_asset("w3", "tilesets/base/ground/ground.png")
    assert await container.snapshot.load_world_asset("w3", "tilesets/does/not/exist.png") is None


@pytest.mark.asyncio
async def test_a_frozen_asset_cannot_be_written_outside_its_world(container, tmp_path) -> None:
    """`rel_path` originates in a .tmj — authored content, hence untrusted. The file
    provider must confine it rather than trusting each caller to."""
    from providers.snapshot.file import FileSnapshotProvider

    provider = FileSnapshotProvider(str(tmp_path / "snapshots"))
    await provider.save_world_asset("w4", "../../escaped.png", b"nope")
    assert not (tmp_path / "escaped.png").exists()
    assert not list(Path(tmp_path).rglob("escaped.png"))
    assert await provider.load_world_asset("w4", "../../escaped.png") is None
