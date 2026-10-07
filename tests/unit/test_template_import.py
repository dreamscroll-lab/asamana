"""Installing a map template from an archive: what lands, and what must not.

The rule the whole suite turns on: a refused archive leaves NOTHING behind. `list_templates`
offers any directory holding a ``map.tmj``, so half an installation is a broken map in the world
picker. Every rejection case therefore asserts on the whole templates directory, not the target
name, so an orphaned staging directory beside it fails too.
"""

from __future__ import annotations

import io
import json
import re
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from world import WorldCatalog
from worlds import tiled
from worlds.template_check import check_template
from worlds.connections import derive_connections
from worlds.template_import import MAX_ENTRIES, STAGING_DIRNAME

from interaction.api import create_app

# A real, contract-satisfying template, read before any test redirects TEMPLATES_DIR.
# Zipping what ships is what makes the green path honest — a hand-built fixture that
# satisfied the contract would be a second template to maintain, and one that did not
# could only ever exercise the failure paths.
_SHIPPED = Path(__file__).resolve().parents[2] / "worlds" / "templates" / "metro"


def _client(container, test_config) -> TestClient:
    # The import route is gated off by default; these tests exercise it.
    test_config.web.dev_tools_enabled = True
    return TestClient(create_app(container, test_config, catalog=WorldCatalog(None)))


def _zip(entries: dict[str, bytes], *, compress: int = zipfile.ZIP_STORED) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compress) as archive:
        for name, body in entries.items():
            archive.writestr(name, body)
    return buf.getvalue()


def _shipped_zip(prefix: str = "", extra: dict[str, bytes] | None = None, mutate=None) -> bytes:
    """The shipped template, optionally with `mutate` applied to its map.tmj text."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as archive:
        for path in sorted(_SHIPPED.rglob("*")):
            if not path.is_file():
                continue
            name = prefix + path.relative_to(_SHIPPED).as_posix()
            if mutate is not None and path.name == "map.tmj":
                archive.writestr(name, mutate(path.read_text(encoding="utf-8")))
            else:
                archive.write(path, name)
        for name, body in (extra or {}).items():
            archive.writestr(name, body)
    return buf.getvalue()


@pytest.fixture()
def templates(tmp_path, monkeypatch) -> Path:
    """An empty template directory standing in for ``worlds/templates/``."""
    monkeypatch.setattr(tiled, "TEMPLATES_DIR", tmp_path)
    return tmp_path


def _put(client: TestClient, name: str, body: bytes):
    return client.put(f"/api/templates/{name}/archive", content=body)


def _nothing_landed(templates: Path) -> bool:
    """No template, and no staged bytes left beside one either (see the module docstring).

    The staging PARENT may remain: it is created once and holds nothing.
    """
    for entry in templates.iterdir():
        if entry.name != STAGING_DIRNAME or any(entry.iterdir()):
            return False
    return True


# ---- the green path ------------------------------------------------------


def test_a_shipped_template_survives_a_round_trip(container, test_config, templates) -> None:
    """What ships imports, and what imports satisfies the contract and is offered."""
    response = _put(_client(container, test_config), "imported_city", _shipped_zip())

    assert response.status_code == 200, response.text
    assert response.json()["files"] > 0
    assert (templates / "imported_city" / "map.tmj").is_file()
    assert tiled.list_templates() == ["imported_city"]
    assert check_template("imported_city") == []

    listed = _client(container, test_config).get("/api/templates").json()
    assert [entry["template"] for entry in listed] == ["imported_city"]


def _declared(map_path: Path) -> dict[str, str]:
    """Each location's location_id → the connections it declares, as installed."""
    def walk(layers):
        for layer in layers:
            if layer.get("type") == "group":
                yield from walk(layer.get("layers", []))
            for obj in layer.get("objects", []) or []:
                if obj.get("type") == "location":
                    yield {p["name"]: p["value"] for p in obj["properties"]}

    doc = json.loads(map_path.read_text(encoding="utf-8"))
    return {p["location_id"]: p.get("connections") for p in walk(doc["layers"])}


def test_connections_come_from_the_ground_not_from_the_archive(
    container, test_config, templates
) -> None:
    """The art already says which places are next to each other, so what the archive
    declares is replaced rather than believed — an edge someone typed across a wall
    cannot survive the trip."""
    archive = _shipped_zip(
        mutate=lambda text: re.sub(
            r'("name":"connections",\s*"type":"string",\s*"value":")[^"]*"',
            r'\1nowhere_at_all"',
            text,
        )
    )

    response = _put(_client(container, test_config), "derived", archive)

    assert response.status_code == 200, response.text
    assert response.json()["connections"] > 0
    installed = _declared(templates / "derived" / "map.tmj")
    expected, _ = derive_connections(
        json.loads((templates / "derived" / "map.tmj").read_text(encoding="utf-8")),
        templates / "derived",
    )
    assert installed == {lid: ",".join(nbrs) for lid, nbrs in expected.items()}
    assert "nowhere_at_all" not in (templates / "derived" / "map.tmj").read_text(encoding="utf-8")


def test_a_map_that_declares_no_connections_at_all_installs(
    container, test_config, templates
) -> None:
    """Nobody should have to type an empty property just so the import can fill it in."""
    archive = _shipped_zip(
        mutate=lambda text: re.sub(
            r'\n[ \t]*\{[^{}]*"name":"connections"[^{}]*\}, ', "", text
        )
    )

    assert _put(_client(container, test_config), "bare", archive).status_code == 200

    installed = _declared(templates / "bare" / "map.tmj")
    assert installed and all(value for value in installed.values())


def test_an_enclosing_directory_is_stripped(container, test_config, templates) -> None:
    """Whether the folder or its contents got zipped is an accident of the packing."""
    assert _put(_client(container, test_config), "wrapped", _shipped_zip("my_map/")).status_code == 200

    assert (templates / "wrapped" / "map.tmj").is_file()
    assert not (templates / "wrapped" / "my_map").exists()


def test_a_finder_archive_imports(container, test_config, templates) -> None:
    """Compressing a folder on macOS adds a second top-level tree and AppleDouble
    sidecars carrying allowed suffixes. Neither may reach the template, and neither
    may stop the wrapper being recognised."""
    archive = _shipped_zip(
        "my_map/",
        extra={"__MACOSX/._my_map": b"junk", "__MACOSX/my_map/._map.tmj": b"junk"},
    )

    assert _put(_client(container, test_config), "from_finder", archive).status_code == 200

    assert (templates / "from_finder" / "map.tmj").is_file()
    assert not (templates / "from_finder" / "__MACOSX").exists()
    assert not (templates / "from_finder" / "._map.tmj").exists()


# ---- refusals ------------------------------------------------------------


def test_a_template_failing_the_contract_leaves_nothing(container, test_config, templates) -> None:
    response = _put(_client(container, test_config), "broken", _zip({"map.tmj": b"{}"}))

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["problems"] and all(isinstance(line, str) for line in detail["problems"])
    assert _nothing_landed(templates)


def test_the_verdict_never_names_the_staging_directory(container, test_config, templates) -> None:
    """The author is shown this verdict; where on the server it ran is not theirs."""
    response = _put(_client(container, test_config), "empty", _zip({"readme.json": b"{}"}))

    assert response.status_code == 422
    printed = repr(response.json()["detail"])
    assert ".incoming" not in printed and "tmp" not in printed


@pytest.mark.parametrize("bad", ["Evil", "_example", "9lives", "a.b", "with-hyphen"])
def test_a_name_outside_the_vocabulary_is_refused(
    container, test_config, templates, bad: str
) -> None:
    """The name is a path segment under the templates directory, so the regex that
    shapes it is the traversal defence rather than a spelling preference. A leading
    underscore is refused for a second reason: it marks a reference sample, which an
    import has no business creating."""
    response = _put(_client(container, test_config), bad, _shipped_zip())

    assert response.status_code == 400, response.text
    assert _nothing_landed(templates)


@pytest.mark.parametrize("bad", ["../evil", "a/b", ""])
def test_a_name_shaped_like_a_path_reaches_nothing(
    container, test_config, templates, bad: str
) -> None:
    """These never even resolve to this route; what matters is that they install
    nothing anywhere."""
    assert _put(_client(container, test_config), bad, _shipped_zip()).status_code != 200
    assert _nothing_landed(templates)


def test_an_archive_of_empty_folders_says_what_it_found(
    container, test_config, templates
) -> None:
    """Zipping a folder tree before filling it is an easy mistake, so the refusal has to
    name what it saw — someone looking at several directories in their own archive has
    no reason to read "no files" as being about those."""
    archive = _zip({"map/": b"", "map/characters/": b"", "map/.DS_Store": b"junk"})

    response = _put(_client(container, test_config), "hollow", archive)

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "map.tmj" in detail and "directories" in detail
    assert _nothing_landed(templates)


def test_something_that_is_not_a_zip_is_refused(container, test_config, templates) -> None:
    response = _put(_client(container, test_config), "junk", b"not a zip at all")

    assert response.status_code == 400
    assert _nothing_landed(templates)


def test_an_archive_escaping_its_own_directory_is_refused(
    container, test_config, templates
) -> None:
    for entry in ("../outside.png", "/absolute.png"):
        response = _put(_client(container, test_config), "escaping", _zip({entry: b"x"}))

        assert response.status_code == 400, entry
        assert not (templates.parent / "outside.png").exists()
        assert _nothing_landed(templates)


def test_a_symlink_entry_is_refused(container, test_config, templates) -> None:
    """A link is how an archive names a file it does not carry."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        info = zipfile.ZipInfo("map.tmj")
        info.external_attr = (0o120777 << 16)  # S_IFLNK
        archive.writestr(info, "/etc/passwd")

    response = _put(_client(container, test_config), "linked", buf.getvalue())

    assert response.status_code == 400
    assert _nothing_landed(templates)


def test_too_many_entries_is_refused(container, test_config, templates) -> None:
    archive = _zip({f"art/{i}.png": b"x" for i in range(MAX_ENTRIES + 1)})

    response = _put(_client(container, test_config), "swarm", archive)

    assert response.status_code == 400
    assert _nothing_landed(templates)


def test_an_archive_that_unpacks_past_the_budget_is_refused(
    container, test_config, templates, monkeypatch
) -> None:
    """Enforced while copying, not from the declared size — the header is the
    archive's own claim about itself."""
    monkeypatch.setattr("worlds.template_import.MAX_UNPACKED_BYTES", 1024)
    archive = _zip({"map.tmj": b"\0" * 4096}, compress=zipfile.ZIP_DEFLATED)

    response = _put(_client(container, test_config), "bomb", archive)

    assert response.status_code == 400
    assert _nothing_landed(templates)


def test_a_body_over_the_cap_never_reaches_the_unpacker(
    container, test_config, templates, monkeypatch
) -> None:
    """The cap is on the stream: buffering first is what an oversized body exploits,
    and this process is also running the worlds."""
    calls: list[str] = []
    monkeypatch.setattr("interaction.api.template_import.MAX_ARCHIVE_BYTES", 64)
    monkeypatch.setattr(
        "interaction.api.template_import.install_template",
        lambda *args, **kwargs: calls.append("called"),
    )

    response = _put(_client(container, test_config), "huge", b"\0" * 256)

    assert response.status_code == 400
    assert calls == []


# ---- replacing, and what the gate hides ----------------------------------


def test_a_name_already_taken_is_refused_and_the_installed_map_is_untouched(
    container, test_config, templates
) -> None:
    """Importing never replaces a map. That directory is the authoritative copy and
    may carry edits made in Tiled since it was installed."""
    client = _client(container, test_config)
    assert _put(client, "city", _shipped_zip()).status_code == 200
    (templates / "city" / "edited_in_tiled.png").write_bytes(b"work done since")

    response = _put(client, "city", _shipped_zip())

    assert response.status_code == 409
    assert (templates / "city" / "edited_in_tiled.png").exists()
    assert check_template("city") == []


def test_a_world_name_already_taken_is_refused(container, test_config, templates) -> None:
    """`world_name` is what the map picker shows — the directory name never reaches a
    user — so two maps sharing one are two rows nobody can tell apart."""
    client = _client(container, test_config)
    assert _put(client, "first_city", _shipped_zip()).status_code == 200

    response = _put(client, "second_city", _shipped_zip())

    assert response.status_code == 409
    assert "world_name" in response.json()["detail"]
    assert "first_city" in response.json()["detail"]
    assert not (templates / "second_city").exists()


def test_a_refused_world_name_leaves_no_staging_behind(
    container, test_config, templates
) -> None:
    client = _client(container, test_config)
    assert _put(client, "first_city", _shipped_zip()).status_code == 200
    _put(client, "second_city", _shipped_zip())

    assert [p.name for p in templates.iterdir() if p.name != STAGING_DIRNAME] == ["first_city"]
    assert not any((templates / STAGING_DIRNAME).iterdir())


def test_staging_is_never_offered_as_a_template(templates) -> None:
    """A staged map sits one level deeper than a template, which is what keeps it out
    of the picker while it is being judged."""
    staged = templates / ".incoming" / "tmpXXXX"
    staged.mkdir(parents=True)
    (staged / tiled.MAP_FILENAME).write_text("{}", encoding="utf-8")

    assert tiled.list_templates(include_examples=True) == []


def test_the_route_is_absent_without_dev_tools(container, test_config, templates) -> None:
    """It writes into the source tree; a deployment that is not someone's own machine
    must not carry it at all."""
    test_config.web.dev_tools_enabled = False
    app = create_app(container, test_config, catalog=WorldCatalog(None))

    assert _put(TestClient(app), "city", _shipped_zip()).status_code != 200
    assert _nothing_landed(templates)


# ---- what gets written ---------------------------------------------------


def test_only_asset_files_are_written(container, test_config, templates, monkeypatch) -> None:
    """The destination is the repository working copy, and the contract check only
    inspects the files it knows about — so a stowaway would land unexamined."""
    monkeypatch.setattr("worlds.template_import.check_template_dir", lambda root: [])
    archive = _zip(
        {
            "map.tmj": b"{}",
            "evil.py": b"import os",
            ".DS_Store": b"junk",
            "characters/body.png": b"art",
            "notes.txt": b"hello",
        }
    )

    response = _put(_client(container, test_config), "filtered", archive)

    assert response.status_code == 200
    written = {p.relative_to(templates / "filtered").as_posix()
               for p in (templates / "filtered").rglob("*") if p.is_file()}
    assert written == {"map.tmj", "characters/body.png"}


def test_an_installed_template_is_readable_like_the_ones_beside_it(
    container, test_config, templates
) -> None:
    """Staging is created private; that mode must not ride the rename into place."""
    assert _put(_client(container, test_config), "city", _shipped_zip()).status_code == 200

    assert (templates / "city").stat().st_mode & 0o777 == 0o755


# ---- deleting -------------------------------------------------------------


def test_deleting_a_map_removes_it_from_the_picker(container, test_config, templates) -> None:
    client = _client(container, test_config)
    assert _put(client, "city", _shipped_zip()).status_code == 200

    assert client.delete("/api/templates/city").status_code == 200

    assert not (templates / "city").exists()
    assert tiled.list_templates() == []


def test_deleting_one_map_leaves_the_others(container, test_config, templates) -> None:
    client = _client(container, test_config)
    assert _put(client, "keep_me", _shipped_zip()).status_code == 200

    assert client.delete("/api/templates/gone").status_code == 404

    assert (templates / "keep_me" / "map.tmj").is_file()
    assert check_template("keep_me") == []


@pytest.mark.parametrize("bad", ["Evil", "9lives", "a.b"])
def test_deleting_refuses_a_name_outside_the_vocabulary(
    container, test_config, templates, bad: str
) -> None:
    assert _client(container, test_config).delete(f"/api/templates/{bad}").status_code == 400


def test_deleting_cannot_reach_the_staging_directory(container, test_config, templates) -> None:
    """A map is a directory holding a map.tmj — that is what keeps anything else in
    here, staging included, out of reach."""
    staged = templates / STAGING_DIRNAME / "tmpXXXX"
    staged.mkdir(parents=True)
    (staged / tiled.MAP_FILENAME).write_text("{}", encoding="utf-8")

    assert _client(container, test_config).delete(
        f"/api/templates/{STAGING_DIRNAME}"
    ).status_code in (400, 404, 405)
    assert staged.is_file() or staged.exists()


@pytest.mark.asyncio
async def test_a_world_keeps_its_map_after_that_map_is_deleted(
    container, test_config, templates
) -> None:
    """The whole point of freezing a map into a world at build.

    A map is a thing you build worlds on, not a thing they hold a reference to — so
    removing one must never reach into a world that already exists. Proven with a
    sentinel the template never contained."""
    client = _client(container, test_config)
    assert _put(client, "city", _shipped_zip()).status_code == 200

    world_id = "world-on-a-deleted-map"
    frozen_map = {"tiledversion": "1.10", "layers": [], "sentinel": "FROZEN-AT-BUILD"}
    sentinel = b"\x89PNG\r\n\x1a\n" + b"FROZEN-AT-BUILD"
    await container.snapshot.save_world_config(
        world_id, {"entities": {}, "runtime_context": {"template": "city"}}
    )
    await container.snapshot.save_world_map(world_id, frozen_map)
    await container.snapshot.save_world_asset(world_id, "tilesets/x.png", sentinel)

    assert client.delete("/api/templates/city").status_code == 200

    assert not (templates / "city").exists()
    assert client.get(f"/api/worlds/{world_id}/map").json() == frozen_map
    art = client.get(f"/api/worlds/{world_id}/map/assets/tilesets/x.png")
    assert art.status_code == 200 and art.content == sentinel
