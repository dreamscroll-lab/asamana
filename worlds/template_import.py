"""Install a map template from a zip archive, or refuse it whole.

A template reaches ``worlds/templates/`` either by being copied there by hand or
through here. Both land in the SAME directory — there is no second registry and no
"uploaded templates" namespace, because `list_templates` discovers a map by its
directory being on disk and a second root would be a second answer to "which maps
exist".

What this adds over a copy is that the contract runs BEFORE the archive lands: the
whole thing is unpacked to a staging directory beside the templates, checked by
``template_check``, and only then renamed into place. A template that fails leaves
nothing at all — which matters because `list_templates` will offer any directory
holding a ``map.tmj``, so half an installation is a broken map in the world picker
and a red test suite.

The archive is untrusted input that gets written into the source tree (compose
bind-mounts this directory read-write onto the working copy), so every entry is
inspected before extraction and only known asset suffixes are written.
"""

from __future__ import annotations

import io
import json
import os
import re
import shutil
import stat
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from core.logging import get_logger
from worlds import tiled
from worlds.connections import apply_to_template, derive_connections
from worlds.template_check import check_template_dir

logger = get_logger(__name__)

# The name is a path segment under ``worlds/templates/``, so this regex IS the
# traversal defence — not a style rule. It also settles two things the template
# contract already states: a leading underscore marks a reference sample
# (``tiled.EXAMPLE_PREFIX``), so an import can never create one; and the name is a
# code-level identifier, never shown to a user, so case is not a choice to make.
_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")

# What may be written. A template is a map, its art and its cast manifest; anything
# else in the archive is the packer's leftovers. Narrow because the destination is
# the repository working copy — an unfiltered extract is an arbitrary file write
# into it, and ``template_check`` only inspects the files it knows about, so a
# stowaway would pass unseen.
_ALLOWED_SUFFIXES = frozenset({".tmj", ".json", ".png", ".jpg", ".jpeg", ".webp", ".xml"})

# Where an archive is unpacked while it is being judged. Inside TEMPLATES_DIR so the
# move into place is a same-filesystem rename; two levels down so a staged map.tmj
# is a grandchild and `list_templates` (which looks one level down) cannot see it.
STAGING_DIRNAME = ".incoming"

# Shipped maps are ~20 MB across ~35 files. Three to six times that is room for a
# denser map and still a bound; nothing here is a guess at what an archive "should"
# be, only at what could not possibly be a template.
MAX_ARCHIVE_BYTES = 64 * 1024 * 1024
MAX_UNPACKED_BYTES = 128 * 1024 * 1024
MAX_ENTRIES = 500

# Junk that macOS puts in an archive. It has to go before the wrapper directory is
# worked out, not just before extraction: Finder adds a SECOND top-level tree
# (``__MACOSX/``) beside the folder you compressed, so a wrapper detector that still
# sees it finds two roots, strips neither, and rejects a perfectly good archive for
# having no map.tmj at its root. The suffix filter is no help — an AppleDouble
# sidecar is named ``._map.tmj`` and carries an allowed suffix.


def _is_packaging_junk(name: PurePosixPath) -> bool:
    return (
        name.parts[0] == "__MACOSX"
        or name.name == ".DS_Store"
        or name.name.startswith("._")
    )


@dataclass(frozen=True)
class Installed:
    """What an install put on disk. ``connections`` is derived, so it is reported —
    the archive that went in is not byte-for-byte what came to rest."""

    files: int
    connections: int


class TemplateImportError(ValueError):
    """An archive that will not be installed, with the contract's verdict if it got that far."""

    def __init__(self, message: str, *, problems: list[str] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.problems = problems or []


class TemplateExistsError(TemplateImportError):
    """A map of this directory name is already installed."""


class WorldNameTakenError(TemplateImportError):
    """A map already installed calls itself by this ``world_name``."""


class TemplateMissingError(TemplateImportError):
    """No map is installed under this name."""


def _entry_path(info: zipfile.ZipInfo) -> PurePosixPath | None:
    """The entry's path if it is a regular file that may be written, else None."""
    if info.is_dir():
        return None
    if stat.S_ISLNK(info.external_attr >> 16):
        raise TemplateImportError(f"archive contains a symbolic link: {info.filename}")
    name = PurePosixPath(info.filename)
    if name.is_absolute() or ".." in name.parts:
        raise TemplateImportError(f"archive contains an unsafe path: {info.filename}")
    return name


def _strip_wrapper(entries: list[tuple[zipfile.ZipInfo, PurePosixPath]]) -> list[
    tuple[zipfile.ZipInfo, PurePosixPath]
]:
    """Drop the single enclosing directory, if the archive has one.

    Zipping a template gives ``my_map/map.tmj``; zipping its contents gives
    ``map.tmj``. Both are what someone means by "here is the template", and which
    one you get is an accident of how it was packed.
    """
    roots = {path.parts[0] for _, path in entries}
    if len(roots) != 1 or any(len(path.parts) == 1 for _, path in entries):
        return entries
    return [(info, PurePosixPath(*path.parts[1:])) for info, path in entries]


def _extract(archive: zipfile.ZipFile, entries: list[tuple[zipfile.ZipInfo, PurePosixPath]],
             staging: Path) -> int:
    """Write the allowed entries into *staging*; return how many landed."""
    root = staging.resolve()
    written = 0
    budget = MAX_UNPACKED_BYTES
    for info, path in entries:
        if path.suffix.lower() not in _ALLOWED_SUFFIXES:
            continue
        target = (staging / path).resolve()
        if not target.is_relative_to(root):
            raise TemplateImportError(f"archive contains an unsafe path: {info.filename}")
        target.parent.mkdir(parents=True, exist_ok=True)
        # Copied with a running budget rather than trusting the declared size: the
        # header is the archive's own claim about itself.
        with archive.open(info) as src, target.open("wb") as dst:
            while chunk := src.read(1 << 20):
                budget -= len(chunk)
                if budget < 0:
                    raise TemplateImportError(
                        f"archive unpacks to more than {MAX_UNPACKED_BYTES >> 20} MB"
                    )
                dst.write(chunk)
        written += 1
    return written


def _derive_connections(staging: Path) -> int:
    """Replace whatever the map declared with the adjacency its own ground implies.

    ``connections`` says which places are next to each other, and the art already
    says that — so it is derived rather than checked against what someone typed.
    Two statements about one fact is what let Chang'an declare an edge across a
    palace wall (see ``worlds/connections``); there is now only one.

    An unreadable map is left alone: the contract check next names the real problem
    (a missing tileset, a broken .tmj) rather than this one standing in for it.
    """
    path = staging / tiled.MAP_FILENAME
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        graph, _ = derive_connections(doc, staging)
    except Exception as exc:  # noqa: BLE001 — the real cause is reported by the check
        logger.warning("template_connections_underivable", extra={"error": str(exc)})
        return 0
    apply_to_template(path, graph)
    return sum(len(neighbours) for neighbours in graph.values()) // 2


def _world_name(root: Path) -> str:
    """The ``world_name`` a map declares, or "" if it declares none it can read."""
    try:
        doc = json.loads((root / tiled.MAP_FILENAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    for prop in doc.get("properties", []) or []:
        if prop.get("name") == "world_name":
            return str(prop.get("value") or "").strip()
    return ""


def _refuse_a_taken_world_name(staging: Path) -> None:
    """Two installed maps may not call themselves the same thing.

    ``world_name`` is the name the map picker shows — the directory name never
    reaches a user — so two maps sharing one are two rows nobody can tell apart.

    Deliberately NOT part of the contract check: that judges one directory on its own
    terms and has no business reading its neighbours. This is an invariant of the
    installed SET, and ``tests/unit/test_world_templates`` holds the same line for a
    map copied in by hand.
    """
    incoming = _world_name(staging)
    if not incoming:
        return
    for installed in tiled.list_templates(include_examples=True):
        if _world_name(tiled.template_dir(installed)) == incoming:
            raise WorldNameTakenError(
                f"world_name {incoming!r} is already used by the map {installed!r} — "
                "it is the name the map picker shows, so the two could not be told apart"
            )


def install_template(name: str, archive: bytes) -> Installed:
    """Unpack, derive, check and install a template.

    Raises ``TemplateImportError`` (``TemplateExistsError`` for a name already
    taken) and leaves ``worlds/templates/`` untouched on every failure.
    """
    if not _NAME_RE.match(name):
        raise TemplateImportError(
            f"Invalid template name {name!r}: must start with a lowercase letter and "
            "contain only lowercase letters, digits and underscores"
        )
    if len(archive) > MAX_ARCHIVE_BYTES:
        raise TemplateImportError(f"archive exceeds {MAX_ARCHIVE_BYTES >> 20} MB")

    dest = tiled.TEMPLATES_DIR / name
    if dest.exists():
        raise TemplateExistsError(
            f"a map already lives in {name!r}. Importing never replaces one: that "
            "directory is the authoritative copy and may carry edits made in Tiled "
            "since. Delete it first, or import under another name"
        )

    try:
        zf = zipfile.ZipFile(io.BytesIO(archive))
    except zipfile.BadZipFile as exc:
        raise TemplateImportError(f"not a readable zip archive: {exc}") from exc

    with zf:
        if len(zf.infolist()) > MAX_ENTRIES:
            raise TemplateImportError(f"archive holds more than {MAX_ENTRIES} entries")
        entries = [
            (info, path)
            for info in zf.infolist()
            if (path := _entry_path(info)) is not None and not _is_packaging_junk(path)
        ]
        if not entries:
            # Worth spelling out: someone who zipped an empty folder tree sees several
            # directories in their archive and has no reason to read "no files" as
            # being about them.
            raise TemplateImportError(
                f"archive holds nothing to install — its {len(zf.infolist())} entries "
                "are all directories or packaging leftovers (.DS_Store and the like). "
                "A map is map.tmj plus the art it names; see worlds/templates/README.md"
            )
        entries = _strip_wrapper(entries)

        staging_parent = tiled.TEMPLATES_DIR / STAGING_DIRNAME
        staging_parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(dir=staging_parent))
        # mkdtemp makes it 0700, and that mode would ride the rename into place — an
        # installed template readable only by the server, sitting beside hand-copied
        # ones at 0755. It is browsed, edited in Tiled and committed like any other.
        staging.chmod(0o755)
        try:
            written = _extract(zf, entries, staging)
            connections = _derive_connections(staging)
            problems = check_template_dir(staging)
            if problems:
                logger.warning(
                    "template_import_rejected",
                    extra={"template": name, "problems": len(problems)},
                )
                raise TemplateImportError(
                    f"template {name!r} does not satisfy the template contract",
                    problems=problems,
                )
            _refuse_a_taken_world_name(staging)
            _install(staging, dest)
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    logger.info(
        "template_imported",
        extra={"template": name, "files": written, "connections": connections},
    )
    return Installed(files=written, connections=connections)


def _install(staging: Path, dest: Path) -> None:
    """Move the staged directory into place — one same-filesystem rename, so atomic.

    Nothing has to be moved aside first: a name already taken was refused long before
    this, and a rename onto a directory with anything in it would fail anyway.
    """
    try:
        os.replace(staging, dest)
    except OSError as exc:
        raise TemplateImportError(f"could not install {dest.name!r}: {exc}") from exc


def delete_template(name: str) -> None:
    """Remove an installed map, art and all.

    Worlds already built on it are untouched: each froze its own copy of the map at
    build and restores from its persisted config, never from here (see
    ``world.initializer.restore``). What goes is the ability to build a NEW world on it.

    The map file has to be there, not merely the directory — that is what makes this a
    map rather than any directory whose name happens to match, and it keeps the staging
    directory out of reach.
    """
    if not _NAME_RE.match(name):
        raise TemplateImportError(f"Invalid map name {name!r}")
    root = tiled.template_dir(name)
    if not (root / tiled.MAP_FILENAME).is_file():
        raise TemplateMissingError(f"no map called {name!r}")
    shutil.rmtree(root)
    logger.info("template_deleted", extra={"template": name})
