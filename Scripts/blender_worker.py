import bpy
import json
import os
import sys
import traceback
import atexit
import shutil
import tempfile
import gc
import struct
import math
from contextlib import contextmanager
from pathlib import Path
from pathlib import PurePosixPath

_WORKER_DIR = os.path.dirname(os.path.abspath(__file__))
if _WORKER_DIR not in sys.path:
    sys.path.insert(0, _WORKER_DIR)

from versioned_content import (
    DEFAULT_VERSIONED_CONTENT_VIEWS,
    _find_variants,
    choose_version,
    split_versioned_relative,
)

_PENDING_STAGING_DIRS = set()


def _cleanup_pending_staging_dirs():
    """Retry temporary-directory cleanup at worker shutdown."""
    for directory in tuple(_PENDING_STAGING_DIRS):
        try:
            shutil.rmtree(directory)
        except FileNotFoundError:
            _PENDING_STAGING_DIRS.discard(directory)
        except OSError as exc:
            print(
                f"[shutdown] Could not remove temporary staging directory "
                f"{directory}: {exc}",
                flush=True,
            )
        else:
            _PENDING_STAGING_DIRS.discard(directory)


def _cleanup_staging_directory(temporary):
    """Clean staging files without masking the actual import exception."""
    directory = temporary.name

    gc.collect()

    try:
        temporary.cleanup()
    except OSError as exc:
        _PENDING_STAGING_DIRS.add(directory)
        print(
            f"[model] Deferred temporary cleanup for {directory}: {exc}",
            flush=True,
        )


atexit.register(_cleanup_pending_staging_dirs)
atexit.register(DEFAULT_VERSIONED_CONTENT_VIEWS.cleanup)

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
BLEND_SIGNATURE = b"BLENDER"
DEFAULT_MATERIAL = "ValveConvert_Default"


class UnsupportedTextureError(RuntimeError):
    """A VTF file could not be decoded into usable pixels."""

class NonGeometryModelError(RuntimeError):
    """An MDL contains no mesh geometry that can be exported to GLB."""


def clear_scene():
    if bpy.context.scene.objects:
        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.object.delete(use_global=False)

    for action in list(bpy.data.actions):
        bpy.data.actions.remove(action, do_unlink=True)

    for collection_name in (
        "meshes",
        "curves",
        "armatures",
        "materials",
        "images",
        "cameras",
        "lights",
    ):
        collection = getattr(bpy.data, collection_name, None)
        if collection is None:
            continue

        for datablock in list(collection):
            if datablock.users == 0:
                collection.remove(datablock)


def ensure_parent(path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)


def get_relative_input_path(input_root, input_path):
    root = os.path.abspath(input_root)
    path = os.path.abspath(input_path)
    relative = os.path.relpath(path, root)

    if relative == os.pardir or relative.startswith(os.pardir + os.sep):
        raise ValueError(f"Input file is outside the input directory: {input_path}")

    return relative.replace("\\", "/")


def get_texture_cache_path(
    input_root,
    input_path,
    output_root,
    source_game_root=None,
):
    """Return the output materials cache matching the source game's root."""
    if source_game_root is None:
        from SourceIO.library.utils.path_utilities import get_mod_path
        from SourceIO.library.utils.tiny_path import TinyPath

        source_directory = os.path.dirname(os.path.abspath(input_path))
        game_root = os.path.abspath(
            str(get_mod_path(TinyPath(source_directory)))
        )

        expected_game_dirs = (
            "materials",
            "maps",
            "models",
            "elements",
        )

        if (
            not os.path.isdir(game_root)
            or not any(
                os.path.isdir(os.path.join(game_root, name))
                for name in expected_game_dirs
            )
        ):
            game_root = os.path.abspath(input_root)

    else:
        game_root = os.path.abspath(str(source_game_root))

    try:
        relative_game_root = os.path.relpath(
            game_root,
            os.path.abspath(input_root),
        )
    except ValueError:
        relative_game_root = os.curdir

    if (
        relative_game_root == os.pardir
        or relative_game_root.startswith(os.pardir + os.sep)
    ):
        relative_game_root = os.curdir

    return os.path.abspath(
        os.path.join(
            output_root,
            relative_game_root,
            "materials",
        )
    )


def pack_used_images():
    for image in bpy.data.images:
        if image.users <= 0 or not image.has_data:
            continue
        try:
            if image.packed_file is None:
                image.pack()
        except Exception as exc:
            print(f"Warning: unable to pack image {image.name}: {exc}", flush=True)


def reset_sourceio_content_state(content_manager):
    """Reset per-import state without unmounting SourceIO content."""
    content_manager.first_import = None
    content_manager.priority_list = None
    content_manager._cache.clear()
    content_manager._clear_meta_caches()


def preflight_vtf(input_path):
    """Verify SourceIO 5.5.4 can decode a VTF before calling its Blender operator."""
    import numpy as np
    from SourceIO.library.utils.pylib.vtf import load_vtf_texture

    try:
        with open(input_path, "rb") as texture_file:
            vtf_data = texture_file.read()

        pixel_data, width, height, is_float = load_vtf_texture(vtf_data)

        if pixel_data is None or width <= 0 or height <= 0:
            raise ValueError("The native decoder returned no pixel data or invalid dimensions.")

        dtype = np.float32 if is_float else np.uint8
        np.frombuffer(pixel_data, dtype=dtype).reshape(height, width, 4)

    except Exception as exc:
        raise UnsupportedTextureError(
            f"SourceIO 5.5.4 rejected this VTF during preflight; "
            f"the Blender import operator was not called: {exc}"
        ) from exc


def import_vtf(input_root, input_path, output_path):
    ensure_parent(output_path)
    relative_path = get_relative_input_path(input_root, input_path)
    image_names_before = {image.name for image in bpy.data.images}

    bpy.ops.sourceio.vtf(
        directory=input_root,
        files=[{"name": relative_path}],
    )

    new_images = [image for image in bpy.data.images if image.name not in image_names_before]
    image = new_images[-1] if new_images else None

    if image is None:
        wanted_name = os.path.basename(input_path).casefold()
        wanted_stem = os.path.splitext(wanted_name)[0]
        for candidate in reversed(list(bpy.data.images)):
            full_path = candidate.get("full_path", "")
            if os.path.basename(full_path).casefold() == wanted_name:
                image = candidate
                break
            if candidate.name.casefold() == wanted_stem:
                image = candidate
                break

    if image is None:
        raise RuntimeError("SourceIO did not produce a discoverable image.")

    image.filepath_raw = output_path
    image.file_format = "PNG"
    image.save()

    with open(output_path, "rb") as file:
        if file.read(len(PNG_SIGNATURE)) != PNG_SIGNATURE:
            raise RuntimeError("The texture exporter did not produce a valid PNG file.")



def ensure_gltf_exporter_registered():
    """Ensure Blender's bundled glTF exporter is available."""
    try:
        bpy.ops.export_scene.gltf.get_rna_type()
        return
    except Exception:
        pass

    import addon_utils

    errors = []
    for module_name in (
        "io_scene_gltf2",
        "bl_ext.blender_org.io_scene_gltf2",
    ):
        try:
            addon_utils.enable(
                module_name,
                default_set=False,
                persistent=False,
            )
            bpy.ops.export_scene.gltf.get_rna_type()
            return
        except Exception as exc:
            errors.append(f"{module_name}: {exc}")

    raise RuntimeError(
        "Blender's glTF 2.0 exporter is unavailable: "
        + " | ".join(errors)
    )


def _iter_material_image_nodes(node_tree, visited=None):
    """Find image nodes in a material tree, including nested node groups."""
    if node_tree is None:
        return

    if visited is None:
        visited = set()

    tree_id = node_tree.as_pointer()
    if tree_id in visited:
        return
    visited.add(tree_id)

    for node in node_tree.nodes:
        if node.type == "TEX_IMAGE" and node.image is not None:
            yield node, node.image
        elif node.type == "GROUP" and node.node_tree is not None:
            yield from _iter_material_image_nodes(
                node.node_tree, visited
            )


def _choose_material_image(entries, include, exclude=()):
    for node, image, label in entries:
        name = label.casefold()
        if (
            any(token in name for token in include)
            and not any(token in name for token in exclude)
        ):
            return image
    return None


def prepare_materials_for_gltf(mesh_objects):
    """
    Convert SourceIO materials to glTF-compatible Principled BSDF materials.
    Preserve recognizable base-color, normal, roughness, metallic and
    emissive images when SourceIO's material graph exposes them.
    """
    materials = {
        slot.material
        for obj in mesh_objects
        for slot in obj.material_slots
        if slot.material is not None
    }

    for material in materials:
        material.use_nodes = True
        entries = []
        packed_ids = set()

        for node, image in _iter_material_image_nodes(
            material.node_tree
        ):
            label = " ".join(
                str(value or "")
                for value in (node.name, node.label, image.name)
            )
            entries.append((node, image, label))

            image_id = image.as_pointer()
            if image_id not in packed_ids:
                packed_ids.add(image_id)
                try:
                    if image.has_data and image.packed_file is None:
                        image.pack()
                except Exception as exc:
                    print(
                        f"[model] Warning: could not pack "
                        f"{image.name!r}: {exc}",
                        flush=True,
                    )

        excluded = (
            "normal", "bump", "rough", "metal", "spec",
            "emissive", "emission", "alpha", "mask",
            "envmap", "environment",
        )
        base_image = _choose_material_image(
            entries,
            (
                "basetexture", "base_texture", "base color",
                "basecolor", "diffuse", "albedo", "color", "colour",
            ),
            excluded,
        )
        if base_image is None:
            base_image = next(
                (
                    image for _, image, label in entries
                    if not any(token in label.casefold() for token in excluded)
                ),
                entries[0][1] if entries else None,
            )

        normal_image = _choose_material_image(
            entries, ("normal", "normalmap", "normal_map", "bump")
        )
        roughness_image = _choose_material_image(
            entries, ("roughness", "rough")
        )
        metallic_image = _choose_material_image(
            entries, ("metallic", "metalness")
        )
        emissive_image = _choose_material_image(
            entries, ("emissive", "emission", "selfillum")
        )

        tree = material.node_tree
        tree.nodes.clear()

        output = tree.nodes.new("ShaderNodeOutputMaterial")
        output.location = (500, 0)

        principled = tree.nodes.new("ShaderNodeBsdfPrincipled")
        principled.location = (200, 0)
        tree.links.new(
            principled.outputs["BSDF"],
            output.inputs["Surface"],
        )

        def connect_image(image, socket_name, y, non_color=False):
            if image is None:
                return None

            if non_color:
                try:
                    image.colorspace_settings.name = "Non-Color"
                except Exception:
                    pass

            texture = tree.nodes.new("ShaderNodeTexImage")
            texture.image = image
            texture.label = image.name
            texture.location = (-500, y)

            socket = principled.inputs.get(socket_name)
            if socket is not None:
                tree.links.new(texture.outputs["Color"], socket)

            return texture

        connect_image(base_image, "Base Color", 200)
        connect_image(roughness_image, "Roughness", -100, True)
        connect_image(metallic_image, "Metallic", -300, True)

        if normal_image is not None:
            normal_texture = connect_image(
                normal_image, "__unused__", -500, True
            )
            if normal_texture is not None:
                normal_map = tree.nodes.new("ShaderNodeNormalMap")
                normal_map.location = (-120, -500)
                tree.links.new(
                    normal_texture.outputs["Color"],
                    normal_map.inputs["Color"],
                )
                normal_socket = principled.inputs.get("Normal")
                if normal_socket is not None:
                    tree.links.new(
                        normal_map.outputs["Normal"],
                        normal_socket,
                    )

        if emissive_image is not None:
            emission_socket = (
                "Emission Color"
                if principled.inputs.get("Emission Color")
                else "Emission"
            )
            connect_image(emissive_image, emission_socket, -700)

        alpha_socket = principled.inputs.get("Alpha")
        if alpha_socket is not None and base_image is not None:
            alpha_texture = next(
                (
                    node for node in tree.nodes
                    if node.type == "TEX_IMAGE"
                    and node.image == base_image
                ),
                None,
            )
            if (
                alpha_texture is not None
                and alpha_texture.outputs.get("Alpha") is not None
            ):
                tree.links.new(
                    alpha_texture.outputs["Alpha"],
                    alpha_socket,
                )

    for image in bpy.data.images:
        try:
            if image.has_data and image.packed_file is None:
                image.pack()
        except Exception as exc:
            print(
                f"[model] Warning: could not pack "
                f"{image.name!r}: {exc}",
                flush=True,
            )


def validate_glb_model(output_path, expected_animation_count=None):
    """Validate GLB 2.0 structure, geometry, and embedded image references."""
    with open(output_path, "rb") as file:
        data = file.read()

    if len(data) < 28:
        raise RuntimeError("GLB is too short to be valid.")

    magic, version, declared_length = struct.unpack_from(
        "<4sII", data, 0
    )
    if magic != b"glTF" or version != 2:
        raise RuntimeError("Output is not a GLB 2.0 file.")
    if declared_length != len(data):
        raise RuntimeError("GLB declared length does not match its file size.")

    json_length, json_type = struct.unpack_from("<II", data, 12)
    if json_type != 0x4E4F534A:
        raise RuntimeError("GLB JSON chunk is missing.")
    json_end = 20 + json_length

    if json_length == 0 or json_end + 8 > len(data):
        raise RuntimeError("GLB JSON chunk has an invalid length.")

    try:
        document = json.loads(
            data[20:json_end].rstrip(b" \t\r\n\x00").decode("utf-8")
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Invalid GLB JSON chunk: {exc}") from exc

    animations = document.get("animations", [])

    if not isinstance(animations, list):
        raise RuntimeError("GLB animation list is invalid.")

    if (
        expected_animation_count is not None
        and len(animations) != expected_animation_count
    ):
        raise RuntimeError(
            f"Expected {expected_animation_count} animation clip(s), "
            f"but the GLB contains {len(animations)}."
        )

    meshes = document.get("meshes", [])
    if not isinstance(meshes, list) or not any(
        isinstance(mesh, dict)
        and isinstance(mesh.get("primitives"), list)
        and mesh["primitives"]
        for mesh in meshes
    ):
        raise RuntimeError("GLB contains no mesh primitives.")

    binary_length, binary_type = struct.unpack_from(
        "<II", data, json_end
    )
    binary_start = json_end + 8

    if binary_type != 0x004E4942:
        raise RuntimeError("GLB binary chunk is missing.")
    if (
        binary_length == 0
        or binary_start + binary_length > len(data)
    ):
        raise RuntimeError("GLB binary chunk has an invalid length.")

    for image in document.get("images", []):
        uri = image.get("uri")
        embedded = (
            isinstance(image.get("bufferView"), int)
            or (
                isinstance(uri, str)
                and uri.casefold().startswith("data:")
            )
        )
        if not embedded:
            raise RuntimeError(
                "GLB references an external image instead of embedding it."
            )

    for buffer in document.get("buffers", []):
        uri = buffer.get("uri")
        if (
            isinstance(uri, str)
            and not uri.casefold().startswith("data:")
        ):
            raise RuntimeError(
                "GLB references an external buffer file."
            )

    print(
        f"[model] GLB validated: meshes={len(meshes)}, "
        f"animations={len(animations)}, "
        f"materials={len(document.get('materials', []))}, "
        f"embedded_images={len(document.get('images', []))}",
        flush=True,
    )


def resolve_dependency_file(game_root, relative_path, model_version):
    """Resolve one dependency on demand; never walk unrelated archive paths."""
    relative = PurePosixPath(
        str(relative_path).replace("\\", "/")
    )

    if relative.is_absolute() or ".." in relative.parts:
        return None

    logical_path = Path(game_root).joinpath(*relative.parts)

    if logical_path.is_file():
        return logical_path

    variants = _find_variants(logical_path)

    if not variants:
        return None

    selected = choose_version(
        variants.keys(),
        model_version,
    )

    return variants[selected] if selected is not None else None


def _dependency_candidates(game_root, relative_path, desired_version):
    """Return dependency candidates from newest eligible version downward."""
    relative = PurePosixPath(
        str(relative_path).replace("\\", "/")
    )

    if relative.is_absolute() or ".." in relative.parts:
        return []

    logical_path = Path(game_root).joinpath(*relative.parts)

    if logical_path.is_file():
        return [(None, logical_path)]

    variants = _find_variants(logical_path)
    if not variants:
        return []

    eligible = [
        version
        for version in variants
        if desired_version is None or version <= desired_version
    ]

    return [
        (version, variants[version])
        for version in sorted(eligible, reverse=True)
    ]


def _load_mdl_for_pair_check(mdl_path):
    """Parse the MDL with the same SourceIO parser used for model import."""
    from SourceIO.library.utils import FileBuffer
    from SourceIO.library.utils.tiny_path import TinyPath

    with FileBuffer(TinyPath(str(mdl_path))) as buffer:
        ident, mdl_format_version = buffer.read_fmt("4sI")
        buffer.seek(0)

        if ident != b"IDST":
            raise ValueError(
                f"Unexpected MDL identifier {ident!r}: {mdl_path}"
            )

        if mdl_format_version == 36:
            from SourceIO.library.models.mdl.v36 import MdlV36
            parser = MdlV36
        elif 37 <= mdl_format_version <= 44:
            from SourceIO.library.models.mdl.v44 import MdlV44
            parser = MdlV44
        elif 45 <= mdl_format_version <= 49:
            from SourceIO.library.models.mdl.v49 import MdlV49
            parser = MdlV49
        elif mdl_format_version == 52:
            from SourceIO.library.models.mdl.v52.mdl_file import MdlV52
            parser = MdlV52
        else:
            return mdl_format_version, None

        mdl = parser.from_buffer(buffer)

    return mdl_format_version, mdl


def _check_vvd_vtx_compatibility(mdl, vvd_path, vtx_path):
    """
    Check whether the candidate VTX indices fit the VVD's model vertex ranges.

    This mirrors the indexing operation that otherwise raises IndexError
    inside SourceIO's MDL importer.
    """
    from SourceIO.blender_bindings.models.common import merge_meshes
    from SourceIO.library.models.vtx import open_vtx
    from SourceIO.library.models.vvd import Vvd
    from SourceIO.library.utils import FileBuffer
    from SourceIO.library.utils.tiny_path import TinyPath

    with FileBuffer(TinyPath(str(vvd_path))) as buffer:
        vvd = Vvd.from_buffer(buffer)

    with FileBuffer(TinyPath(str(vtx_path))) as buffer:
        vtx = open_vtx(buffer)

    all_vertices = vvd.lod_data[0]

    if len(vtx.body_parts) != len(mdl.body_parts):
        return False, (
            f"VTX body-part count {len(vtx.body_parts)} does not match "
            f"MDL count {len(mdl.body_parts)}"
        )

    checked_models = 0

    for vtx_body_part, body_part in zip(
        vtx.body_parts,
        mdl.body_parts,
    ):
        if len(vtx_body_part.models) != len(body_part.models):
            return False, (
                f"VTX model count does not match MDL body part "
                f"{body_part.name!r}"
            )

        for vtx_model, model in zip(
            vtx_body_part.models,
            body_part.models,
        ):
            if model.vertex_count == 0:
                continue

            if not vtx_model.model_lods:
                return False, (
                    f"VTX model {model.name!r} has no LOD data"
                )

            start = model.vertex_offset
            end = start + model.vertex_count
            model_vertices = all_vertices[start:end]

            if len(model_vertices) != model.vertex_count:
                return False, (
                    f"MDL model {model.name!r} requests vertices "
                    f"[{start}:{end}], but the VVD does not contain "
                    f"that range"
                )

            try:
                vtx_vertices, _, _ = merge_meshes(
                    model,
                    vtx_model.model_lods[0],
                )
            except Exception as exc:
                return False, (
                    f"Could not build VTX indices for {model.name!r}: "
                    f"{type(exc).__name__}: {exc}"
                )

            if vtx_vertices:
                lowest = min(vtx_vertices)
                highest = max(vtx_vertices)

                if lowest < 0 or highest >= len(model_vertices):
                    return False, (
                        f"VTX index range {lowest}..{highest} exceeds "
                        f"the VVD model vertex range 0.."
                        f"{len(model_vertices) - 1} for {model.name!r}"
                    )

            checked_models += 1

    return True, f"checked {checked_models} model(s)"


def _select_compatible_model_dependencies(
    physical_model_path,
    game_root,
    model_relative_to_game,
    model_version,
):
    """Select the preferred eligible VVD/VTX pair that passes index checks."""
    mdl_format_version, mdl = _load_mdl_for_pair_check(
        physical_model_path
    )

    if mdl is not None:
        has_mesh_geometry = any(
            getattr(model, "vertex_count", 0) > 0
            for body_part in getattr(mdl, "body_parts", ())
            for model in getattr(body_part, "models", ())
        )

        if not has_mesh_geometry:
            raise NonGeometryModelError(
                "MDL contains no mesh geometry; it may be an "
                "animation/sequence-only asset and cannot produce "
                f"a mesh GLB: {physical_model_path}"
            )

    model_path = PurePosixPath(model_relative_to_game)

    vvd_candidates = _dependency_candidates(
        game_root,
        model_path.with_suffix(".vvd"),
        model_version,
    )

    vtx_suffixes = (
        ".dx90.vtx",
        ".dx12.vtx",
        ".vtx",
        ".dx11.vtx",
        ".dx80.vtx",
        ".dx70.vtx",
    )

    vtx_candidates = []

    for suffix in vtx_suffixes:
        relative = model_path.with_suffix(suffix)

        for version, path in _dependency_candidates(
            game_root,
            relative,
            model_version,
        ):
            vtx_candidates.append((suffix, version, path))

    if not vtx_candidates:
        raise FileNotFoundError(
            f"No eligible VTX dependency found for {physical_model_path} "
            f"at model version {model_version}"
        )

    if mdl_format_version == 36:
        selected_vtx = vtx_candidates[0]
        selected_vvd = vvd_candidates[0] if vvd_candidates else None
        return selected_vvd, selected_vtx

    if not vvd_candidates:
        raise FileNotFoundError(
            f"No eligible VVD dependency found for {physical_model_path} "
            f"at model version {model_version}"
        )

    if mdl is None:
        return vvd_candidates[0], vtx_candidates[0]

    attempted = []

    for vtx_candidate in vtx_candidates:
        vtx_suffix, vtx_version, vtx_path = vtx_candidate

        for vvd_version, vvd_path in vvd_candidates:
            try:
                compatible, reason = _check_vvd_vtx_compatibility(
                    mdl,
                    vvd_path,
                    vtx_path,
                )
            except Exception as exc:
                compatible = False
                reason = f"{type(exc).__name__}: {exc}"

            if compatible:
                return (
                    (vvd_version, vvd_path),
                    (vtx_suffix, vtx_version, vtx_path),
                )

            attempted.append(
                f"VTX {vtx_path} + VVD {vvd_path}: {reason}"
            )

    details = "\n".join(f"  - {line}" for line in attempted[:8])

    raise RuntimeError(
        f"No compatible VVD/VTX pair was found for {physical_model_path} "
        f"at model version {model_version}.\n"
        f"Tested {len(attempted)} candidate pair(s).\n{details}"
    )


def get_versioned_game_context(input_root, input_path):
    """Return the physical input, game root, logical path, and requested version."""
    archive_root = Path(input_root).resolve()
    source_path = Path(input_path).resolve()

    try:
        relative_input = source_path.relative_to(archive_root).as_posix()
    except ValueError as exc:
        raise ValueError(
            f"Input file is outside the input directory: {input_path}"
        ) from exc

    logical_relative, desired_version = split_versioned_relative(
        relative_input
    )
    logical_parts = PurePosixPath(logical_relative).parts

    if not logical_parts:
        raise RuntimeError(
            f"Cannot determine game root for asset: {input_path}"
        )

    content_dir_names = {
        "materials",
        "models",
        "maps",
        "elements",
    }

    if (
        logical_parts[0].casefold() in content_dir_names
        and any(
            (archive_root / name).is_dir()
            for name in content_dir_names
        )
    ):
        game_root = archive_root
        relative_to_game = PurePosixPath(logical_relative).as_posix()
    else:
        if len(logical_parts) < 2:
            raise RuntimeError(
                f"Cannot determine game root for asset: {input_path}"
            )

        game_root = archive_root / logical_parts[0]
        relative_to_game = PurePosixPath(
            *logical_parts[1:]
        ).as_posix()

    if not game_root.is_dir():
        raise RuntimeError(
            f"Expected game content root does not exist: {game_root}"
        )

    return (
        source_path,
        game_root,
        relative_to_game,
        desired_version,
    )


def create_version_aware_provider(game_root, model_version):
    """Build a SourceIO 5.5.4 provider for one game root/version context."""
    from SourceIO.library.shared.app_id import SteamAppId
    from SourceIO.library.shared.content_manager.provider import ContentProvider
    from SourceIO.library.utils import FileBuffer
    from SourceIO.library.utils.tiny_path import TinyPath

    root = Path(game_root).resolve()

    class VersionAwareLooseFilesProvider(ContentProvider):
        def __init__(self):
            self.game_root = root
            self.model_version = model_version
            super().__init__(TinyPath(str(self.game_root)))

        @property
        def root(self):
            return TinyPath(str(self.game_root))

        @property
        def name(self):
            version_label = (
                "latest"
                if self.model_version is None
                else str(self.model_version)
            )
            return (
                f"VCTVersioned-{self.game_root.name}"
                f"-v{version_label}"
            )

        @property
        def steam_id(self):
            return SteamAppId.UNKNOWN

        def _resolve(self, filepath):
            path = TinyPath(filepath)

            if path.is_absolute():
                absolute = Path(str(path))

                try:
                    relative = absolute.relative_to(
                        self.game_root
                    ).as_posix()
                except ValueError:
                    return None

            else:
                relative = str(path).replace("\\", "/")

            pure = PurePosixPath(relative)

            if pure.is_absolute() or ".." in pure.parts:
                return None

            return resolve_dependency_file(
                self.game_root,
                pure.as_posix(),
                self.model_version,
            )

        def check(self, filepath):
            resolved = self._resolve(filepath)
            return resolved is not None and resolved.is_file()

        def find_file(self, filepath):
            resolved = self._resolve(filepath)

            if resolved is None or not resolved.is_file():
                return None

            return FileBuffer(TinyPath(str(resolved)))

        def get_relative_path(self, filepath):
            try:
                relative = Path(str(filepath)).relative_to(
                    self.game_root
                )
            except (ValueError, OSError):
                return None

            return TinyPath(relative.as_posix())

        def get_provider_from_path(self, filepath):
            return self if self.check(filepath) else None

        def get_steamid_from_asset(self, asset_path):
            return self.steam_id if self.check(asset_path) else None

        def glob(self, pattern):
            return iter(())

    return VersionAwareLooseFilesProvider()


def _mounted_resource_collections(scene):
    """Return the legacy and/or grouped SourceIO mounted-resource collections."""
    result = []
    props = getattr(scene, "sourceio_props", None)

    candidates = [
        (
            getattr(props, "mounted_resources", None)
            if props is not None
            else None
        ),
        getattr(scene, "mounted_resources", None),
    ]

    for collection in candidates:
        if (
            collection is not None
            and all(collection is not existing for existing in result)
        ):
            result.append(collection)

    return result


def _clear_blender_collection(collection):
    for index in reversed(range(len(collection))):
        collection.remove(index)


@contextmanager
def isolated_versioned_content(content_manager, provider):
    """Run one SourceIO import with only this version-aware provider mounted."""
    saved_collections = []

    for collection in _mounted_resource_collections(bpy.context.scene):
        records = [
            (item.path, item.name, item.hash)
            for item in collection
        ]
        saved_collections.append((collection, records))

    try:
        for collection, _ in saved_collections:
            _clear_blender_collection(collection)

        content_manager.clean()
        content_manager.add_child(provider)
        reset_sourceio_content_state(content_manager)

        yield

    finally:
        content_manager.clean()
        reset_sourceio_content_state(content_manager)

        for collection, records in saved_collections:
            _clear_blender_collection(collection)

            for path, name, resource_hash in records:
                item = collection.add()
                item.path = path
                item.name = name
                item.hash = resource_hash


def stage_versioned_model(input_root, input_path):
    """Stage one MDL with a structurally compatible VVD/VTX pair."""
    (
        source_path,
        game_root,
        model_relative_to_game,
        model_version,
    ) = get_versioned_game_context(input_root, input_path)

    if source_path.is_file():
        physical_model_path = source_path
    else:
        physical_model_path = resolve_dependency_file(
            game_root,
            model_relative_to_game,
            model_version,
        )

    if physical_model_path is None or not physical_model_path.is_file():
        raise FileNotFoundError(f"Cannot resolve MDL: {input_path}")

    selected_vvd, selected_vtx = _select_compatible_model_dependencies(
        physical_model_path,
        game_root,
        model_relative_to_game,
        model_version,
    )

    temporary = tempfile.TemporaryDirectory(prefix="vct-model-")

    try:
        staged_root = Path(temporary.name)
        logical_model = PurePosixPath(model_relative_to_game)

        staged_model = staged_root.joinpath(*logical_model.parts)
        staged_model.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(physical_model_path, staged_model)

        if selected_vvd is not None:
            _, physical_vvd = selected_vvd
            logical_vvd = logical_model.with_suffix(".vvd")
            staged_vvd = staged_root.joinpath(*logical_vvd.parts)
            staged_vvd.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(physical_vvd, staged_vvd)

        vtx_suffix, vtx_version, physical_vtx = selected_vtx
        logical_vtx = logical_model.with_suffix(vtx_suffix)
        staged_vtx = staged_root.joinpath(*logical_vtx.parts)
        staged_vtx.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(physical_vtx, staged_vtx)

        for suffix in (".phy", ".ani", ".vvc"):
            relative = logical_model.with_suffix(suffix).as_posix()
            physical = resolve_dependency_file(
                game_root,
                relative,
                model_version,
            )

            if physical is None or not physical.is_file():
                continue

            staged_optional = staged_root.joinpath(
                *PurePosixPath(relative).parts
            )
            staged_optional.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(physical, staged_optional)

        vvd_description = (
            f"{selected_vvd[1]} (variant {selected_vvd[0]})"
            if selected_vvd is not None
            else "not required by MDL format"
        )

        print(f"[model] Selected MDL: {physical_model_path}", flush=True)
        print(f"[model] Selected VVD: {vvd_description}", flush=True)
        print(
            f"[model] Selected VTX: {physical_vtx} "
            f"(variant {vtx_version}; suffix {vtx_suffix})",
            flush=True,
        )

        return temporary, staged_model, game_root, model_version

    except Exception:
        _cleanup_staging_directory(temporary)
        raise

     
def load_source1_animation_data(mdl, mdl_buffer, content_manager, model_path):
    """Decode Source 1 animations directly through SourceIO's data parser."""
    from SourceIO.library.models.mdl.load_animations import (
        load_animations_from_mdl,
    )
    from SourceIO.library.utils.tiny_path import TinyPath

    animations = []
    visited = set()
    descriptor_count = 0

    def parse_mdl(buffer):
        buffer.seek(0)
        ident, version = buffer.read_fmt("4sI")
        buffer.seek(0)

        if ident != b"IDST":
            return version, None

        if version == 36:
            from SourceIO.library.models.mdl.v36 import MdlV36
            parser = MdlV36
        elif 37 <= version <= 44:
            from SourceIO.library.models.mdl.v44 import MdlV44
            parser = MdlV44
        elif 45 <= version <= 49:
            from SourceIO.library.models.mdl.v49 import MdlV49
            parser = MdlV49
        else:
            return version, None

        return version, parser.from_buffer(buffer)

    def visit(current_mdl, current_buffer, current_path, depth=0):
        nonlocal descriptor_count

        if depth > 16:
            print(
                f"[model] Animation include depth limit reached at "
                f"{current_path}",
                flush=True,
            )
            return

        logical_path = PurePosixPath(
            str(current_path).replace("\\", "/")
        )
        key = logical_path.as_posix().casefold()

        if key in visited:
            return
        visited.add(key)

        descriptors = getattr(current_mdl, "anim_descs", ()) or ()
        descriptor_count += sum(
            1
            for descriptor in descriptors
            if getattr(descriptor, "frame_count", 0) > 0
        )

        try:
            current_buffer.seek(0)
            decoded = load_animations_from_mdl(
                current_mdl,
                current_buffer,
                content_manager,
                TinyPath(logical_path.as_posix()),
            )

            for animation in decoded or ():
                if (
                    getattr(animation, "frame_count", 0) > 0
                    and getattr(animation, "frames", None)
                ):
                    animations.append((logical_path.as_posix(), animation))

        except Exception as exc:
            print(
                f"[model] Could not decode animations in "
                f"{logical_path}: {type(exc).__name__}: {exc}",
                flush=True,
            )

        for include in getattr(current_mdl, "include_models", ()) or ():
            include_name = str(include).replace("\\", "/")
            include_path = PurePosixPath(include_name)

            candidates = [include_path]
            relative_candidate = logical_path.parent / include_path.name
            if relative_candidate not in candidates:
                candidates.append(relative_candidate)

            include_buffer = None
            resolved_path = None

            for candidate in candidates:
                try:
                    include_buffer = content_manager.find_file(
                        TinyPath(candidate.as_posix())
                    )
                except Exception:
                    include_buffer = None

                if include_buffer is not None:
                    resolved_path = candidate
                    break

            if include_buffer is None:
                print(
                    f"[model] Animation include not found: {include_name}",
                    flush=True,
                )
                continue

            try:
                include_buffer.seek(0)
                _, include_mdl = parse_mdl(include_buffer)

                if include_mdl is None:
                    print(
                        f"[model] Unsupported animation include: "
                        f"{resolved_path}",
                        flush=True,
                    )
                    continue

                visit(
                    include_mdl,
                    include_buffer,
                    resolved_path,
                    depth + 1,
                )

            except Exception as exc:
                print(
                    f"[model] Could not read animation include "
                    f"{resolved_path}: {type(exc).__name__}: {exc}",
                    flush=True,
                )
            finally:
                close = getattr(include_buffer, "close", None)
                if callable(close):
                    close()

    visit(mdl, mdl_buffer, model_path)
    return animations, descriptor_count


def _finite_number(value, fallback=0.0):
    """Replace NaN and infinity with a finite fallback value."""
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return fallback

    return value if math.isfinite(value) else fallback


def sanitize_scene_for_gltf(mesh_objects, export_objects):
    """Remove non-finite numeric values from data exported to glTF."""
    for obj in export_objects:
        for prop, defaults in (
            ("location", (0.0, 0.0, 0.0)),
            ("scale", (1.0, 1.0, 1.0)),
        ):
            values = tuple(getattr(obj, prop))
            sanitized = tuple(
                _finite_number(value, defaults[i])
                for i, value in enumerate(values)
            )
            if values != sanitized:
                setattr(obj, prop, sanitized)

        if obj.rotation_mode == "QUATERNION":
            values = tuple(obj.rotation_quaternion)
            defaults = (1.0, 0.0, 0.0, 0.0)
            sanitized = tuple(
                _finite_number(value, defaults[i])
                for i, value in enumerate(values)
            )
            if values != sanitized:
                obj.rotation_quaternion = sanitized
        elif obj.rotation_mode != "AXIS_ANGLE":
            values = tuple(obj.rotation_euler)
            sanitized = tuple(_finite_number(value) for value in values)
            if values != sanitized:
                obj.rotation_euler = sanitized

    for obj in mesh_objects:
        mesh = obj.data

        for vertex in mesh.vertices:
            values = tuple(vertex.co)
            sanitized = tuple(_finite_number(value) for value in values)
            if values != sanitized:
                vertex.co = sanitized

        shape_keys = mesh.shape_keys
        if shape_keys is not None:
            for key_block in shape_keys.key_blocks:
                for point in key_block.data:
                    values = tuple(point.co)
                    sanitized = tuple(
                        _finite_number(value) for value in values
                    )
                    if values != sanitized:
                        point.co = sanitized

        for attribute in mesh.attributes:
            for item in attribute.data:
                for prop in ("value", "vector", "color"):
                    try:
                        value = getattr(item, prop)
                        if isinstance(value, (int, float)):
                            sanitized = _finite_number(value)
                            if sanitized != value:
                                setattr(item, prop, sanitized)
                        else:
                            values = tuple(value)
                            sanitized = tuple(
                                _finite_number(component)
                                for component in values
                            )
                            if values != sanitized:
                                setattr(item, prop, sanitized)
                    except (AttributeError, TypeError, ValueError):
                        pass

    materials = {
        slot.material
        for obj in mesh_objects
        for slot in obj.material_slots
        if slot.material is not None
    }

    for material in materials:
        if not material.use_nodes or material.node_tree is None:
            continue

        for node in material.node_tree.nodes:
            for socket in node.inputs:
                try:
                    value = socket.default_value
                    if isinstance(value, (int, float)):
                        sanitized = _finite_number(value)
                        if sanitized != value:
                            socket.default_value = sanitized
                    else:
                        values = tuple(value)
                        sanitized = tuple(
                            _finite_number(component)
                            for component in values
                        )
                        if values != sanitized:
                            socket.default_value = sanitized
                except (AttributeError, TypeError, ValueError):
                    pass


def create_source1_animation_actions(armature_objects, animation_sources, scale=1.0):
    """Build Blender 5-compatible slotted Actions from decoded SourceIO data."""
    from math import radians
    from mathutils import Euler, Matrix, Quaternion, Vector
    from SourceIO.blender_bindings.utils.bpy_utils import ActionCurveFactory

    armatures = [
        obj for obj in armature_objects
        if obj.type == "ARMATURE" and obj.data is not None
    ]
    if not armatures:
        return []

    root_rotation = Euler(
        (0.0, 0.0, radians(-90.0))
    ).to_matrix().to_4x4()

    created_actions = []

    for model_path, animation in animation_sources:
        frame_count = int(getattr(animation, "frame_count", 0))
        bone_frames = getattr(animation, "frames", {}) or {}

        if frame_count <= 0 or not bone_frames:
            continue

        armature = max(
            armatures,
            key=lambda obj: sum(
                1 for name in bone_frames
                if obj.data.bones.get(name) is not None
            ),
        )

        matching_bones = [
            name for name in bone_frames
            if armature.data.bones.get(name) is not None
        ]
        if not matching_bones:
            print(
                f"[model] Skipping animation {animation.name!r} from "
                f"{model_path}: no matching bones in the imported rig",
                flush=True,
            )
            continue

        action_name = (
            f"{PurePosixPath(model_path).stem}_{animation.name}"
        )
        action = bpy.data.actions.new(name=action_name)
        action["source_fps"] = float(
            getattr(animation, "fps", 0.0) or 30.0
        )

        scene = bpy.context.scene
        gltf_frame_rate = (
            float(scene.render.fps)
            * float(scene.render.fps_base)
        )
        source_fps = float(action["source_fps"])
        frame_scale = (
            gltf_frame_rate / source_fps
            if source_fps > 0.0
            else 1.0
        )

        factory = ActionCurveFactory(action, armature)
        curve_count = 0

        for bone_name, frames in bone_frames.items():
            bone = armature.data.bones.get(bone_name)
            if bone is None or frames is None:
                continue

            count = min(frame_count, len(frames))
            if count <= 0:
                continue

            parent = bone.parent
            parent_rest = (
                parent.matrix_local.copy()
                if parent is not None
                else Matrix.Identity(4)
            )
            rest_inverse = bone.matrix_local.inverted()

            positions = [[0.0] * count for _ in range(3)]
            rotations = [[0.0] * count for _ in range(4)]
            last_quaternion = None
            last_location = None

            for frame_index in range(count):
                frame_data = frames[frame_index]
                position = Vector(
                    tuple(_finite_number(v) for v in frame_data["pos"])
                ) * scale

                x, y, z, w = (
                    _finite_number(v, 0.0)
                    for v in frame_data["rot"]
                )
                quaternion = Quaternion((w, x, y, z))

                if quaternion.dot(quaternion) < 1e-12:
                    quaternion = Quaternion()
                else:
                    quaternion.normalize()

                if getattr(animation, "is_delta", False):
                    if parent is None:
                        position = root_rotation @ position

                    location = rest_inverse @ position

                else:
                    animated_local = (
                        Matrix.Translation(position)
                        @ quaternion.to_matrix().to_4x4()
                    )
                    armature_space = parent_rest @ animated_local

                    if parent is None:
                        armature_space = root_rotation @ armature_space

                    basis = rest_inverse @ armature_space
                    location, quaternion, _ = basis.decompose()

                if not all(math.isfinite(float(v)) for v in location):
                    location = (
                        last_location.copy()
                        if last_location is not None
                        else Vector((0.0, 0.0, 0.0))
                    )

                if not all(math.isfinite(float(v)) for v in quaternion):
                    quaternion = (
                        last_quaternion.copy()
                        if last_quaternion is not None
                        else Quaternion()
                    )

                if (
                    last_quaternion is not None
                    and quaternion.dot(last_quaternion) < 0.0
                ):
                    quaternion = -quaternion

                last_quaternion = quaternion.copy()
                last_location = location.copy()

                for axis in range(3):
                    positions[axis][frame_index] = float(location[axis])
                for axis in range(4):
                    rotations[axis][frame_index] = float(quaternion[axis])

            group = factory.new_group(bone_name)
            data_path = f'pose.bones["{bone_name}"].'

            for property_name, values in (
                ("location", positions),
                ("rotation_quaternion", rotations),
            ):
                for axis, channel_values in enumerate(values):
                    curve = factory.new_fcurve(
                        data_path=data_path + property_name,
                        index=axis,
                        group=group,
                    )
                    curve.keyframe_points.add(count=count)

                    coordinates = []
                    for frame_index, value in enumerate(channel_values):
                        coordinates.extend(
                            (
                                float(frame_index) * frame_scale,
                                float(value),
                            )
                        )

                    curve.keyframe_points.foreach_set(
                        "co", coordinates
                    )

                    for point in curve.keyframe_points:
                        point.interpolation = "LINEAR"

                    curve.update()
                    curve_count += 1

        if curve_count == 0:
            bpy.data.actions.remove(action, do_unlink=True)
            continue

        created_actions.append(action)

    print(
        f"[model] Created {len(created_actions)} animation action(s) "
        f"from decoded SourceIO data",
        flush=True,
    )
    return created_actions


def prepare_animation_tracks_for_gltf(armature_objects, actions):
    """Put each imported SourceIO Action on its own NLA track."""
    if not actions:
        print("[model] No animation actions were imported.", flush=True)
        return 0

    armature_objects = [
        obj for obj in armature_objects
        if obj.type == "ARMATURE"
    ]

    if not armature_objects:
        raise RuntimeError(
            f"SourceIO imported {len(actions)} animation action(s), "
            "but no armature is available to export them."
        )

    tracks_created = 0

    for action in actions:
        slots = list(getattr(action, "slots", ()))
        target_armature = None
        target_slot = None

        for armature in armature_objects:
            slot = next(
                (
                    candidate
                    for candidate in slots
                    if getattr(candidate, "name", None) == armature.name
                ),
                None,
            )

            if slot is not None:
                target_armature = armature
                target_slot = slot
                break

        if target_armature is None and len(armature_objects) == 1:
            target_armature = armature_objects[0]
            if len(slots) == 1:
                target_slot = slots[0]

        if target_armature is None:
            raise RuntimeError(
                f"Could not determine which armature owns animation "
                f"action {action.name!r}."
            )

        if slots and target_slot is None:
            raise RuntimeError(
                f"Could not resolve a Blender 5 Action Slot for "
                f"animation action {action.name!r}."
            )

        animation_data = target_armature.animation_data_create()
        track = animation_data.nla_tracks.new()
        track.name = action.name

        strip = track.strips.new(
            action.name,
            int(bpy.context.scene.frame_start),
            action,
        )

        if target_slot is not None:
            if not hasattr(strip, "action_slot"):
                raise RuntimeError(
                    "This Blender build cannot assign Action Slots "
                    "to NLA strips; animation export cannot be guaranteed."
                )
            strip.action_slot = target_slot

        strip.blend_type = "REPLACE"
        strip.extrapolation = "NOTHING"

        strip.scale = 1.0

        track.mute = False
        tracks_created += 1

    for armature in armature_objects:
        if armature.animation_data is not None:
            armature.animation_data.action = None

    print(
        f"[model] Prepared {tracks_created} NLA animation track(s).",
        flush=True,
    )

    return tracks_created


def import_model(input_root, input_path, output_path, output_root):
    from SourceIO.library.shared.content_manager import ContentManager

    ensure_parent(output_path)

    temporary, staged_model, game_root, model_version = (
        stage_versioned_model(input_root, input_path)
    )

    try:
        source_mdl_version, source_mdl = _load_mdl_for_pair_check(
            staged_model
        )

        animation_sources = []
        animation_descriptor_count = 0

        resolved_input_path = str(staged_model)
        model_directory = str(staged_model.parent)

        texture_cache_path = get_texture_cache_path(
            input_root,
            input_path,
            output_root,
            source_game_root=game_root,
        )
        os.makedirs(texture_cache_path, exist_ok=True)

        scene = bpy.context.scene

        if hasattr(scene, "sourceio_props"):
            scene.sourceio_props.TextureCachePath = texture_cache_path
        elif hasattr(scene, "TextureCachePath"):
            scene.TextureCachePath = texture_cache_path
        else:
            raise RuntimeError(
                "SourceIO registered neither Scene.sourceio_props nor "
                "the legacy Scene.TextureCachePath property."
            )

        print(
            f"[model] SourceIO texture cache: {texture_cache_path}",
            flush=True,
        )

        content_manager = ContentManager()
        versioned_provider = create_version_aware_provider(
            game_root,
            model_version,
        )

        with isolated_versioned_content(
            content_manager,
            versioned_provider,
        ):
            operator_result = bpy.ops.sourceio.mdl(
                directory=model_directory,
                files=[
                    {"name": os.path.basename(resolved_input_path)}
                ],
                discover_resources=False,
                import_textures=True,
                import_animations=False,
                import_include_animations=False,
            )

            if (
                "CANCELLED" not in operator_result
                and source_mdl is not None
                and 36 <= source_mdl_version <= 49
            ):
                from SourceIO.library.utils import FileBuffer
                from SourceIO.library.utils.tiny_path import TinyPath

                logical_model_path = PurePosixPath(
                    staged_model.relative_to(
                        Path(temporary.name)
                    ).as_posix()
                )

                with FileBuffer(
                    TinyPath(str(staged_model))
                ) as animation_buffer:
                    (
                        animation_sources,
                        animation_descriptor_count,
                    ) = load_source1_animation_data(
                        source_mdl,
                        animation_buffer,
                        content_manager,
                        logical_model_path,
                    )

    finally:
        _cleanup_staging_directory(temporary)

    if "CANCELLED" in operator_result:
        raise RuntimeError(
            f"SourceIO 5.5.4 cancelled the MDL import for {input_path}. "
            "Check the model's required VVD/VTX companion files and the "
            "SourceIO error immediately preceding this message."
        )

    print(
        f"[model] MDL animation descriptors={animation_descriptor_count}; "
        f"decoded animation clips={len(animation_sources)}",
        flush=True,
    )

    mesh_objects = [
        obj
        for obj in bpy.context.scene.objects
        if obj.type == "MESH"
        and obj.data is not None
        and len(obj.data.vertices) > 0
        and len(obj.data.polygons) > 0
    ]

    if not mesh_objects:
        raise RuntimeError(
            f"SourceIO completed the MDL operator but imported no usable "
            f"mesh geometry for {input_path}. Check its MDL version and "
            "matching VVD/VTX companion files."
        )

    fallback = bpy.data.materials.get(DEFAULT_MATERIAL)

    if fallback is None:
        fallback = bpy.data.materials.new(DEFAULT_MATERIAL)

    for obj in mesh_objects:
        if len(obj.data.materials) == 0:
            obj.data.materials.append(fallback)

    prepare_materials_for_gltf(mesh_objects)
    ensure_gltf_exporter_registered()

    export_objects = list(mesh_objects)

    for obj in mesh_objects:
        parent = obj.parent
        while parent is not None:
            if (
                parent.type == "ARMATURE"
                and parent not in export_objects
            ):
                export_objects.append(parent)
            parent = parent.parent

        for modifier in obj.modifiers:
            if (
                modifier.type == "ARMATURE"
                and modifier.object is not None
                and modifier.object not in export_objects
            ):
                export_objects.append(modifier.object)

        armature_objects = [
        obj for obj in export_objects
        if obj.type == "ARMATURE"
    ]

    if (
        animation_descriptor_count > 0
        and armature_objects
        and not animation_sources
    ):
        raise RuntimeError(
            f"SourceIO found {animation_descriptor_count} non-empty "
            "animation descriptor(s), but its data parser decoded no "
            "animation clips. Refusing to export a silently static GLB."
        )

    imported_actions = create_source1_animation_actions(
        armature_objects,
        animation_sources,
        scale=1.0,
    )

    if (
        animation_descriptor_count > 0
        and armature_objects
        and not imported_actions
    ):
        raise RuntimeError(
            "Animation data was decoded, but no clips could be bound "
            "to the imported armature. Refusing to export a silently "
            "static GLB."
        )

    expected_animation_count = prepare_animation_tracks_for_gltf(
        armature_objects,
        imported_actions,
    )

    sanitize_scene_for_gltf(mesh_objects, export_objects)
    bpy.ops.object.select_all(action="DESELECT")
    for obj in export_objects:
        obj.select_set(True)

    bpy.context.view_layer.objects.active = mesh_objects[0]

    export_result = bpy.ops.export_scene.gltf(
        filepath=output_path,
        check_existing=False,
        export_format="GLB",
        export_image_format="AUTO",
        export_materials="EXPORT",
        export_animations=True,
        export_animation_mode="ACTIONS",
        export_merge_animation="ACTION",
        export_frame_range=False,
        export_frame_step=1,
        export_force_sampling=False,
        export_cameras=False,
        export_lights=False,
        export_yup=True,
        export_apply=False,
        export_skins=True,
        export_morph=True,
        use_selection=True,
    )

    if "CANCELLED" in export_result:
        raise RuntimeError("Blender's glTF exporter cancelled the model export.")

    validate_glb_model(
        output_path,
        expected_animation_count=expected_animation_count,
    )


def node_tree_has_image_texture(node_tree, visited=None):
    """Return True if a material node tree contains a usable image texture."""
    if node_tree is None:
        return False

    if visited is None:
        visited = set()

    pointer = node_tree.as_pointer()
    if pointer in visited:
        return False
    visited.add(pointer)

    for node in node_tree.nodes:
        if node.type == "TEX_IMAGE":
            image = node.image
            if image is not None and image.has_data:
                return True

        if node.type == "GROUP" and node_tree_has_image_texture(node.node_tree, visited):
            return True

    return False


def validate_bsp_scene():
    """Require usable BSP geometry, but never reject a map for missing textures."""
    mesh_objects = [
        obj for obj in bpy.context.scene.objects
        if obj.type == "MESH"
        and obj.data is not None
        and len(obj.data.vertices) > 0
        and len(obj.data.polygons) > 0
    ]

    if not mesh_objects:
        raise RuntimeError("BSP import produced no usable mesh geometry.")

    materials = {
        slot.material
        for obj in mesh_objects
        for slot in obj.material_slots
        if slot.material is not None
    }

    textured_materials = [
        material for material in materials
        if material.use_nodes
        and node_tree_has_image_texture(material.node_tree)
    ]

    print(
        f"[map] Geometry OK: {len(mesh_objects)} mesh objects; "
        f"{len(materials)} assigned materials; "
        f"{len(textured_materials)} textured materials.",
        flush=True,
    )

    if not textured_materials:
        print(
            "[map] WARNING: no usable image-textured materials were found. "
            "Saving the map anyway so its geometry is preserved.",
            flush=True,
        )


def import_bsp(input_root, input_path, output_path, output_root):
    from SourceIO.library.shared.content_manager import ContentManager

    if os.path.splitext(input_path)[1].casefold() != ".bsp":
        raise ValueError(
            "Only an explicit .bsp source may create a .blend output."
        )

    ensure_parent(output_path)

    relative_path = get_relative_input_path(
        input_root,
        input_path,
    )

    resolved_root, resolved_filename, map_version = (
        DEFAULT_VERSIONED_CONTENT_VIEWS.resolve(
            input_root,
            relative_path,
        )
    )

    resolved_input_path = os.path.join(
        str(resolved_root),
        *PurePosixPath(resolved_filename).parts,
    )

    _, game_root, _, _ = get_versioned_game_context(
        input_root,
        input_path,
    )

    texture_cache_path = get_texture_cache_path(
        input_root,
        input_path,
        output_root,
        source_game_root=game_root,
    )
    os.makedirs(texture_cache_path, exist_ok=True)

    scene = bpy.context.scene

    if hasattr(scene, "sourceio_props"):
        scene.sourceio_props.TextureCachePath = texture_cache_path
    elif hasattr(scene, "TextureCachePath"):
        scene.TextureCachePath = texture_cache_path
    else:
        raise RuntimeError(
            "SourceIO registered neither Scene.sourceio_props nor "
            "the legacy Scene.TextureCachePath property."
        )

    content_manager = ContentManager()
    versioned_provider = create_version_aware_provider(
        game_root,
        map_version,
    )

    with isolated_versioned_content(
        content_manager,
        versioned_provider,
    ):
        operator_result = bpy.ops.sourceio.bsp(
            filepath=resolved_input_path,
            discover_resources=False,
            import_textures=True,
            use_bvlg=False,
        )

    if "CANCELLED" in operator_result:
        raise RuntimeError(
            f"SourceIO 5.5.4 cancelled the BSP import for {input_path}."
        )

    validate_bsp_scene()
    pack_used_images()

    bpy.ops.wm.save_as_mainfile(
        filepath=output_path,
        check_existing=False,
    )

    with open(output_path, "rb") as file:
        if file.read(len(BLEND_SIGNATURE)) != BLEND_SIGNATURE:
            raise RuntimeError(
                "BSP conversion did not produce a valid Blender project file."
            )



def validate_output(asset_type, output_path):
    if not os.path.isfile(output_path) or os.path.getsize(output_path) == 0:
        raise RuntimeError(
            "Converter returned without creating a non-empty output file."
        )

    extension = os.path.splitext(output_path)[1].casefold()

    if extension == ".blend" and asset_type != "map":
        raise RuntimeError(
            "A .blend output is only allowed for BSP map jobs."
        )
    if asset_type == "map" and extension != ".blend":
        raise RuntimeError("A BSP job must produce a .blend file.")
    if asset_type == "texture" and extension != ".png":
        raise RuntimeError("A texture job must produce a .png file.")
    if asset_type == "model":
        if extension != ".glb":
            raise RuntimeError("A model job must produce a .glb file.")
        validate_glb_model(output_path)


def remove_failed_outputs(asset_type, output_path):
    try:
        if os.path.isfile(output_path):
            os.remove(output_path)
    except OSError as exc:
        print(
            f"Warning: could not remove failed output "
            f"{output_path}: {exc}",
            flush=True,
        )


def ensure_sourceio_registered():
    import importlib
    import importlib.util
    import os
    import sys
    import bpy

    legacy_scene_property_names = (
        "TextureCachePath",
        "use_bvlg",
        "use_instances",
        "import_materials",
        "import_physics",
        "replace_entity",
        "mounted_resources",
        "mounted_resources_index",
    )

    if (
        hasattr(bpy.types.Scene, "sourceio_props")
        or all(
            hasattr(bpy.types.Scene, name)
            for name in legacy_scene_property_names
        )
    ):
        print(
            "[startup] SourceIO Scene properties are already registered.",
            flush=True,
        )
        return

    addon_dirs = list(bpy.utils.script_paths(subdir="addons"))

    user_addons_dir = bpy.utils.user_resource(
        "SCRIPTS",
        path="addons",
    )
    if user_addons_dir:
        addon_dirs.insert(0, user_addons_dir)

    searched_dirs = []
    addon_root = None

    for addons_dir in dict.fromkeys(addon_dirs):
        searched_dirs.append(addons_dir)

        if not os.path.isdir(addons_dir):
            continue

        try:
            entries = list(os.scandir(addons_dir))
        except OSError as exc:
            print(
                f"[startup] Cannot scan {addons_dir}: {exc}",
                flush=True,
            )
            continue

        for entry in entries:
            try:
                if not entry.is_dir():
                    continue
            except OSError:
                continue

            if "sourceio" not in entry.name.casefold():
                continue

            candidate = entry.path

            required_files = (
                os.path.join(candidate, "__init__.py"),
                os.path.join(
                    candidate,
                    "blender_bindings",
                    "bindings.py",
                ),
                os.path.join(
                    candidate,
                    "blender_bindings",
                    "attributes",
                    "__init__.py",
                ),
            )

            if all(os.path.isfile(path) for path in required_files):
                addon_root = candidate
                break

        if addon_root is not None:
            break

    if addon_root is None:
        raise RuntimeError(
            "Could not locate a complete SourceIO add-on installation. "
            "Searched: " + "; ".join(searched_dirs)
        )

    init_path = os.path.join(addon_root, "__init__.py")
    attributes_path = os.path.join(
        addon_root,
        "blender_bindings",
        "attributes",
        "__init__.py",
    )

    print(
        f"[startup] SourceIO add-on directory: {addon_root}",
        flush=True,
    )

    stale_modules = [
        name
        for name in tuple(sys.modules)
        if name == "SourceIO" or name.startswith("SourceIO.")
    ]

    for name in sorted(stale_modules, key=len, reverse=True):
        sys.modules.pop(name, None)

    importlib.invalidate_caches()

    package_spec = importlib.util.spec_from_file_location(
        "SourceIO",
        init_path,
        submodule_search_locations=[addon_root],
    )

    if package_spec is None or package_spec.loader is None:
        raise RuntimeError(
            f"Could not create a SourceIO package loader for: {init_path}"
        )

    sourceio_module = importlib.util.module_from_spec(package_spec)
    sys.modules["SourceIO"] = sourceio_module

    try:
        package_spec.loader.exec_module(sourceio_module)
    except Exception:
        for name in tuple(sys.modules):
            if name == "SourceIO" or name.startswith("SourceIO."):
                sys.modules.pop(name, None)
        raise

    print(
        f"[startup] SourceIO package loaded: "
        f"{sourceio_module.__file__}",
        flush=True,
    )

    attributes_module = importlib.import_module(
        "SourceIO.blender_bindings.attributes"
    )
    bindings_module = importlib.import_module(
        "SourceIO.blender_bindings.bindings"
    )

    sourceio_version = getattr(
        sourceio_module,
        "bl_info",
        {},
    ).get("version", "unknown")

    print(
        f"[startup] SourceIO version: {sourceio_version}",
        flush=True,
    )
    print(
        f"[startup] SourceIO attributes: {attributes_module.__file__}",
        flush=True,
    )
    print(
        f"[startup] SourceIO bindings: {bindings_module.__file__}",
        flush=True,
    )

    has_grouped_properties = hasattr(
        bpy.types.Scene,
        "sourceio_props",
    )
    has_legacy_properties = all(
        hasattr(bpy.types.Scene, name)
        for name in legacy_scene_property_names
    )

    if not (has_grouped_properties or has_legacy_properties):
        print(
            "[startup] Registering SourceIO through its official "
            "bindings registration function...",
            flush=True,
        )

        bindings_module.register()

    has_grouped_properties = hasattr(
        bpy.types.Scene,
        "sourceio_props",
    )
    has_legacy_properties = all(
        hasattr(bpy.types.Scene, name)
        for name in legacy_scene_property_names
    )

    if not (has_grouped_properties or has_legacy_properties):
        raise RuntimeError(
            "SourceIO registration returned without creating a recognized "
            "Scene property API. "
            f"Version: {sourceio_version}; "
            f"bindings: {bindings_module.__file__}; "
            f"attributes: {attributes_module.__file__}"
        )

    property_api = (
        "Scene.sourceio_props"
        if has_grouped_properties
        else "legacy direct Scene properties"
    )

    print(
        f"[startup] SourceIO registration verified. API: {property_api}",
        flush=True,
    )


def main():
    if "--" not in sys.argv:
        raise RuntimeError("Expected a manifest path after '--'.")

    manifest_path = sys.argv[sys.argv.index("--") + 1]
    with open(manifest_path, "r", encoding="utf-8") as file:
        manifest = json.load(file)

    input_root = os.path.abspath(manifest["input_root"])
    output_root = os.path.abspath(manifest["output_root"])
    jobs = manifest["jobs"]
    sourceio_registered = False

    if jobs:
        ensure_sourceio_registered()
        sourceio_registered = True

    results = []

    for job in jobs:
        asset_type = job.get("type", job.get("Type"))
        input_path = job.get("input_path", job.get("InputPath"))
        output_path = job.get("output_path", job.get("OutputPath"))
        print(f"[{asset_type}] {input_path}", flush=True)

        try:
            extension = os.path.splitext(input_path)[1].casefold()
            output_extension = os.path.splitext(output_path)[1].casefold()
            if output_extension == ".blend" and not (asset_type == "map" and extension == ".bsp"):
                raise ValueError("Only an explicit .bsp input may create a .blend output.")
            if asset_type == "map" and extension != ".bsp":
                raise ValueError("A map conversion job must point to an explicit .bsp file.")

            clear_scene()
            if asset_type == "texture":
                preflight_vtf(input_path)
                import_vtf(input_root, input_path, output_path)
            elif asset_type == "model":
                import_model(input_root, input_path, output_path, output_root)
            elif asset_type == "map":
                import_bsp(input_root, input_path, output_path, output_root)
            else:
                raise ValueError(f"Unsupported conversion job type: {asset_type}. VMT-to-BLEND is disabled.")

            validate_output(asset_type, output_path)
            output_paths = [output_path]

            results.append({
                "InputPath": input_path,
                "OutputPaths": output_paths,
                "Status": 0,
                "Message": None,
            })

        except NonGeometryModelError as exc:
            print(
                f"[model-skip] {input_path}: {exc}",
                flush=True,
            )
            remove_failed_outputs(asset_type, output_path)
            results.append({
                "InputPath": input_path,
                "OutputPaths": [],
                "Status": 2,
                "Message": str(exc),
            })

        except UnsupportedTextureError as exc:
            print(
                f"[texture-skip] {input_path}: {exc}",
                flush=True,
            )
            remove_failed_outputs(asset_type, output_path)
            results.append({
                "InputPath": input_path,
                "OutputPaths": [],
                "Status": 3,
                "Message": str(exc),
            })

        except Exception as exc:
            traceback.print_exc()
            remove_failed_outputs(asset_type, output_path)
            results.append({
                "InputPath": input_path,
                "OutputPaths": [],
                "Status": 3,
                "Message": str(exc),
            })

    if sourceio_registered:
        try:
            from SourceIO.blender_bindings.bindings import unregister_custom_icon

            unregister_custom_icon()
        except Exception as exc:
            print(
                f"[shutdown] Warning: could not release SourceIO icon previews: {exc}",
                flush=True,
            )

    result_path = os.path.splitext(manifest_path)[0] + ".results.json"
    with open(result_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)


if __name__ == "__main__":
    main()