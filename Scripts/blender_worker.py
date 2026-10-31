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
import warnings
from contextlib import contextmanager
from pathlib import Path
from pathlib import PurePosixPath

warnings.filterwarnings("ignore", category=DeprecationWarning)

_WORKER_DIR = os.path.dirname(os.path.abspath(__file__))
if _WORKER_DIR not in sys.path:
    sys.path.insert(0, _WORKER_DIR)

from versioned_content import (
    DEFAULT_VERSIONED_CONTENT_VIEWS,
    dependency_candidates as _metadata_dependency_candidates,
    get_selection_context,
    resolve_dependency_path as _metadata_resolve_dependency_path,
    split_versioned_relative,
)

_PENDING_STAGING_DIRS = set()
_VCT_VTX_INDEX_RECOVERY_ACTIVE = False


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


def _collect_material_image_entries(material):
    entries = []

    for node, image in _iter_material_image_nodes(material.node_tree):
        label = " ".join(
            str(value or "")
            for value in (node.name, node.label, image.name)
        )
        entries.append((node, image, label))

    return entries


def _images_for_material_role(entries, role):
    if role == "base_color":
        include = (
            "basetexture", "base_texture", "base texture",
            "basecolor", "base color", "albedo", "diffuse",
            "texture2", "detail", "compress", "stretch",
        )
        exclude = (
            "normal", "bump", "rough", "metal", "spec",
            "emissive", "emission", "alpha", "mask",
            "envmap", "environment", "lightwarp", "phong",
            "selfillum",
        )
    elif role == "normal":
        include = (
            "normal", "bumpmap", "bump_map", "bumpcompress",
            "bumpstretch",
        )
        exclude = ("envmap", "phong", "selfillum", "mask")
    else:
        return []

    images = []
    seen = set()

    for node, image, _ in entries:
        node_label = " ".join(
            str(value or "") for value in (node.name, node.label)
        ).casefold()

        label = (
            node_label
            if "$" in node_label
            else f"{node_label} {image.name}".casefold()
        )

        if not any(token in label for token in include):
            continue
        if any(token in label for token in exclude):
            continue

        image_id = image.as_pointer()
        if image_id not in seen:
            seen.add(image_id)
            images.append(image)

    return images


def _material_tree_has_node_type(node_tree, wanted, visited=None):
    if node_tree is None:
        return False

    if visited is None:
        visited = set()

    pointer = node_tree.as_pointer()
    if pointer in visited:
        return False
    visited.add(pointer)

    for node in node_tree.nodes:
        if node.type in wanted:
            return True

        if node.type == "GROUP" and node.node_tree is not None:
            if _material_tree_has_node_type(
                node.node_tree, wanted, visited
            ):
                return True

    return False


def _bake_source_material_channel(obj, material, source_images, role):
    """Bake a SourceIO material channel before simplifying its node tree."""
    if not source_images:
        return None

    if len(obj.data.uv_layers) == 0:
        return None

    node_tree = material.node_tree

    has_bsdf = _material_tree_has_node_type(
        node_tree,
        {
            "BSDF_PRINCIPLED",
            "BSDF_DIFFUSE",
            "BSDF_GLOSSY",
            "BSDF_ANISOTROPIC",
        },
    )

    if role == "normal":
        if not has_bsdf:
            return None
        bake_type = "NORMAL"
    elif has_bsdf:
        bake_type = "DIFFUSE"
    elif _material_tree_has_node_type(node_tree, {"EMISSION"}):
        bake_type = "EMIT"
    else:
        return None

    width = min(
        4096,
        max(1, max(int(image.size[0]) for image in source_images)),
    )
    height = min(
        4096,
        max(1, max(int(image.size[1]) for image in source_images)),
    )

    image = bpy.data.images.new(
        name=f"{material.name}_{role}_baked",
        width=width,
        height=height,
        alpha=True,
        float_buffer=False,
    )
    image.use_fake_user = True

    scene = bpy.context.scene
    old_engine = scene.render.engine
    old_active = bpy.context.view_layer.objects.active
    old_selected = list(bpy.context.selected_objects)
    old_samples = getattr(scene.cycles, "samples", None)

    tree_states = []
    seen_trees = set()

    target_node = None
    success = False

    try:
        for slot in obj.material_slots:
            other_material = slot.material
            if (
                other_material is None
                or other_material.node_tree is None
            ):
                continue

            tree = other_material.node_tree
            pointer = tree.as_pointer()

            if pointer in seen_trees:
                continue
            seen_trees.add(pointer)

            selected_nodes = [
                (node, node.select) for node in tree.nodes
            ]
            tree_states.append(
                (tree, tree.nodes.active, selected_nodes)
            )

            for node in tree.nodes:
                node.select = False
            tree.nodes.active = None

        target_node = node_tree.nodes.new("ShaderNodeTexImage")
        target_node.name = "ValveConvert Bake Target"
        target_node.label = "ValveConvert Bake Target"
        target_node.image = image
        target_node.select = True
        node_tree.nodes.active = target_node

        for selected_obj in bpy.context.selected_objects:
            selected_obj.select_set(False)

        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj

        scene.render.engine = "CYCLES"
        if old_samples is not None:
            scene.cycles.samples = 1

        if bake_type == "NORMAL":
            bpy.ops.object.bake(
                type="NORMAL",
                normal_space="TANGENT",
                margin=4,
                use_clear=True,
            )
            image.colorspace_settings.name = "Non-Color"
        else:
            bake_options = {
                "type": bake_type,
                "margin": 4,
                "use_clear": True,
            }

            if bake_type == "DIFFUSE":
                bake_options["pass_filter"] = {"COLOR"}

            bpy.ops.object.bake(**bake_options)
            image.colorspace_settings.name = "sRGB"

        image.pack()
        success = True

    except Exception as exc:
        print(
            f"[model] Could not bake {role} for "
            f"{material.name!r}; retaining the original texture "
            f"selection: {type(exc).__name__}: {exc}",
            flush=True,
        )

    finally:
        if target_node is not None:
            try:
                node_tree.nodes.remove(target_node)
            except (ReferenceError, RuntimeError):
                pass

        for tree, active_node, selected_nodes in tree_states:
            for node, was_selected in selected_nodes:
                try:
                    node.select = was_selected
                except ReferenceError:
                    pass

            try:
                tree.nodes.active = active_node
            except (ReferenceError, RuntimeError):
                pass

        scene.render.engine = old_engine
        if old_samples is not None:
            scene.cycles.samples = old_samples

        for selected_obj in bpy.context.selected_objects:
            selected_obj.select_set(False)

        for selected_obj in old_selected:
            try:
                selected_obj.select_set(True)
            except ReferenceError:
                pass

        try:
            bpy.context.view_layer.objects.active = old_active
        except (ReferenceError, RuntimeError):
            pass

        if not success:
            bpy.data.images.remove(image, do_unlink=True)
            return None

    return image


def _source_material_parameter(material, wanted_name):
    params = material.get("vmt_parameters")
    wanted_name = wanted_name.casefold()

    def search(mapping):
        try:
            items = list(mapping.items())
        except (AttributeError, TypeError):
            return False, None

        for key, value in items:
            if str(key).casefold() == wanted_name:
                return True, value

        for _, value in items:
            if hasattr(value, "items"):
                found, result = search(value)
                if found:
                    return True, result

        return False, None

    return search(params)


def _source_material_alpha_policy(material):
    def flag(name):
        found, value = _source_material_parameter(material, name)
        if not found or value is None:
            return False

        if isinstance(value, str):
            return value.strip().casefold() in {
                "1", "true", "yes", "on"
            }

        return _finite_number(value, 0.0) != 0.0

    found_alpha, alpha_value = _source_material_parameter(
        material, "$alpha"
    )
    alpha_factor = (
        _finite_number(alpha_value, 1.0)
        if found_alpha
        else 1.0
    )
    alpha_factor = min(1.0, max(0.0, alpha_factor))

    found_cutoff, cutoff_value = _source_material_parameter(
        material, "$alphatestreference"
    )
    alpha_cutoff = (
        _finite_number(cutoff_value, 0.5)
        if found_cutoff
        else 0.5
    )

    translucent = flag("$translucent")
    alpha_test = flag("$alphatest")
    additive = flag("$additive")
    vertex_alpha = flag("$vertexalpha")

    has_vmt_parameters = bool(
        params := material.get("vmt_parameters")
    )

    if not has_vmt_parameters:
        surface_method = getattr(
            material, "surface_render_method", ""
        )
        blend_method = getattr(material, "blend_method", "")

        translucent = (
            surface_method == "BLENDED"
            or blend_method == "BLEND"
        )
        alpha_test = (
            surface_method == "DITHERED"
            or blend_method in {"HASHED", "CLIP"}
        )

    use_texture_alpha = translucent or alpha_test or additive

    return (
        use_texture_alpha,
        vertex_alpha,
        alpha_factor,
        alpha_test,
        alpha_cutoff,
    )


def _walk_material_nodes(node_tree, visited=None):
    """Walk a material graph, including nested node groups."""
    if node_tree is None:
        return

    if visited is None:
        visited = set()

    pointer = node_tree.as_pointer()
    if pointer in visited:
        return

    visited.add(pointer)

    for node in node_tree.nodes:
        yield node

        if node.type == "GROUP" and node.node_tree is not None:
            yield from _walk_material_nodes(
                node.node_tree, visited
            )


def _clone_material_groups(node_tree, cache=None, created=None):
    """
    Make nested node groups private to a temporary material.

    Baking modifies a copy of the graph, never SourceIO's original groups.
    """
    if cache is None:
        cache = {}

    if created is None:
        created = []

    for node in node_tree.nodes:
        if node.type != "GROUP" or node.node_tree is None:
            continue

        original = node.node_tree
        pointer = original.as_pointer()

        if pointer not in cache:
            copied = original.copy()
            cache[pointer] = copied
            created.append(copied)

            _clone_material_groups(
                copied, cache, created
            )

        node.node_tree = cache[pointer]

    return created


def _socket_by_names(node, names):
    for name in names:
        socket = node.inputs.get(name)
        if socket is not None:
            return socket

    return None


def _rgba_value(value, fallback):
    """Convert scalar or color socket values into an RGBA tuple."""
    try:
        values = tuple(value)
    except (TypeError, ValueError):
        try:
            scalar = float(value)
        except (TypeError, ValueError, OverflowError):
            scalar = fallback
        return (scalar, scalar, scalar, 1.0)

    if len(values) >= 4:
        return tuple(float(v) for v in values[:4])

    if len(values) == 3:
        return (*[float(v) for v in values], 1.0)

    if len(values) == 1:
        scalar = float(values[0])
        return (scalar, scalar, scalar, 1.0)

    return (fallback, fallback, fallback, 1.0)


def _material_blend_mode(material):
    """Return OPAQUE, MASK, or BLEND from Blender's material settings."""
    legacy = getattr(material, "blend_method", "OPAQUE")

    if legacy == "CLIP":
        return "MASK"

    if legacy in {"BLEND", "HASHED"}:
        return "BLEND"

    if getattr(material, "surface_render_method", "") == "BLENDED":
        return "BLEND"

    return "OPAQUE"


def _material_uses_channel(material, channel):
    """
    Determine whether a channel carries source information worth baking.

    This examines socket connections and shader nodes, not texture filenames.
    """
    nodes = list(_walk_material_nodes(material.node_tree))

    if channel == "base_color":
        return True

    if channel == "normal":
        for node in nodes:
            if node.type in {"NORMAL_MAP", "BUMP"}:
                return True

            if node.type == "BSDF_PRINCIPLED":
                socket = node.inputs.get("Normal")
                if socket is not None and socket.is_linked:
                    return True

        return False

    if channel == "alpha":
        if _material_blend_mode(material) != "OPAQUE":
            return True

        for node in nodes:
            if node.type in {"BSDF_TRANSPARENT", "HOLDOUT"}:
                return True

            socket = node.inputs.get("Alpha")
            if socket is not None:
                if socket.is_linked:
                    return True

                if _finite_number(
                    socket.default_value, 1.0
                ) < 0.99999:
                    return True

        return False

    if channel == "roughness":
        for node in nodes:
            socket = node.inputs.get("Roughness")
            if socket is None:
                continue

            if socket.is_linked:
                return True

            if abs(
                _finite_number(socket.default_value, 0.5) - 0.5
            ) > 1e-6:
                return True

        return False

    if channel == "metallic":
        for node in nodes:
            socket = node.inputs.get("Metallic")
            if socket is None:
                continue

            if socket.is_linked:
                return True

            if abs(
                _finite_number(socket.default_value, 0.0)
            ) > 1e-6:
                return True

        return False

    if channel == "emission":
        for node in nodes:
            if node.type == "EMISSION":
                return True

            if node.type != "BSDF_PRINCIPLED":
                continue

            color = _socket_by_names(
                node, ("Emission Color", "Emission")
            )
            strength = node.inputs.get("Emission Strength")

            if color is not None:
                if color.is_linked:
                    return True

                values = _rgba_value(
                    color.default_value, 0.0
                )
                if max(abs(v) for v in values[:3]) > 1e-6:
                    return True

            if strength is not None and strength.is_linked:
                return True

        return False

    return False


def _bypass_transparency_for_color(node_tree):
    """
    Remove direct Transparent-BSDF branches from a color bake.

    Color should be baked as straight RGB. Baking transparency into RGB as well
    would premultiply the color and produce dark fringes after glTF export.
    """
    for node in list(node_tree.nodes):
        if node.type == "GROUP" and node.node_tree is not None:
            _bypass_transparency_for_color(node.node_tree)

    for node in list(node_tree.nodes):
        if node.type == "MIX_SHADER":
            shader_inputs = (node.inputs[1], node.inputs[2])
        elif node.type == "ADD_SHADER":
            shader_inputs = (node.inputs[0], node.inputs[1])
        else:
            continue

        linked = []
        for socket in shader_inputs:
            links = list(socket.links)
            linked.append(links[0] if links else None)

        transparent_index = None

        for index, link in enumerate(linked):
            if (
                link is not None
                and link.from_node.type == "BSDF_TRANSPARENT"
            ):
                transparent_index = index
                break

        if transparent_index is None:
            continue

        other_index = 1 - transparent_index
        other_link = linked[other_index]

        # Only bypass the blend when its other shader branch is connected.
        if other_link is None:
            continue

        replacement_output = other_link.from_socket
        destinations = [
            link.to_socket
            for link in list(node.outputs[0].links)
        ]

        for link in list(node_tree.links):
            if link.from_node == node or link.to_node == node:
                node_tree.links.remove(link)

        for destination in destinations:
            node_tree.links.new(
                replacement_output, destination
            )

        node_tree.nodes.remove(node)


def _channel_input_for_shader(node, channel):
    """Return the source socket and fallback value for one baked channel."""
    if channel == "base_color":
        if node.type == "BSDF_PRINCIPLED":
            socket = node.inputs.get("Base Color")
        else:
            socket = node.inputs.get("Color")

        return socket, (0.0, 0.0, 0.0, 1.0)

    if channel == "roughness":
        return node.inputs.get("Roughness"), 0.5

    if channel == "metallic":
        return node.inputs.get("Metallic"), 0.0

    if channel == "emission":
        if node.type == "EMISSION":
            return node.inputs.get("Color"), (0.0, 0.0, 0.0, 1.0)

        if node.type == "BSDF_PRINCIPLED":
            return (
                _socket_by_names(
                    node, ("Emission Color", "Emission")
                ),
                (0.0, 0.0, 0.0, 1.0),
            )

        return None, (0.0, 0.0, 0.0, 1.0)

    if channel == "alpha":
        if node.type in {"BSDF_TRANSPARENT", "HOLDOUT"}:
            return None, 0.0

        return node.inputs.get("Alpha"), 1.0

    return None, 0.0


def _replace_shader_leaves_for_bake(node_tree, channel):
    """
    Replace surface shader leaves with emission shaders carrying one channel.

    MixShader, AddShader, math nodes, texture nodes, UV transforms and other
    graph logic remain in place. This is what preserves layered materials.
    """
    for node in list(node_tree.nodes):
        if node.type == "GROUP" and node.node_tree is not None:
            _replace_shader_leaves_for_bake(
                node.node_tree, channel
            )

    if channel == "base_color":
        _bypass_transparency_for_color(node_tree)

    combiners = {
        "GROUP",
        "GROUP_INPUT",
        "GROUP_OUTPUT",
        "OUTPUT_MATERIAL",
        "MIX_SHADER",
        "ADD_SHADER",
        "REROUTE",
    }

    for node in list(node_tree.nodes):
        if node.type in combiners:
            continue

        shader_outputs = [
            socket for socket in node.outputs
            if socket.type == "SHADER"
        ]
        if not shader_outputs:
            continue

        source_socket, fallback = _channel_input_for_shader(
            node, channel
        )

        source_link = None
        if source_socket is not None and source_socket.is_linked:
            source_link = source_socket.links[0].from_socket

        if source_link is not None:
            color_default = (0.0, 0.0, 0.0, 1.0)
        elif source_socket is not None:
            color_default = _rgba_value(
                source_socket.default_value,
                float(fallback)
                if isinstance(fallback, (int, float))
                else 0.0,
            )
        else:
            color_default = _rgba_value(fallback, 0.0)

        # Emission-only materials carry their color through the emission
        # channel. Their roughness and metallic channels are not meaningful.
        emission = node_tree.nodes.new("ShaderNodeEmission")
        emission.label = f"ValveConvert bake: {channel}"
        emission.inputs["Color"].default_value = color_default
        emission.inputs["Strength"].default_value = 1.0
        emission.location = node.location

        if source_link is not None:
            node_tree.links.new(
                source_link, emission.inputs["Color"]
            )

        destinations = [
            link.to_socket
            for link in list(node_tree.links)
            if link.from_node == node
            and link.from_socket.type == "SHADER"
        ]

        node_tree.nodes.remove(node)

        for destination in destinations:
            node_tree.links.new(
                emission.outputs["Emission"], destination
            )


def _read_uv_layer_values(mesh, layer_name):
    """Read one UV layer into independent Python-owned storage."""
    from array import array

    layer = mesh.uv_layers.get(layer_name)
    if layer is None:
        raise RuntimeError(
            f"UV layer {layer_name!r} does not exist."
        )

    values = array(
        "f", [0.0]
    ) * (len(layer.data) * 2)

    if values:
        layer.data.foreach_get("uv", values)

    return values


def _uv_bounds(values):
    """Return min U, max U, min V, max V."""
    if not values:
        raise ValueError("Cannot calculate bounds of an empty UV layer.")

    return (
        min(values[0::2]),
        max(values[0::2]),
        min(values[1::2]),
        max(values[1::2]),
    )


def _uv_area_by_material(mesh, values):
    """
    Sum absolute UV polygon areas, grouped by material index.
    """
    areas = {}

    for polygon in mesh.polygons:
        start = polygon.loop_start
        count = polygon.loop_total

        if count < 3:
            continue

        twice_area = 0.0

        for offset in range(count):
            current = start + offset
            following = start + ((offset + 1) % count)

            u1 = values[current * 2]
            v1 = values[current * 2 + 1]
            u2 = values[following * 2]
            v2 = values[following * 2 + 1]

            twice_area += u1 * v2 - u2 * v1

        area = abs(twice_area) * 0.5

        material_index = polygon.material_index
        areas[material_index] = (
            areas.get(material_index, 0.0) + area
        )

    return areas


def _set_active_uv_layer(mesh, layer_name):
    """
    Make one UV layer both active and active-render.
    """
    index = mesh.uv_layers.find(layer_name)

    if index < 0:
        raise RuntimeError(
            f"Cannot activate missing UV layer {layer_name!r}."
        )

    mesh.uv_layers.active_index = index

    for layer in mesh.uv_layers:
        layer.active_render = (layer.name == layer_name)


def _prepare_bake_uv_layout(obj):
    """
    Preserve all original UV maps and generate an independent bake atlas.
    """
    mesh = obj.data

    if obj.type != "MESH":
        raise TypeError(
            f"Expected a mesh object, got {obj.type!r}."
        )

    if obj.mode != "OBJECT":
        raise RuntimeError(
            f"Cannot prepare UVs for {obj.name!r}: "
            "the object must be in Object Mode."
        )

    if len(mesh.uv_layers) == 0:
        raise RuntimeError(
            f"Cannot bake {obj.name!r}: SourceIO supplied no UV maps."
        )

    source_active_layer = mesh.uv_layers.active
    if source_active_layer is None:
        source_active_layer = mesh.uv_layers[0]

    source_render_layer = next(
        (
            layer
            for layer in mesh.uv_layers
            if layer.active_render
        ),
        source_active_layer,
    )

    source_active_name = source_active_layer.name
    source_render_name = source_render_layer.name

    source_snapshots = {}
    source_areas = {}

    expected_value_count = len(mesh.loops) * 2

    for layer in mesh.uv_layers:
        layer_name = layer.name
        values = _read_uv_layer_values(mesh, layer_name)

        if len(values) != expected_value_count:
            raise RuntimeError(
                f"UV layer {layer_name!r} on {obj.name!r} has "
                f"{len(values) // 2} UV loops; expected "
                f"{len(mesh.loops)}."
            )

        if not all(math.isfinite(value) for value in values):
            raise RuntimeError(
                f"UV layer {layer_name!r} on {obj.name!r} "
                "contains non-finite coordinates."
            )

        source_snapshots[layer_name] = values
        source_areas[layer_name] = _uv_area_by_material(
            mesh,
            values,
        )

    bake_name = "VCT_BakeUV"
    suffix = 1

    while mesh.uv_layers.get(bake_name) is not None:
        bake_name = f"VCT_BakeUV_{suffix}"
        suffix += 1

    previous_selected = tuple(
        bpy.context.selected_objects
    )
    previous_active = (
        bpy.context.view_layer.objects.active
    )

    layer_created = False

    try:
        for selected in bpy.context.selected_objects:
            selected.select_set(False)

        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj

        bpy.context.view_layer.update()

        mesh = obj.data
        mesh.uv_layers.new(
            name=bake_name,
            do_init=False,
        )
        layer_created = True

        _set_active_uv_layer(mesh, bake_name)

        mesh.update()
        bpy.context.view_layer.update()

        mode_result = bpy.ops.object.mode_set(mode="EDIT")
        if "CANCELLED" in mode_result:
            raise RuntimeError(
                f"Could not enter Edit Mode for UV atlas generation "
                f"on {obj.name!r}."
            )

        bpy.ops.mesh.select_all(action="SELECT")

        result = bpy.ops.uv.smart_project(
            angle_limit=math.radians(45.0),
            margin_method="FRACTION",
            rotate_method="AXIS_ALIGNED_Y",
            island_margin=0.002,
            area_weight=1.0,
            correct_aspect=True,
            scale_to_bounds=False,
        )

        if "CANCELLED" in result:
            raise RuntimeError(
                f"Blender cancelled Smart UV Project for {obj.name!r}."
            )

        bpy.ops.object.mode_set(mode="OBJECT")

        mesh = obj.data

        for layer_name, before in source_snapshots.items():
            after = _read_uv_layer_values(mesh, layer_name)

            if (
                len(after) != len(before)
                or any(
                    abs(old - new) > 1e-6
                    for old, new in zip(before, after)
                )
            ):
                raise RuntimeError(
                    f"UV atlas generation modified original UV map "
                    f"{layer_name!r} on {obj.name!r}. "
                    "The original source UVs must remain unchanged."
                )

        bake_values = _read_uv_layer_values(
            mesh,
            bake_name,
        )

        if len(bake_values) != expected_value_count:
            raise RuntimeError(
                f"Generated bake UV map {bake_name!r} on {obj.name!r} "
                "has an incorrect loop count."
            )

        if not all(math.isfinite(value) for value in bake_values):
            raise RuntimeError(
                f"Generated bake UV map {bake_name!r} on {obj.name!r} "
                "contains non-finite coordinates."
            )

        bake_bounds = _uv_bounds(bake_values)

        if any(
            value < -1e-5 or value > 1.0 + 1e-5
            for value in bake_values
        ):
            raise RuntimeError(
                f"Generated bake UV map {bake_name!r} on {obj.name!r} "
                f"is outside 0..1: {bake_bounds}"
            )

        bake_areas = _uv_area_by_material(
            mesh,
            bake_values,
        )

        for polygon in mesh.polygons:
            if polygon.area <= 1e-12:
                continue

            if bake_areas.get(polygon.material_index, 0.0) <= 1e-12:
                raise RuntimeError(
                    f"Generated bake UV map has no usable coverage for "
                    f"material slot {polygon.material_index} on "
                    f"{obj.name!r}."
                )

        _set_active_uv_layer(mesh, bake_name)

        print(
            f"[uv-bake] {obj.name!r}: "
            f"source_active={source_active_name!r}; "
            f"source_render={source_render_name!r}; "
            f"source_active_bounds="
            f"{_uv_bounds(source_snapshots[source_active_name])}; "
            f"bake={bake_name!r}; "
            f"bake_bounds={bake_bounds}; "
            f"preserved_source_uv_maps={len(source_snapshots)}",
            flush=True,
        )

        return (
            source_active_name,
            source_render_name,
            bake_name,
            source_areas,
            bake_areas,
        )

    except Exception:
        if obj.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")

        if layer_created:
            mesh = obj.data
            generated_layer = mesh.uv_layers.get(bake_name)

            if generated_layer is not None:
                mesh.uv_layers.remove(generated_layer)

        raise

    finally:
        if obj.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")

        for selected in bpy.context.selected_objects:
            selected.select_set(False)

        for selected in previous_selected:
            if (
                bpy.context.view_layer.objects.get(selected.name)
                is not None
            ):
                selected.select_set(True)

        if (
            previous_active is not None
            and bpy.context.view_layer.objects.get(previous_active.name)
            is not None
        ):
            bpy.context.view_layer.objects.active = previous_active
        else:
            bpy.context.view_layer.objects.active = None


def _pin_material_uv_sampling(
    node_tree,
    source_active_uv_name,
    source_render_uv_name,
    visited=None,
):
    """
    Make all source material coordinate sampling independent of the
    active bake UV map.
    """
    if node_tree is None:
        return {
            "image_inputs": 0,
            "uv_map_nodes": 0,
            "texture_coordinate_nodes": 0,
        }

    if visited is None:
        visited = set()

    pointer = node_tree.as_pointer()
    if pointer in visited:
        return {
            "image_inputs": 0,
            "uv_map_nodes": 0,
            "texture_coordinate_nodes": 0,
        }

    visited.add(pointer)

    counts = {
        "image_inputs": 0,
        "uv_map_nodes": 0,
        "texture_coordinate_nodes": 0,
    }

    uv_nodes = {}

    def get_uv_node(uv_name):
        uv_node = uv_nodes.get(uv_name)

        if uv_node is None:
            uv_node = node_tree.nodes.new("ShaderNodeUVMap")
            uv_node.name = f"VCT Source UV {uv_name}"
            uv_node.label = f"VCT Source UV: {uv_name}"
            uv_node.uv_map = uv_name
            uv_nodes[uv_name] = uv_node

        return uv_node

    for node in list(node_tree.nodes):
        if node.type == "GROUP" and node.node_tree is not None:
            nested = _pin_material_uv_sampling(
                node.node_tree,
                source_active_uv_name,
                source_render_uv_name,
                visited,
            )

            for key, value in nested.items():
                counts[key] += value

        elif node.type == "UVMAP":
            if not str(node.uv_map or "").strip():
                node.uv_map = source_render_uv_name
                counts["uv_map_nodes"] += 1

        elif node.type == "TEX_COORD":
            uv_output = node.outputs.get("UV")

            if uv_output is not None and uv_output.is_linked:
                source_node = get_uv_node(source_render_uv_name)

                for link in list(uv_output.links):
                    destination = link.to_socket
                    node_tree.links.remove(link)
                    node_tree.links.new(
                        source_node.outputs["UV"],
                        destination,
                    )

                counts["texture_coordinate_nodes"] += 1

        elif node.type == "TEX_IMAGE" and node.image is not None:
            vector = node.inputs.get("Vector")

            if vector is not None and not vector.is_linked:
                source_node = get_uv_node(source_active_uv_name)

                node_tree.links.new(
                    source_node.outputs["UV"],
                    vector,
                )

                counts["image_inputs"] += 1

    return counts


def _choose_material_bake_resolution(
    source_images,
    material_slot_index,
    source_uv_areas,
    bake_uv_areas,
):
    """
    Estimate atlas resolution from source UV texel area versus destination
    atlas area. The 4096 cap prevents unbounded image allocation.
    """
    base_resolution = max(
        [512]
        + [
            max(int(image.size[0]), int(image.size[1]))
            for image in source_images
            if image.size[0] > 0 and image.size[1] > 0
        ]
    )

    source_area = max(
        (
            material_areas.get(material_slot_index, 0.0)
            for material_areas in source_uv_areas.values()
        ),
        default=0.0,
    )

    bake_area = bake_uv_areas.get(material_slot_index, 0.0)

    if source_area > 1e-12 and bake_area <= 1e-12:
        raise RuntimeError(
            f"Material slot {material_slot_index} has source UV area "
            "but no corresponding bake atlas area."
        )

    requested_resolution = base_resolution

    if source_area > 1e-12 and bake_area > 1e-12:
        density_scale = math.sqrt(source_area / bake_area)
        requested_resolution = max(
            base_resolution,
            math.ceil(base_resolution * density_scale),
        )

    maximum_resolution = 4096
    resolution = max(
        64,
        min(maximum_resolution, requested_resolution),
    )

    if requested_resolution > maximum_resolution:
        print(
            f"[material] WARNING: Estimated bake resolution "
            f"{requested_resolution} exceeds the "
            f"{maximum_resolution} limit for material slot "
            f"{material_slot_index}; the atlas will have reduced "
            "texel density.",
            flush=True,
        )

    return (
        resolution,
        requested_resolution,
        source_area,
        bake_area,
    )


def _bake_source_material_channel(
    obj,
    slot_index,
    source_material,
    channel,
    resolution,
    source_active_uv_name,
    source_render_uv_name,
    bake_uv_name,
):
    """Bake one evaluated channel into a texture."""
    temporary_material = source_material.copy()
    created_groups = []
    image = None

    scene = bpy.context.scene
    slot = obj.material_slots[slot_index]

    old_material = slot.material
    old_engine = scene.render.engine
    old_active_object = bpy.context.view_layer.objects.active
    old_selected_objects = list(bpy.context.selected_objects)
    old_samples = scene.cycles.samples
    old_margin = scene.render.bake.margin

    tree_states = []
    seen_trees = set()

    try:
        created_groups = _clone_material_groups(
            temporary_material.node_tree
        )

        uv_pin_counts = _pin_material_uv_sampling(
            temporary_material.node_tree,
            source_active_uv_name,
            source_render_uv_name,
        )

        print(
            f"[uv-source] {obj.name!r} / {source_material.name!r}: "
            f"pinned_image_inputs={uv_pin_counts['image_inputs']}; "
            f"pinned_uv_map_nodes={uv_pin_counts['uv_map_nodes']}; "
            f"pinned_texture_coordinate_nodes="
            f"{uv_pin_counts['texture_coordinate_nodes']}",
            flush=True,
        )

        if channel != "normal":
            _replace_shader_leaves_for_bake(
                temporary_material.node_tree, channel
            )

        image = bpy.data.images.new(
            name=f"{source_material.name}_{channel}_bake",
            width=resolution,
            height=resolution,
            alpha=True,
            float_buffer=False,
        )

        if channel in {"normal", "roughness", "metallic", "alpha"}:
            image.colorspace_settings.name = "Non-Color"
        else:
            image.colorspace_settings.name = "sRGB"

        target = temporary_material.node_tree.nodes.new(
            "ShaderNodeTexImage"
        )
        target.name = "ValveConvert Bake Target"
        target.image = image
        target.select = True
        temporary_material.node_tree.nodes.active = target

        slot.material = temporary_material

        for material_slot in obj.material_slots:
            material = material_slot.material

            if (
                material is None
                or material.node_tree is None
            ):
                continue

            tree = material.node_tree
            pointer = tree.as_pointer()

            if pointer in seen_trees:
                continue

            seen_trees.add(pointer)
            tree_states.append(
                (
                    tree,
                    tree.nodes.active,
                    [
                        (node, node.select)
                        for node in tree.nodes
                    ],
                )
            )

            for node in tree.nodes:
                node.select = False

            tree.nodes.active = None

        temporary_material.node_tree.nodes.active = target
        target.select = True

        for selected in bpy.context.selected_objects:
            selected.select_set(False)

        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj

        _set_active_uv_layer(
            obj.data,
            bake_uv_name,
        )

        scene.render.engine = "CYCLES"
        scene.cycles.samples = 1
        scene.render.bake.margin = 8

        if channel == "normal":
            bpy.ops.object.bake(
                type="NORMAL",
                normal_space="TANGENT",
                margin=8,
                use_clear=True,
            )
        else:
            bpy.ops.object.bake(
                type="EMIT",
                margin=8,
                use_clear=True,
            )

        image.pack()
        return image

    except Exception as exc:
        if image is not None:
            bpy.data.images.remove(image, do_unlink=True)

        raise RuntimeError(
            f"Could not bake {channel!r} for material "
            f"{source_material.name!r} on object {obj.name!r}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    finally:
        slot.material = old_material

        for tree, active_node, node_states in tree_states:
            for node, selected in node_states:
                try:
                    node.select = selected
                except ReferenceError:
                    pass

            try:
                tree.nodes.active = active_node
            except (ReferenceError, RuntimeError):
                pass

        scene.render.engine = old_engine
        scene.cycles.samples = old_samples
        scene.render.bake.margin = old_margin

        for selected in bpy.context.selected_objects:
            selected.select_set(False)

        for selected in old_selected_objects:
            try:
                selected.select_set(True)
            except ReferenceError:
                pass

        try:
            bpy.context.view_layer.objects.active = old_active_object
        except (ReferenceError, RuntimeError):
            pass

        bpy.data.materials.remove(
            temporary_material, do_unlink=True
        )

        for group in reversed(created_groups):
            if group.users == 0:
                bpy.data.node_groups.remove(group)


def _connect_probe_input(tree, source_socket, target_socket):
    if source_socket is None:
        return

    if source_socket.is_linked:
        if source_socket.type == "VALUE":
            combine = tree.nodes.new("ShaderNodeCombineColor")
            for name in ("Red", "Green", "Blue"):
                for link in list(source_socket.links):
                    tree.links.new(link.from_socket, combine.inputs[name])
            tree.links.new(combine.outputs["Color"], target_socket)
        else:
            for link in list(source_socket.links):
                tree.links.new(link.from_socket, target_socket)
        return

    value = getattr(source_socket, "default_value", None)
    if value is None:
        return

    if source_socket.type == "VALUE":
        value = float(value)
        target_socket.default_value = (value, value, value, 1.0)
    else:
        target_socket.default_value = value


def _replace_shader_leaves_with_channel_emission(
    tree,
    channel,
    visited=None,
):
    """Keep upstream paths intact; replace surface leaves with channel probes."""
    if tree is None:
        return 0

    if visited is None:
        visited = set()

    pointer = tree.as_pointer()
    if pointer in visited:
        return 0
    visited.add(pointer)

    converted = 0

    for node in list(tree.nodes):
        if node.type == "GROUP" and node.node_tree is not None:
            converted += _replace_shader_leaves_with_channel_emission(
                node.node_tree,
                channel,
                visited,
            )

    shader_types = {
        "BSDF_PRINCIPLED",
        "EMISSION",
        "BSDF_DIFFUSE",
        "BSDF_GLOSSY",
        "BSDF_ANISOTROPIC",
        "BSDF_TRANSLUCENT",
        "BSDF_GLASS",
        "BSDF_REFRACTION",
        "BSDF_TOON",
        "BSDF_VELVET",
    }

    for node in list(tree.nodes):
        if node.type not in shader_types:
            continue

        output = next(
            (
                socket
                for socket in node.outputs
                if socket.type == "SHADER" and socket.is_linked
            ),
            None,
        )
        if output is None:
            continue

        destinations = [
            link.to_socket
            for link in list(output.links)
        ]

        probe = tree.nodes.new("ShaderNodeEmission")
        probe.location = node.location
        probe.label = f"VCT {channel} probe"

        color = probe.inputs["Color"]

        if channel == "base_color":
            source = (
                node.inputs.get("Color")
                if node.type == "EMISSION"
                else (
                    node.inputs.get("Base Color")
                    or node.inputs.get("Color")
                )
            )
            _connect_probe_input(tree, source, color)

        elif channel == "emission":
            source = (
                node.inputs.get("Color")
                if node.type == "EMISSION"
                else (
                    node.inputs.get("Emission Color")
                    or node.inputs.get("Emission")
                )
            )

            strength = (
                node.inputs.get("Strength")
                if node.type == "EMISSION"
                else node.inputs.get("Emission Strength")
            )

            _connect_probe_input(tree, source, color)

            if strength is None:
                probe.inputs["Strength"].default_value = 0.0
            elif strength.is_linked:
                for link in list(strength.links):
                    tree.links.new(
                        link.from_socket,
                        probe.inputs["Strength"],
                    )
            else:
                probe.inputs["Strength"].default_value = float(
                    strength.default_value
                )

        elif channel in {"roughness", "metallic"}:
            source = node.inputs.get(
                "Roughness" if channel == "roughness" else "Metallic"
            )
            default = 0.5 if channel == "roughness" else 0.0

            if source is None:
                color.default_value = (
                    default,
                    default,
                    default,
                    1.0,
                )
            else:
                _connect_probe_input(tree, source, color)

        for destination in destinations:
            tree.links.new(
                probe.outputs["Emission"],
                destination,
            )

        tree.nodes.remove(node)
        converted += 1

    return converted


def _render_sourceio_channel(
    source_material,
    channel,
    uv_names,
    active_uv,
    render_uv,
    width,
    height,
    alpha_pass=False,
):
    """
    Render the connected material graph on a temporary normalized UV quad.
    """
    material = source_material.copy()
    material.name = f"{source_material.name}_VCT_{channel}"

    groups = _clone_material_groups(material.node_tree)
    scene = None
    image = None

    try:
        _pin_material_uv_sampling(
            material.node_tree,
            active_uv,
            render_uv,
        )

        for node, _ in _iter_material_image_nodes(material.node_tree):
            node.extension = "REPEAT"

        if alpha_pass:
            if hasattr(material, "blend_method"):
                material.blend_method = "BLEND"
            if hasattr(material, "surface_render_method"):
                material.surface_render_method = "BLENDED"

        else:
            _bypass_transparency_for_color(material.node_tree)

            if not _replace_shader_leaves_with_channel_emission(
                material.node_tree,
                channel,
            ):
                raise RuntimeError(
                    f"No connected shader output for {channel!r}."
                )

        aspect = width / height

        scene = bpy.data.scenes.new(f"VCT_{channel}_Scene")
        available_engines = {
            item.identifier
            for item in scene.render.bl_rna.properties["engine"].enum_items
        }

        render_engine = next(
            (
                engine
                for engine in (
                    "BLENDER_EEVEE_NEXT",
                    "BLENDER_EEVEE",
                    "CYCLES",
                )
                if engine in available_engines
            ),
            None,
        )

        if render_engine is None:
            raise RuntimeError(
                "No supported material-compositing render engine is available. "
                f"Available engines: {sorted(available_engines)}"
            )

        scene.render.engine = render_engine
        scene.render.resolution_x = width
        scene.render.resolution_y = height
        scene.render.resolution_percentage = 100
        scene.render.film_transparent = True

        scene.render.image_settings.file_format = "PNG"
        scene.render.image_settings.color_mode = "RGBA"
        scene.render.image_settings.color_depth = "16"
        scene.render.image_settings.compression = 0
        scene.render.use_file_extension = False

        scene.view_settings.view_transform = (
            "Standard"
            if channel in {"base_color", "emission"}
            else "Raw"
        )

        try:
            scene.view_settings.look = "None"
        except (AttributeError, TypeError):
            pass

        camera_data = bpy.data.cameras.new(
            f"VCT_{channel}_CameraData"
        )
        camera_data.type = "ORTHO"
        camera_data.ortho_scale = aspect
        camera_data.clip_start = 0.01
        camera_data.clip_end = 100.0

        camera = bpy.data.objects.new(
            f"VCT_{channel}_Camera",
            camera_data,
        )
        camera.location = (aspect * 0.5, 0.5, 2.0)
        scene.collection.objects.link(camera)
        scene.camera = camera

        mesh = bpy.data.meshes.new(f"VCT_{channel}_QuadMesh")
        mesh.from_pydata(
            [
                (0, 0, 0),
                (aspect, 0, 0),
                (aspect, 1, 0),
                (0, 1, 0),
            ],
            [],
            [(0, 1, 2, 3)],
        )

        uv_values = (
            (0, 0),
            (1, 0),
            (1, 1),
            (0, 1),
        )

        for name in uv_names:
            layer = mesh.uv_layers.new(name=name)

            for loop_index, loop in enumerate(mesh.loops):
                layer.data[loop_index].uv = uv_values[
                    loop.vertex_index
                ]

            if name == active_uv:
                mesh.uv_layers.active_index = len(mesh.uv_layers) - 1

        plane = bpy.data.objects.new(
            f"VCT_{channel}_Quad",
            mesh,
        )
        scene.collection.objects.link(plane)
        plane.data.materials.append(material)

        with tempfile.TemporaryDirectory(
            prefix="vct_graph_render_"
        ) as temp_dir:
            filepath = os.path.join(temp_dir, f"{channel}.png")
            scene.render.filepath = filepath

            result = bpy.ops.render.render(
                write_still=True,
                scene=scene.name,
            )

            if "CANCELLED" in result or not os.path.isfile(filepath):
                raise RuntimeError(
                    "Eevee did not write the composite."
                )

            image = bpy.data.images.load(
                filepath,
                check_existing=False,
            )
            image.name = (
                f"{source_material.name}_{channel}_composite"
            )
            image.alpha_mode = "STRAIGHT"

            image.colorspace_settings.name = (
                "sRGB"
                if channel in {"base_color", "emission"}
                else "Non-Color"
            )
            image.pack()

        return image

    except Exception as exc:
        if image is not None:
            bpy.data.images.remove(image, do_unlink=True)

        raise RuntimeError(
            f"Node-path evaluation failed for "
            f"{source_material.name!r} ({channel}): "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    finally:
        if scene is not None:
            for obj in list(scene.objects):
                datablock = obj.data
                bpy.data.objects.remove(obj, do_unlink=True)

                if datablock is not None and datablock.users == 0:
                    if isinstance(datablock, bpy.types.Mesh):
                        bpy.data.meshes.remove(datablock)
                    elif isinstance(datablock, bpy.types.Camera):
                        bpy.data.cameras.remove(datablock)

            bpy.data.scenes.remove(scene)

        if material.name in bpy.data.materials:
            bpy.data.materials.remove(material, do_unlink=True)

        for group in reversed(groups):
            if group.users == 0:
                bpy.data.node_groups.remove(group)


def _repack_modified_image(image):
    """Pack the current image buffer, independent of its original filepath."""
    original_filepath = image.filepath_raw
    original_format = image.file_format

    try:
        with tempfile.TemporaryDirectory(
            prefix="vct_image_pack_"
        ) as temp_dir:
            filepath = os.path.join(
                temp_dir,
                "composite.png",
            )

            image.filepath_raw = filepath
            image.file_format = "PNG"
            image.save()

            with open(filepath, "rb") as file:
                encoded_data = file.read()

            if not encoded_data:
                raise RuntimeError(
                    f"Image encoding produced an empty file: {image.name!r}"
                )

            image.pack(
                data=encoded_data,
                data_len=len(encoded_data),
            )

    finally:
        image.filepath_raw = original_filepath
        image.file_format = original_format


def _merge_rendered_alpha(color_image, alpha_image, factor=1.0):
    import numpy as np

    if tuple(color_image.size) != tuple(alpha_image.size):
        raise RuntimeError(
            "Composite color/alpha dimensions do not match."
        )

    count = color_image.size[0] * color_image.size[1]

    color = np.empty(count * 4, dtype=np.float32)
    alpha = np.empty(count * 4, dtype=np.float32)

    color_image.pixels.foreach_get(color)
    alpha_image.pixels.foreach_get(alpha)

    color[3::4] = np.clip(
        alpha[3::4] * factor,
        0.0,
        1.0,
    )

    color_image.pixels.foreach_set(color)
    color_image.alpha_mode = "STRAIGHT"
    color_image.update()

    _repack_modified_image(color_image)


def _preserve_sourceio_normal_map(
    source,
    tree,
    principled,
    uv_name,
):
    """Keep SourceIO's direct image -> NormalMap path in tangent space."""
    source_bsdf = next(
        (
            node
            for node in _walk_material_nodes(source.node_tree)
            if node.type == "BSDF_PRINCIPLED"
            and node.inputs.get("Normal") is not None
            and node.inputs["Normal"].is_linked
        ),
        None,
    )

    if source_bsdf is None:
        return

    normal_map = source_bsdf.inputs["Normal"].links[0].from_node

    if normal_map.type != "NORMAL_MAP":
        return

    color_input = normal_map.inputs.get("Color")
    if color_input is None or not color_input.is_linked:
        return

    image_node = color_input.links[0].from_node
    if image_node.type != "TEX_IMAGE" or image_node.image is None:
        return

    uv = tree.nodes.new("ShaderNodeUVMap")
    uv.uv_map = uv_name

    texture = tree.nodes.new("ShaderNodeTexImage")
    texture.image = image_node.image
    texture.extension = "REPEAT"

    normal = tree.nodes.new("ShaderNodeNormalMap")
    normal.space = normal_map.space

    tree.links.new(uv.outputs["UV"], texture.inputs["Vector"])
    tree.links.new(texture.outputs["Color"], normal.inputs["Color"])
    tree.links.new(normal.outputs["Normal"], principled.inputs["Normal"])


def prepare_materials_for_gltf(mesh_objects):
    """Flatten connected SourceIO channel paths without Cycles or UV atlasing."""
    for obj in mesh_objects:
        if obj.type != "MESH":
            continue

        mesh = obj.data

        if not mesh.uv_layers:
            if any(
                slot.material is not None
                and slot.material.node_tree is not None
                and any(
                    True
                    for _ in _iter_material_image_nodes(
                        slot.material.node_tree
                    )
                )
                for slot in obj.material_slots
            ):
                raise RuntimeError(
                    f"Textured mesh {obj.name!r} has no SourceIO UV map."
                )
            continue

        active_layer = mesh.uv_layers.active or mesh.uv_layers[0]
        render_layer = next(
            (
                layer
                for layer in mesh.uv_layers
                if layer.active_render
            ),
            active_layer,
        )

        active_uv = active_layer.name
        render_uv = render_layer.name
        uv_names = [layer.name for layer in mesh.uv_layers]

        for slot in obj.material_slots:
            source = slot.material

            if source is None or source.node_tree is None:
                continue

            nodes = list(_walk_material_nodes(source.node_tree))

            uses_multiple_uvs = (
                active_uv != render_uv
                and any(
                    node.type == "TEX_COORD"
                    and node.outputs.get("UV") is not None
                    and node.outputs["UV"].is_linked
                    for node in nodes
                )
            ) or any(
                node.type == "UVMAP"
                and node.uv_map
                and node.uv_map != active_uv
                for node in nodes
            )

            if uses_multiple_uvs:
                print(
                    f"[material] Keeping {source.name!r} native on "
                    f"{obj.name!r}: graph samples another UV domain.",
                    flush=True,
                )
                continue

            images = [
                image
                for _, image in _iter_material_image_nodes(
                    source.node_tree
                )
            ]

            width = max(
                [512] + [int(image.size[0]) for image in images]
            )
            height = max(
                [512] + [int(image.size[1]) for image in images]
            )

            is_unlit = (
                not any(
                    node.type == "BSDF_PRINCIPLED"
                    for node in nodes
                )
                and any(
                    node.type == "EMISSION"
                    for node in nodes
                )
            )

            (
                use_alpha,
                vertex_alpha,
                alpha_factor,
                alpha_test,
                cutoff,
            ) = _source_material_alpha_policy(source)

            if vertex_alpha:
                print(
                    f"[material] Keeping {source.name!r} native on "
                    f"{obj.name!r}: $vertexalpha is mesh-dependent.",
                    flush=True,
                )
                continue

            has_alpha = (
                _material_uses_channel(source, "alpha")
                or use_alpha
                or alpha_factor < 0.99999
                or any(
                    node.type in {"BSDF_TRANSPARENT", "HOLDOUT"}
                    for node in nodes
                )
            )

            base_image = _render_sourceio_channel(
                source,
                "base_color",
                uv_names,
                active_uv,
                render_uv,
                width,
                height,
            )

            if has_alpha:
                alpha_image = _render_sourceio_channel(
                    source,
                    "alpha",
                    uv_names,
                    active_uv,
                    render_uv,
                    width,
                    height,
                    alpha_pass=True,
                )

                try:
                    _merge_rendered_alpha(
                        base_image,
                        alpha_image,
                        alpha_factor,
                    )
                finally:
                    bpy.data.images.remove(
                        alpha_image,
                        do_unlink=True,
                    )

            emission_image = None
            if (
                not is_unlit
                and _material_uses_channel(source, "emission")
            ):
                emission_image = _render_sourceio_channel(
                    source,
                    "emission",
                    uv_names,
                    active_uv,
                    render_uv,
                    width,
                    height,
                )

            roughness_image = None
            if _material_uses_channel(source, "roughness"):
                roughness_image = _render_sourceio_channel(
                    source,
                    "roughness",
                    uv_names,
                    active_uv,
                    render_uv,
                    width,
                    height,
                )

            metallic_image = None
            if _material_uses_channel(source, "metallic"):
                metallic_image = _render_sourceio_channel(
                    source,
                    "metallic",
                    uv_names,
                    active_uv,
                    render_uv,
                    width,
                    height,
                )

            final = source.copy()
            final.name = f"{source.name}_glTF"

            tree = final.node_tree
            tree.nodes.clear()

            output = tree.nodes.new("ShaderNodeOutputMaterial")
            bsdf = tree.nodes.new("ShaderNodeBsdfPrincipled")

            tree.links.new(
                bsdf.outputs["BSDF"],
                output.inputs["Surface"],
            )

            uv_node = tree.nodes.new("ShaderNodeUVMap")
            uv_node.name = "VCT Composite UV"
            uv_node.uv_map = active_uv

            def add_texture(image, socket_name, y):
                texture = tree.nodes.new("ShaderNodeTexImage")
                texture.image = image
                texture.extension = "REPEAT"
                texture.location = (-450, y)

                tree.links.new(
                    uv_node.outputs["UV"],
                    texture.inputs["Vector"],
                )

                socket = bsdf.inputs.get(socket_name)
                if socket is not None:
                    tree.links.new(
                        texture.outputs["Color"],
                        socket,
                    )

                return texture

            base_texture = add_texture(
                base_image,
                "Base Color",
                200,
            )

            if has_alpha:
                alpha_socket = bsdf.inputs.get("Alpha")
                if alpha_socket is not None:
                    tree.links.new(
                        base_texture.outputs["Alpha"],
                        alpha_socket,
                    )

            if roughness_image is not None:
                add_texture(
                    roughness_image,
                    "Roughness",
                    -100,
                )

            if metallic_image is not None:
                add_texture(
                    metallic_image,
                    "Metallic",
                    -300,
                )

            if emission_image is not None:
                emission_socket = (
                    bsdf.inputs.get("Emission Color")
                    or bsdf.inputs.get("Emission")
                )

                if emission_socket is not None:
                    add_texture(
                        emission_image,
                        emission_socket.name,
                        -500,
                    )

                strength = bsdf.inputs.get("Emission Strength")
                if strength is not None:
                    strength.default_value = 1.0

            _preserve_sourceio_normal_map(
                source,
                tree,
                bsdf,
                active_uv,
            )

            if has_alpha:
                if alpha_test:
                    if hasattr(final, "blend_method"):
                        final.blend_method = "CLIP"
                    if hasattr(final, "surface_render_method"):
                        final.surface_render_method = "DITHERED"
                    if hasattr(final, "alpha_threshold"):
                        final.alpha_threshold = cutoff
                else:
                    if hasattr(final, "blend_method"):
                        final.blend_method = "BLEND"
                    if hasattr(final, "surface_render_method"):
                        final.surface_render_method = "BLENDED"

            slot.material = final

            print(
                f"[material] Node-path composite {source.name!r} "
                f"on {obj.name!r}: {width}x{height}; "
                f"alpha={has_alpha}; "
                f"emission={emission_image is not None}; "
                f"roughness={roughness_image is not None}; "
                f"metallic={metallic_image is not None}; "
                f"uv={active_uv!r}; "
                f"source_images={len(images)}",
                flush=True,
            )

    for image in bpy.data.images:
        if (
            image.users > 0
            and image.has_data
            and image.packed_file is None
        ):
            try:
                image.pack()
            except Exception as exc:
                print(
                    f"[material] Could not pack {image.name!r}: {exc}",
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

    materials = document.get("materials", [])
    if not isinstance(materials, list):
        raise RuntimeError("GLB material list is invalid.")
    for mesh_index, mesh in enumerate(meshes):
        for primitive_index, primitive in enumerate(
            mesh.get("primitives", [])
        ):
            material_index = primitive.get("material")

            if material_index is None:
                continue

            if (
                not isinstance(material_index, int)
                or material_index < 0
                or material_index >= len(materials)
            ):
                raise RuntimeError(
                    f"GLB mesh {mesh_index}, primitive "
                    f"{primitive_index} references an invalid "
                    f"material index: {material_index!r}."
                )

            attributes = primitive.get("attributes", {})

            if not isinstance(attributes, dict):
                raise RuntimeError(
                    f"GLB mesh {mesh_index}, primitive "
                    f"{primitive_index} has invalid attributes."
                )

            material = materials[material_index]
            texture_infos = []

            pbr = material.get("pbrMetallicRoughness", {})

            if isinstance(pbr, dict):
                for key in (
                    "baseColorTexture",
                    "metallicRoughnessTexture",
                ):
                    info = pbr.get(key)
                    if isinstance(info, dict):
                        texture_infos.append(info)

            for key in (
                "normalTexture",
                "occlusionTexture",
                "emissiveTexture",
            ):
                info = material.get(key)
                if isinstance(info, dict):
                    texture_infos.append(info)

            for info in texture_infos:
                tex_coord = info.get("texCoord", 0)

                if (
                    not isinstance(tex_coord, int)
                    or tex_coord < 0
                ):
                    raise RuntimeError(
                        f"GLB material {material_index} contains "
                        f"invalid texCoord: {tex_coord!r}."
                    )

                attribute_name = f"TEXCOORD_{tex_coord}"

                if attribute_name not in attributes:
                    raise RuntimeError(
                        f"GLB mesh {mesh_index}, primitive "
                        f"{primitive_index}, material "
                        f"{material_index} references "
                        f"{attribute_name}, but the mesh does not "
                        "contain that UV attribute."
                    )

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

    return _metadata_resolve_dependency_path(
        game_root,
        relative.as_posix(),
        model_version,
    )


def _dependency_candidates(game_root, relative_path, desired_version):
    """Return dependency candidates newest-to-oldest in the requested history."""
    relative = PurePosixPath(
        str(relative_path).replace("\\", "/")
    )

    if relative.is_absolute() or ".." in relative.parts:
        return []

    return _metadata_dependency_candidates(
        game_root,
        relative.as_posix(),
        desired_version,
    )


def _install_sourceio_frame_animation_compatibility():
    """Decode constant and per-frame Source 1 animation channels together."""
    import numpy as np

    from SourceIO.library.models.mdl.structs.compressed_vectors import (
        Quat48,
        Quat48S,
    )
    from SourceIO.library.models.mdl.structs.frame_anim import StudioFrameAnim
    from SourceIO.library.models.mdl.structs.local_animation import (
        ANIM_DTYPE,
        AnimDescFlags,
        AniBoneFlags,
        StudioAnimDesc,
    )

    def read_frame_animations(self, buffer, bones, section_frame_count):
        try:
            frame_count = max(0, int(section_frame_count))
            entry_offset = buffer.tell()

            frame_anim = StudioFrameAnim.from_buffer(buffer)
            bone_flags = [
                AniBoneFlags(buffer.read_uint8())
                for _ in bones
            ]

            if frame_count == 0:
                return None

            has_constants = frame_anim.constant_offset > 0
            has_frames = (
                frame_anim.frame_offset > 0
                and frame_anim.frame_length > 0
            )

            if not has_constants and not has_frames:
                return None

            animation_data = {}
            is_delta = bool(self.flags & AnimDescFlags.DELTA)

            def samples_for(bone):
                samples = animation_data.get(bone.name)

                if samples is None:
                    samples = np.zeros(
                        (frame_count,),
                        dtype=ANIM_DTYPE,
                    )

                    if is_delta:
                        samples["pos"] = (0.0, 0.0, 0.0)
                        samples["rot"] = (0.0, 0.0, 0.0, 1.0)
                    else:
                        samples["pos"] = tuple(
                            float(v) for v in bone.position
                        )
                        samples["rot"] = tuple(
                            float(v) for v in bone.quat
                        )

                    animation_data[bone.name] = samples

                return samples

            if not is_delta:
                for bone in bones:
                    samples_for(bone)

            if has_constants:
                buffer.seek(entry_offset + frame_anim.constant_offset)

                for bone in bones:
                    flag = bone_flags[bone.bone_id]
                    rotation = None
                    position = None

                    if flag & AniBoneFlags.CONST_ROT2:
                        rotation = Quat48S.read(buffer)

                    if flag & AniBoneFlags.RAW_ROT:
                        rotation = Quat48.read(buffer)

                    if flag & AniBoneFlags.RAW_POS:
                        position = buffer.read_fmt("3e")

                    if flag & AniBoneFlags.CONST_POS2:
                        position = buffer.read_fmt("3f")

                    if rotation is not None or position is not None:
                        samples = samples_for(bone)

                        if rotation is not None:
                            samples["rot"] = rotation

                        if position is not None:
                            samples["pos"] = position

            if has_frames:
                frame_length = int(frame_anim.frame_length)

                for frame_index in range(frame_count):
                    frame_start = (
                        entry_offset
                        + frame_anim.frame_offset
                        + frame_index * frame_length
                    )
                    buffer.seek(frame_start)

                    for bone in bones:
                        flag = bone_flags[bone.bone_id]
                        rotation = None
                        position = None

                        if flag & AniBoneFlags.ANIM_ROT2:
                            rotation = Quat48S.read(buffer)

                        if flag & AniBoneFlags.ANIM_ROT:
                            rotation = Quat48.read(buffer)

                        if flag & AniBoneFlags.ANIM_POS:
                            position = buffer.read_fmt("3e")

                        if flag & AniBoneFlags.FULL_ANIM_POS:
                            position = buffer.read_fmt("3f")

                        if rotation is not None or position is not None:
                            samples = samples_for(bone)

                            if rotation is not None:
                                samples[frame_index]["rot"] = rotation

                            if position is not None:
                                samples[frame_index]["pos"] = position

                    consumed = buffer.tell() - frame_start
                    if consumed > frame_length:
                        raise ValueError(
                            f"decoded frame uses {consumed} bytes, "
                            f"but frame_length is {frame_length}"
                        )

            return animation_data or None

        except Exception as exc:
            print(
                "[model] WARNING: Could not decode frame animation "
                f"{getattr(self, 'name', '<unnamed>')!r}: "
                f"{type(exc).__name__}: {exc}. "
                "This animation will be skipped.",
                flush=True,
            )
            return None

    StudioAnimDesc._read_frame_animations = read_frame_animations
    StudioAnimDesc._vct_frame_animation_compatibility = True

    original_read_animation_frames = (
        StudioAnimDesc._read_animation_frames
    )

    if not getattr(
        original_read_animation_frames,
        "_vct_default_samples_installed",
        False,
    ):

        def read_animation_frames_with_defaults(
            self,
            buffer,
            bones,
            section_frame_count,
        ):
            frame_count = max(0, int(section_frame_count))

            if frame_count == 0:
                return None

            decoded = original_read_animation_frames(
                self,
                buffer,
                bones,
                frame_count,
            )

            is_delta = bool(self.flags & AnimDescFlags.DELTA)

            has_sections = bool(
                self.section_offset
                and self.section_frame_count
            )
            if decoded is None and not has_sections:
                return None

            result = {}

            for bone in bones:
                samples = np.zeros(
                    (frame_count,),
                    dtype=ANIM_DTYPE,
                )

                if is_delta:
                    samples["pos"] = (0.0, 0.0, 0.0)
                    samples["rot"] = (0.0, 0.0, 0.0, 1.0)
                else:
                    samples["pos"] = tuple(
                        float(v) for v in bone.position
                    )
                    samples["rot"] = tuple(
                        float(v) for v in bone.quat
                    )

                if decoded:
                    bone_samples = decoded.get(bone.name)

                    if bone_samples is not None:
                        copy_count = min(
                            frame_count,
                            len(bone_samples),
                        )
                        samples[:copy_count] = (
                            bone_samples[:copy_count]
                        )

                result[bone.name] = samples

            return result

        read_animation_frames_with_defaults._vct_default_samples_installed = (
            True
        )
        StudioAnimDesc._read_animation_frames = (
            read_animation_frames_with_defaults
        )


def _load_mdl_for_pair_check(mdl_path):
    """Parse the MDL with the same SourceIO parser used for model import."""
    _install_sourceio_frame_animation_compatibility()

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


def _install_sourceio_vtx_index_recovery():
    """Install a worker-local shim that drops triangles with invalid VVD indices."""
    import importlib

    common = importlib.import_module(
        "SourceIO.blender_bindings.models.common"
    )
    current_merge = common.merge_meshes

    if getattr(current_merge, "_vct_recovery_wrapper", False):
        recovery_merge = current_merge
        original_merge = current_merge._vct_original_merge
    else:
        original_merge = current_merge

        def recovery_merge(model, vtx_model):
            vtx_vertices, indices, materials = original_merge(
                model, vtx_model
            )

            if not _VCT_VTX_INDEX_RECOVERY_ACTIVE:
                return vtx_vertices, indices, materials

            import numpy as np

            source_vertices = np.asarray(
                vtx_vertices, dtype=np.int64
            ).reshape(-1)
            render_indices = np.asarray(
                indices, dtype=np.int64
            ).reshape(-1)
            material_indices = np.asarray(
                materials
            ).reshape(-1)

            if render_indices.size % 3:
                raise RuntimeError(
                    f"Cannot recover VTX geometry for {model.name!r}: "
                    "the index buffer contains incomplete triangles."
                )

            triangles = render_indices.reshape((-1, 3))

            invalid_source = (
                (source_vertices < 0)
                | (source_vertices >= int(model.vertex_count))
            )
            invalid_render = (
                (render_indices < 0)
                | (render_indices >= len(source_vertices))
            )

            if not invalid_source.any() and not invalid_render.any():
                return vtx_vertices, indices, materials

            if len(material_indices) != len(triangles):
                raise RuntimeError(
                    f"Cannot recover VTX geometry for {model.name!r}: "
                    "triangle and material counts do not match."
                )

            valid_triangles = np.all(
                (triangles >= 0)
                & (triangles < len(source_vertices)),
                axis=1,
            )

            candidate_rows = np.flatnonzero(valid_triangles)

            if candidate_rows.size:
                mapped_vertices = source_vertices[
                    triangles[candidate_rows]
                ]
                valid_triangles[candidate_rows] = np.all(
                    (mapped_vertices >= 0)
                    & (mapped_vertices < int(model.vertex_count)),
                    axis=1,
                )

            retained = triangles[valid_triangles]

            if retained.size == 0:
                raise RuntimeError(
                    f"Cannot recover VTX geometry for {model.name!r}: "
                    "no valid triangles remain after filtering invalid "
                    "VVD/VTX indices."
                )

            used_render_vertices = np.unique(retained)
            remap = np.full(
                len(source_vertices), -1, dtype=np.int64
            )
            remap[used_render_vertices] = np.arange(
                len(used_render_vertices), dtype=np.int64
            )

            repaired_vertices = source_vertices[
                used_render_vertices
            ].tolist()
            repaired_indices = remap[retained].reshape(-1)
            repaired_materials = material_indices[valid_triangles]

            print(
                f"[model] WARNING: Recovered mismatched VVD/VTX data "
                f"for {model.name!r}; discarded "
                f"{int((~valid_triangles).sum())} invalid triangle(s).",
                flush=True,
            )

            return (
                repaired_vertices,
                repaired_indices,
                repaired_materials,
            )

        recovery_merge._vct_recovery_wrapper = True
        recovery_merge._vct_original_merge = original_merge
        common.merge_meshes = recovery_merge

    for module in tuple(sys.modules.values()):
        if module is None:
            continue

        module_name = getattr(module, "__name__", "")
        if not module_name.startswith(
            "SourceIO.blender_bindings.models."
        ):
            continue

        if getattr(module, "merge_meshes", None) is original_merge:
            module.merge_meshes = recovery_merge


@contextmanager
def sourceio_vtx_index_recovery(enabled):
    """Enable the worker-local recovery shim for one SourceIO import."""
    global _VCT_VTX_INDEX_RECOVERY_ACTIVE

    if not enabled:
        yield
        return

    _install_sourceio_vtx_index_recovery()
    previous = _VCT_VTX_INDEX_RECOVERY_ACTIVE
    _VCT_VTX_INDEX_RECOVERY_ACTIVE = True

    try:
        yield
    finally:
        _VCT_VTX_INDEX_RECOVERY_ACTIVE = previous


def _check_vvd_vtx_compatibility(mdl, vvd_path, vtx_path):
    """Validate a pair and identify recoverable structural/index mismatches."""
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

    total_mdl_models = sum(
        1
        for body_part in mdl.body_parts
        for model in body_part.models
        if model.vertex_count > 0
    )

    info = {
        "invalid_vertex_references": 0,
        "total_vertex_references": 0,
        "invalid_render_indices": 0,
        "total_render_indices": 0,
        "invalid_triangles": 0,
        "total_triangles": 0,
        "missing_mdl_body_parts": max(
            len(mdl.body_parts) - len(vtx.body_parts), 0
        ),
        "extra_vtx_body_parts": max(
            len(vtx.body_parts) - len(mdl.body_parts), 0
        ),
        "missing_mdl_models": 0,
        "extra_vtx_models": 0,
        "total_mdl_body_parts": len(mdl.body_parts),
        "total_mdl_models": total_mdl_models,
    }

    structural_reasons = []
    recovery_reasons = []
    checked_models = 0

    if len(vtx.body_parts) != len(mdl.body_parts):
        if len(vtx.body_parts) < len(mdl.body_parts):
            for body_part in mdl.body_parts[len(vtx.body_parts):]:
                info["missing_mdl_models"] += sum(
                    1
                    for model in body_part.models
                    if model.vertex_count > 0
                )

            structural_reasons.append(
                f"VTX has {len(vtx.body_parts)} body part(s), but MDL "
                f"has {len(mdl.body_parts)}; unmatched MDL body parts "
                "will be omitted"
            )
        else:
            structural_reasons.append(
                f"VTX has {len(vtx.body_parts)} body part(s), but MDL "
                f"has {len(mdl.body_parts)}; extra VTX body parts "
                "will be ignored"
            )

    for vtx_body_part, body_part in zip(
        vtx.body_parts, mdl.body_parts
    ):
        vtx_models = vtx_body_part.models
        mdl_models = body_part.models

        if len(vtx_models) != len(mdl_models):
            if len(vtx_models) < len(mdl_models):
                missing_models = mdl_models[len(vtx_models):]
                info["missing_mdl_models"] += sum(
                    1
                    for model in missing_models
                    if model.vertex_count > 0
                )
            else:
                info["extra_vtx_models"] += (
                    len(vtx_models) - len(mdl_models)
                )

            structural_reasons.append(
                f"VTX has {len(vtx_models)} model slot(s), but MDL has "
                f"{len(mdl_models)} for body part {body_part.name!r}; "
                "unmatched entries will be omitted"
            )

        for vtx_model, model in zip(vtx_models, mdl_models):
            if model.vertex_count == 0:
                continue

            if not vtx_model.model_lods:
                return False, (
                    f"VTX model {model.name!r} has no LOD data"
                ), None

            start = model.vertex_offset
            end = start + model.vertex_count
            model_vertices = all_vertices[start:end]

            if len(model_vertices) != model.vertex_count:
                return False, (
                    f"MDL model {model.name!r} requests vertices "
                    f"[{start}:{end}], but the VVD does not contain "
                    "that range"
                ), None

            try:
                vtx_vertices, indices, _ = merge_meshes(
                    model, vtx_model.model_lods[0]
                )
            except Exception as exc:
                return False, (
                    f"Could not build VTX indices for {model.name!r}: "
                    f"{type(exc).__name__}: {exc}"
                ), None

            vtx_vertices = [int(value) for value in vtx_vertices]
            indices = [int(value) for value in indices]

            if len(indices) % 3:
                return False, (
                    f"VTX index buffer for {model.name!r} does not "
                    "contain complete triangles"
                ), None

            invalid_vertices = sum(
                value < 0 or value >= len(model_vertices)
                for value in vtx_vertices
            )
            invalid_indices = sum(
                value < 0 or value >= len(vtx_vertices)
                for value in indices
            )

            triangle_count = len(indices) // 3
            valid_triangle_count = 0

            for offset in range(0, len(indices), 3):
                triangle_valid = True

                for render_index in indices[offset:offset + 3]:
                    if (
                        render_index < 0
                        or render_index >= len(vtx_vertices)
                    ):
                        triangle_valid = False
                        continue

                    vertex_index = vtx_vertices[render_index]
                    if (
                        vertex_index < 0
                        or vertex_index >= len(model_vertices)
                    ):
                        triangle_valid = False

                if triangle_valid:
                    valid_triangle_count += 1

            info["invalid_vertex_references"] += invalid_vertices
            info["total_vertex_references"] += len(vtx_vertices)
            info["invalid_render_indices"] += invalid_indices
            info["total_render_indices"] += len(indices)
            info["invalid_triangles"] += (
                triangle_count - valid_triangle_count
            )
            info["total_triangles"] += triangle_count
            checked_models += 1

            if invalid_vertices or invalid_indices:
                recovery_reasons.append(
                    f"VTX/VVD index mismatch for {model.name!r}: "
                    f"{invalid_vertices} invalid vertex reference(s), "
                    f"{invalid_indices} invalid render index value(s)"
                )

                if valid_triangle_count == 0:
                    return False, (
                        f"VTX/VVD mismatch for {model.name!r} leaves no "
                        "valid triangles to import"
                    ), None

    if structural_reasons or recovery_reasons:
        if info["total_triangles"] == 0:
            return False, (
                "; ".join(structural_reasons + recovery_reasons)
                + "; no usable triangle geometry exists in the "
                "overlapping models"
            ), None

        reasons = structural_reasons + recovery_reasons
        return False, "; ".join(reasons), info

    return True, f"checked {checked_models} model(s)", None


def _select_compatible_model_dependencies(
    physical_model_path,
    game_root,
    model_relative_to_game,
    model_version,
):
    """Prefer a fully compatible pair; otherwise choose the safest recoverable pair."""
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
            game_root, relative, model_version
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
        return selected_vvd, selected_vtx, False

    if not vvd_candidates:
        raise FileNotFoundError(
            f"No eligible VVD dependency found for {physical_model_path} "
            f"at model version {model_version}"
        )

    if mdl is None:
        return vvd_candidates[0], vtx_candidates[0], False

    attempted = []
    recovery_candidates = []

    for vtx_candidate in vtx_candidates:
        for vvd_candidate in vvd_candidates:
            vtx_suffix, vtx_version, vtx_path = vtx_candidate
            vvd_version, vvd_path = vvd_candidate

            try:
                compatible, reason, recovery_info = (
                    _check_vvd_vtx_compatibility(
                        mdl, vvd_path, vtx_path
                    )
                )
            except Exception as exc:
                compatible = False
                reason = f"{type(exc).__name__}: {exc}"
                recovery_info = None

            if compatible:
                return vvd_candidate, vtx_candidate, False

            attempted.append(
                f"VTX {vtx_path} + VVD {vvd_path}: {reason}"
            )

            if recovery_info is not None:
                invalid_triangles = recovery_info["invalid_triangles"]
                total_triangles = recovery_info["total_triangles"]
                invalid_vertices = recovery_info[
                    "invalid_vertex_references"
                ]
                total_vertices = recovery_info[
                    "total_vertex_references"
                ]
                invalid_indices = recovery_info[
                    "invalid_render_indices"
                ]
                total_indices = recovery_info["total_render_indices"]

                structural_loss = max(
                    recovery_info["missing_mdl_models"]
                    / max(recovery_info["total_mdl_models"], 1),
                    recovery_info["missing_mdl_body_parts"]
                    / max(recovery_info["total_mdl_body_parts"], 1),
                )

                triangle_loss = (
                    invalid_triangles / max(total_triangles, 1)
                )

                index_loss = (
                    invalid_vertices + invalid_indices
                ) / max(total_vertices + total_indices, 1)

                score = (
                    structural_loss + triangle_loss,
                    structural_loss,
                    triangle_loss,
                    index_loss,
                    invalid_triangles,
                    invalid_vertices,
                    invalid_indices,
                )

                recovery_candidates.append((
                    score,
                    vvd_candidate,
                    vtx_candidate,
                    reason,
                    recovery_info,
                ))

    if recovery_candidates:
        (
            _score,
            selected_vvd,
            selected_vtx,
            reason,
            info,
        ) = min(recovery_candidates, key=lambda item: item[0])

        print(
            f"[model] WARNING: No fully compatible VVD/VTX pair exists "
            f"for {physical_model_path}. Using the least-damaged "
            "recoverable pair. "
            f"Invalid triangles: {info['invalid_triangles']} of "
            f"{info['total_triangles']}; "
            f"omitted MDL body parts: "
            f"{info['missing_mdl_body_parts']}; "
            f"omitted MDL model slots: "
            f"{info['missing_mdl_models']}; "
            f"ignored extra VTX body parts: "
            f"{info['extra_vtx_body_parts']}. "
            f"Details: {reason}",
            flush=True,
        )
        return selected_vvd, selected_vtx, True

    details = "\n".join(f"  - {line}" for line in attempted[:8])

    raise RuntimeError(
        f"No compatible or safely recoverable VVD/VTX pair was found "
        f"for {physical_model_path} at model version {model_version}.\n"
        f"Tested {len(attempted)} candidate pair(s).\n{details}"
    )


def get_versioned_game_context(input_root, input_path):
    """Return the physical input, game root, logical path, and history context."""
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

    selection_context = get_selection_context(
        archive_root,
        source_path,
        desired_version,
    )

    return (
        source_path,
        game_root,
        relative_to_game,
        selection_context,
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

    selected_vvd, selected_vtx, recover_vtx_indices = (
        _select_compatible_model_dependencies(
            physical_model_path,
            game_root,
            model_relative_to_game,
            model_version,
        )
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

        return (
            temporary,
            staged_model,
            game_root,
            model_version,
            recover_vtx_indices,
        )

    except Exception:
        _cleanup_staging_directory(temporary)
        raise

     
def load_source1_animation_data(mdl, mdl_buffer, content_manager, model_path):
    """Decode Source 1 animations directly through SourceIO's data parser."""
    from SourceIO.library.models.mdl.load_animations import (
        load_animations_from_mdl,
    )
    from SourceIO.library.utils.tiny_path import TinyPath
    from SourceIO.library.models.mdl.structs.local_animation import AnimDescFlags

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
        post_delta_by_name = {
            str(descriptor.name).casefold(): bool(
                descriptor.flags & AnimDescFlags.POST
            )
            for descriptor in descriptors
        }
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
                    animations.append((
                        logical_path.as_posix(),
                        animation,
                        post_delta_by_name.get(
                            str(animation.name).casefold(),
                            False,
                        ),
                    ))

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


def install_gltf_json_finite_guard():
    """Prevent Blender's glTF JSON serializer from crashing on NaN/Inf."""
    import importlib

    gltf_export = None

    for module_name in (
        "io_scene_gltf2.blender.exp.export",
        "bl_ext.blender_org.io_scene_gltf2.blender.exp.export",
    ):
        try:
            gltf_export = importlib.import_module(module_name)
            break
        except ImportError:
            continue

    if gltf_export is None:
        raise RuntimeError(
            "Could not locate Blender's glTF exporter module."
        )

    original_fix_json = gltf_export.__fix_json

    if getattr(original_fix_json, "_valveconvert_finite_guard", False):
        return

    state = {"inside": False}

    def sanitize_json(value, path="$", key=None, bad=None):
        if isinstance(value, float):
            if math.isfinite(value):
                return value

            bad[0] += 1
            if len(bad[1]) < 8:
                bad[1].append(path)

            return 1.0 if key == "scale" else 0.0

        if isinstance(value, dict):
            result = {}

            for child_key, child_value in value.items():
                child_path = f"{path}.{child_key}"

                if (
                    child_key == "rotation"
                    and isinstance(child_value, (list, tuple))
                    and any(
                        isinstance(component, float)
                        and not math.isfinite(component)
                        for component in child_value
                    )
                ):
                    bad[0] += 1
                    if len(bad[1]) < 8:
                        bad[1].append(child_path)

                    result[child_key] = [0.0, 0.0, 0.0, 1.0]
                else:
                    result[child_key] = sanitize_json(
                        child_value,
                        child_path,
                        child_key,
                        bad,
                    )

            return result

        if isinstance(value, (list, tuple)):
            return [
                sanitize_json(
                    item,
                    f"{path}[{index}]",
                    key,
                    bad,
                )
                for index, item in enumerate(value)
            ]

        return value

    def finite_fix_json(obj, export_settings):
        if state["inside"]:
            return original_fix_json(obj, export_settings)

        state["inside"] = True
        bad = [0, []]

        try:
            cleaned = sanitize_json(obj, bad=bad)

            if bad[0]:
                print(
                    f"[gltf] Replaced {bad[0]} non-finite JSON value(s). "
                    f"First affected paths: {', '.join(bad[1])}",
                    flush=True,
                )

            return original_fix_json(cleaned, export_settings)
        finally:
            state["inside"] = False

    finite_fix_json._valveconvert_finite_guard = True
    gltf_export.__fix_json = finite_fix_json


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
        if material.node_tree is None:
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
    from mathutils import Matrix, Quaternion, Vector
    from SourceIO.blender_bindings.utils.bpy_utils import ActionCurveFactory

    armatures = [
        obj for obj in armature_objects
        if obj.type == "ARMATURE" and obj.data is not None
    ]
    if not armatures:
        return []

    created_actions = []

    for animation_source in animation_sources:
        if len(animation_source) == 3:
            model_path, animation, post_delta = animation_source
        else:
            model_path, animation = animation_source
            post_delta = False

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
        action.use_fake_user = True

        source_fps = _finite_number(
            getattr(animation, "fps", 30.0),
            30.0,
        )
        if source_fps <= 0.0:
            source_fps = 30.0

        action["source_fps"] = source_fps

        scene = bpy.context.scene
        fps_base = _finite_number(scene.render.fps_base, 1.0)
        if fps_base <= 0.0:
            fps_base = 1.0

        gltf_frame_rate = (
            _finite_number(scene.render.fps, 30.0) / fps_base
        )
        frame_scale = gltf_frame_rate / source_fps

        factory = ActionCurveFactory(action, armature)
        curve_count = 0

        for bone_name, frames in bone_frames.items():
            bone = armature.data.bones.get(bone_name)
            pose_bone = armature.pose.bones.get(bone_name)
            if bone is None or pose_bone is None or frames is None:
                continue

            pose_bone.rotation_mode = "QUATERNION"

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
            rest_local = (
                parent_rest.inverted() @ bone.matrix_local
                if parent is not None
                else bone.matrix_local.copy()
            )

            rest_position = rest_local.to_translation()
            rest_quaternion = rest_local.to_quaternion()

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

                is_delta = bool(
                    getattr(animation, "is_delta", False)
                )

                if is_delta:
                    position = rest_position + position
                    if post_delta:
                        quaternion = rest_quaternion @ quaternion
                    else:
                        quaternion = quaternion @ rest_quaternion

                animated_local = (
                    Matrix.Translation(position)
                    @ quaternion.to_matrix().to_4x4()
                )

                armature_space = parent_rest @ animated_local
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
    from SourceIO.library.utils.math_utilities import SOURCE1_HAMMER_UNIT_TO_METERS

    ensure_parent(output_path)

    (
        temporary,
        staged_model,
        game_root,
        model_version,
        recover_vtx_indices,
    ) = stage_versioned_model(input_root, input_path)

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
            with sourceio_vtx_index_recovery(recover_vtx_indices):
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
    install_gltf_json_finite_guard()

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
        print(
            f"[model] WARNING: Found {animation_descriptor_count} "
            "animation descriptor(s), but none could be decoded. "
            "Exporting the model without animation.",
            flush=True,
        )
        imported_actions = []
    else:
        imported_actions = create_source1_animation_actions(
            armature_objects,
            animation_sources,
            scale=SOURCE1_HAMMER_UNIT_TO_METERS,
        )

        if (
            animation_descriptor_count > 0
            and armature_objects
            and animation_sources
            and not imported_actions
        ):
            print(
                "[model] WARNING: Animation data was decoded, but no "
                "clips could be bound to the armature. "
                "Exporting the model without animation.",
                flush=True,
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
        export_image_format="WEBP",
        export_image_quality=100,
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
        if material.node_tree is not None
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

    resolved_root, resolved_filename, _ = (
        DEFAULT_VERSIONED_CONTENT_VIEWS.resolve(
            input_root,
            relative_path,
        )
    )

    resolved_input_path = os.path.join(
        str(resolved_root),
        *PurePosixPath(resolved_filename).parts,
    )

    _, game_root, _, map_version = get_versioned_game_context(
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